#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/12/19
# @Author  : Dynamic CP-GNN Implementation
# @File    : main_dynamic.py
# @Software: PyCharm
# @Describe: Main training script for Dynamic CP-GNN (D-CP-GNN)

from statistics import mean
import torch
import dgl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from models import DynamicContextGNN
from utils import load_dynamic_data, EarlyStopping, load_latest_model, evaluate
import argparse

parser = argparse.ArgumentParser(description='Dynamic CP-GNN Training')
parser.add_argument('-n', default=0, type=int, help='GPU ID')
parser.add_argument('-s', '--snapshots', default=5, type=int, help='Number of temporal snapshots')
args = parser.parse_args()


def main(config):
    """
    Main training function for Dynamic CP-GNN
    """
    print("=" * 60)
    print("Dynamic CP-GNN (D-CP-GNN) Training")
    print("=" * 60)
    
    # Load dynamic data with temporal snapshots
    print(f"Loading dynamic DBLP data with {config.dynamic_config['num_snapshots']} snapshots...")
    dataloader = load_dynamic_data(config.data_config, num_snapshots=config.dynamic_config['num_snapshots'])
    
    # Get snapshots
    snapshots = dataloader.get_all_snapshots()
    print(f"Created {len(snapshots)} temporal snapshots:")
    for i, snapshot in enumerate(snapshots):
        print(f"  Snapshot {i+1} ({config.dynamic_config['snapshot_years'][i]}): "
              f"{snapshot.number_of_nodes('p')} papers, "
              f"{snapshot.number_of_nodes('a')} authors, "
              f"{snapshot.number_of_nodes('c')} conferences, "
              f"{snapshot.number_of_nodes('t')} terms")
    
    # Update config for dynamic model
    if config.data_config['dataset'] in ['AIFB', 'AM', 'BGS', 'MUTAG']:
        config.data_config['primary_type'] = dataloader.predict_category
        config.model_config['primary_type'] = dataloader.predict_category
    
    # Load classification data (from final snapshot)
    CF_data = dataloader.load_classification_data()
    print(f"Classification data loaded: {len(CF_data[3])} train samples, {len(CF_data[4])} test samples")
    
    # Set device
    device = torch.device('cuda:{}'.format(args.n) if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create Dynamic CP-GNN model
    print("Initializing Dynamic CP-GNN model...")
    model = DynamicContextGNN(snapshots, config.model_config)
    model = model.to(device)
    
    # Load checkpoint if continuing training
    if config.train_config['continue']:
        model = load_latest_model(config.train_config['checkpoint_path'], model)
    
    # Initialize training components
    stopper = EarlyStopping(
        checkpoint_path=config.train_config['checkpoint_path'], 
        config=config,
        patience=config.train_config['patience']
    )
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.train_config['lr'], 
        weight_decay=config.train_config['l2']
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer, 'min', 
        factor=config.train_config['factor'], 
        patience=config.train_config['patience'] // 3, 
        verbose=True
    )
    
    print("=" * 60)
    print("Starting Dynamic CP-GNN Training...")
    print("=" * 60)
    
    # Training loop
    for epoch in range(config.train_config['total_epoch']):
        model.train()
        running_loss = []
        
        # Train on each snapshot's context edges
        for snapshot_idx in range(len(snapshots)):
            # Load edges for this snapshot
            edges_data_dict = dataloader.load_train_k_context_edges_for_snapshot(
                snapshot_idx,
                config.data_config['K_length'],
                config.data_config['primary_type'],
                config.train_config['pos_num_for_each_hop'],
                config.train_config['neg_num_for_each_hop']
            )
            
            # Create dataloaders for this snapshot
            dataloader_dict = {
                key: DataLoader(
                    dataset, 
                    batch_size=config.train_config['batch_size'],
                    num_workers=config.train_config['sample_workers'], 
                    collate_fn=dataset.collate,
                    shuffle=True, 
                    pin_memory=True
                )
                for key, dataset in edges_data_dict.items()
                if len(dataset) > 0
            }
            
            # Train on this snapshot's edges
            for k_hop, dataloader_iter in dataloader_dict.items():
                for pos_src, pos_dst, neg_src, neg_dst in dataloader_iter:
                    # Move data to device
                    pos_src = pos_src.to(device)
                    pos_dst = pos_dst.to(device)
                    neg_src = neg_src.to(device)
                    neg_dst = neg_dst.to(device)
                    
                    # Forward pass through temporal sequence
                    # Get dynamic embeddings from final snapshot
                    p_context_emb = model(k_hop, device)
                    
                    # Get primary embeddings for loss computation
                    final_snapshot = snapshots[-1]
                    num_primary_nodes = final_snapshot.number_of_nodes(config.model_config['primary_type'])
                    p_emb = model.primary_emb.weight[:num_primary_nodes]
                    
                    # Compute loss
                    loss = model.get_loss(k_hop, pos_src, pos_dst, neg_src, neg_dst, p_emb, p_context_emb)
                    running_loss.append(loss.item())
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        
        # Calculate mean loss and update scheduler
        mean_loss = mean(running_loss) if running_loss else float('inf')
        scheduler.step(mean_loss)
        
        # Print progress
        print(f"Epoch: {epoch+1}/{config.train_config['total_epoch']}, Loss: {mean_loss:.6f}")
        
        # Evaluate model
        model.eval()
        with torch.no_grad():
            # Get final dynamic embeddings for evaluation
            final_embeddings = model(config.data_config['K_length'], device)
            p_emb = final_embeddings.detach().cpu().numpy()
            
            # Evaluate on classification task
            evaluate(
                p_emb, CF_data, None, 
                config.evaluate_config['method'], 
                metric=config.data_config['task'],
                random_state=config.evaluate_config['random_state'],
                max_iter=config.evaluate_config['max_iter'], 
                n_jobs=config.evaluate_config['n_jobs']
            )
        
        # Early stopping check
        early_stop = stopper.step(mean_loss, model)
        if early_stop:
            print("Early stopping triggered!")
            break
    
    print("=" * 60)
    print("Training completed! Evaluating final model...")
    print("=" * 60)
    
    # Final evaluation
    checkpoint_path = stopper.filepath
    evaluate_dynamic_task(config, checkpoint_path)
    
    return


def evaluate_dynamic_task(config, checkpoint_path=None):
    """
    Evaluate the trained Dynamic CP-GNN model
    """
    print("Loading data for final evaluation...")
    dataloader = load_dynamic_data(config.data_config, num_snapshots=config.dynamic_config['num_snapshots'])
    snapshots = dataloader.get_all_snapshots()
    
    if config.data_config['dataset'] in ['AIFB', 'AM', 'BGS', 'MUTAG']:
        config.data_config['primary_type'] = dataloader.predict_category
        config.model_config['primary_type'] = dataloader.predict_category
    
    # Load model
    if not checkpoint_path:
        model = DynamicContextGNN(snapshots, config.model_config)
        model = load_latest_model(config.train_config['checkpoint_path'], model)
    else:
        import importlib
        import os
        config_path = os.path.join(checkpoint_path, 'config')
        config_path = os.path.relpath(config_path)
        config_file = config_path.replace(os.sep, '.')
        model_path = os.path.join(checkpoint_path, 'model.pth')
        config = importlib.import_module(config_file)
        model = DynamicContextGNN(snapshots, config.model_config)
        model.load_state_dict(torch.load(model_path, weights_only=True))
    
    # Get final dynamic embeddings
    device = torch.device('cuda:{}'.format(args.n) if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        final_embeddings = model(config.data_config['K_length'], device)
        p_emb = final_embeddings.detach().cpu().numpy()
    
    # Load classification data and evaluate
    CF_data = dataloader.load_classification_data()
    
    result_save_path = evaluate(
        p_emb, CF_data, None, 
        method=config.evaluate_config['method'],  
        metric=config.data_config['task'], 
        save_result=True,
        result_path=config.evaluate_config['result_path'],
        random_state=config.evaluate_config['random_state'],
        max_iter=config.evaluate_config['max_iter'], 
        n_jobs=config.evaluate_config['n_jobs']
    )
    
    if result_save_path:
        from utils.helper import save_config, save_attention_matrix, generate_attention_heat_map
        
        # Save configuration
        save_config(config, result_save_path)
        
        # Save model
        model_save_path = os.path.join(result_save_path, "model.pth")
        torch.save(model.state_dict(), model_save_path)
        
        # Save attention matrices
        attention_matrix_path = save_attention_matrix(model, result_save_path, config.data_config['K_length'])
        if attention_matrix_path and config.evaluate_config['save_heat_map']:
            # Use final snapshot for attention visualization
            final_snapshot = snapshots[-1]
            generate_attention_heat_map(final_snapshot.ntypes, attention_matrix_path)
        
        print(f"Results saved to: {result_save_path}")


if __name__ == "__main__":
    import config_dynamic as config
    
    # Update number of snapshots from command line argument
    if args.snapshots != 5:
        config.data_config['num_snapshots'] = args.snapshots
        config.dynamic_config['num_snapshots'] = args.snapshots
        config.dynamic_config['snapshot_years'] = list(range(2010, 2010 + args.snapshots))
    
    main(config) 