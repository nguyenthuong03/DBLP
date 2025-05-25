#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/12/19
# @Author  : AE-CP-GNN Extension
# @File    : main_AE.py
# @Software: PyCharm
# @Describe: Main training script for Attribute Enhanced CP-GNN

from statistics import mean
import torch
import dgl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import argparse

from models import AE_ContextGNN, create_ae_model_config
from evaluate import evaluate_task
from utils import load_data, EarlyStopping, load_latest_model, evaluate

parser = argparse.ArgumentParser(description='AE-CP-GNN Training')
parser.add_argument('-n', default=0, type=int, help='GPU ID')
parser.add_argument('-alpha', default=0.5, type=float, help='Alpha parameter for combining embeddings')
parser.add_argument('--grid_search', action='store_true', help='Perform grid search over alpha values')
args = parser.parse_args()


def train_ae_model(config, alpha=0.5):
    """
    Train the AE-CP-GNN model with specified alpha value.
    
    Args:
        config: Configuration object
        alpha: Weighting parameter for combining embeddings
        
    Returns:
        Final model and results
    """
    print(f"\n=== Training AE-CP-GNN with alpha={alpha} ===")
    
    # Update model config with alpha
    config.model_config = create_ae_model_config(
        config.model_config, 
        alpha=alpha,
        max_text_length=128,
        freeze_bert=True
    )
    
    # Load data
    dataloader = load_data(config.data_config)
    if config.data_config['dataset'] in ['AIFB', 'AM', 'BGS', 'MUTAG']:
        config.data_config['primary_type'] = dataloader.predict_category
        config.model_config['primary_type'] = dataloader.predict_category
    
    hg = dataloader.heter_graph
    edges_data_dict = dataloader.load_train_k_context_edges(
        hg, config.data_config['K_length'],
        config.data_config['primary_type'],
        config.train_config['pos_num_for_each_hop'],
        config.train_config['neg_num_for_each_hop']
    )

    CF_data = dataloader.load_classification_data()
    device = torch.device('cuda:{}'.format(args.n) if torch.cuda.is_available() else 'cpu')
    
    # Create data loaders
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

    # Initialize AE-CP-GNN model
    model = AE_ContextGNN(hg, config.model_config, config.data_config)
    model = model.to(device)
    
    print(f"Model device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Load checkpoint if continuing training
    if config.train_config['continue']:
        model = load_latest_model(config.train_config['checkpoint_path'], model)

    # Update checkpoint path to include alpha
    checkpoint_path_alpha = config.train_config['checkpoint_path'] + f'_alpha_{alpha}'
    stopper = EarlyStopping(
        checkpoint_path=checkpoint_path_alpha, 
        config=config,
        patience=config.train_config['patience']
    )

    # Optimizer and scheduler
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
    
    print("Start training AE-CP-GNN...")
    for epoch in range(config.train_config['total_epoch']):
        model.train()
        running_loss = []
        
        for k_hop, dataloader in dataloader_dict.items():
            
            for pos_src, pos_dst, neg_src, neg_dst in dataloader:
                # Move data to device
                pos_src, pos_dst = pos_src.to(device), pos_dst.to(device)
                neg_src, neg_dst = neg_src.to(device), neg_dst.to(device)
                
                # Forward pass with AE-CP-GNN
                p_context_emb = model(k_hop).detach()  # Get combined embeddings
                p_emb = model.primary_emb.weight  # Original primary embeddings
                
                # Compute loss
                loss = model.get_loss(k_hop, pos_src, pos_dst, neg_src, neg_dst, p_emb, p_context_emb)
                running_loss.append(loss.item())
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        mean_loss = mean(running_loss)
        scheduler.step(mean_loss)
        
        print(f"Epoch: {epoch}/{config.train_config['total_epoch']}, Loss: {mean_loss:.6f}, Alpha: {alpha}")
        
        # Early stopping check
        early_stop = stopper.step(mean_loss, model)
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            p_emb = model(config.data_config['K_length']).detach().cpu().numpy()  # Use combined embeddings
            evaluate(
                p_emb, CF_data, None, 
                config.evaluate_config['method'], 
                metric=config.data_config['task'],
                random_state=config.evaluate_config['random_state'],
                max_iter=config.evaluate_config['max_iter'], 
                n_jobs=config.evaluate_config['n_jobs']
            )
        
        if early_stop:
            break

    # Final evaluation
    checkpoint_path = stopper.filepath
    print(f"Training completed. Best model saved at: {checkpoint_path}")
    
    return checkpoint_path, mean_loss


def grid_search_alpha(config):
    """
    Perform grid search over alpha values.
    
    Args:
        config: Configuration object
        
    Returns:
        Best alpha value and corresponding results
    """
    from config_AE import alpha_values
    
    best_alpha = 0.5
    best_loss = float('inf')
    results = {}
    
    print("=== Starting Grid Search for Alpha ===")
    
    for alpha in alpha_values:
        try:
            checkpoint_path, final_loss = train_ae_model(config, alpha)
            results[alpha] = {
                'checkpoint_path': checkpoint_path,
                'final_loss': final_loss
            }
            
            if final_loss < best_loss:
                best_loss = final_loss
                best_alpha = alpha
                
            print(f"Alpha {alpha}: Final Loss = {final_loss:.6f}")
            
        except Exception as e:
            print(f"Error training with alpha {alpha}: {e}")
            results[alpha] = {'error': str(e)}
    
    print(f"\n=== Grid Search Results ===")
    for alpha, result in results.items():
        if 'error' not in result:
            print(f"Alpha {alpha}: Loss = {result['final_loss']:.6f}")
        else:
            print(f"Alpha {alpha}: Error = {result['error']}")
    
    print(f"\nBest Alpha: {best_alpha} with Loss: {best_loss:.6f}")
    
    return best_alpha, results


def main(config):
    """Main training function."""
    if args.grid_search:
        # Perform grid search
        best_alpha, results = grid_search_alpha(config)
        
        # Train final model with best alpha
        print(f"\n=== Training Final Model with Best Alpha = {best_alpha} ===")
        checkpoint_path, _ = train_ae_model(config, best_alpha)
        
    else:
        # Train with single alpha value
        checkpoint_path, _ = train_ae_model(config, args.alpha)
    
    # Final evaluation
    print(f"\n=== Final Evaluation ===")
    evaluate_task(config, checkpoint_path)


if __name__ == "__main__":
    import config_AE
    
    print("=== AE-CP-GNN Training Script ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"DGL version: {dgl.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(args.n)}")
    
    main(config_AE) 