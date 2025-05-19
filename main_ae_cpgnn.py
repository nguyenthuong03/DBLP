#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Describe: Main script for training and evaluating Attribute-enhanced CP-GNN

from statistics import mean
import os
import torch
import dgl
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from models import AttributeEnhancedCGNN
from evaluate import evaluate_task
from utils import load_data, EarlyStopping, load_latest_model, evaluate
from utils.attribute_processor import load_node_attributes

parser = argparse.ArgumentParser(description='Which GPU to run?')
parser.add_argument('-n', default=0, type=int, help='GPU ID')
parser.add_argument('-c', default='config_AE_DBLP.py', type=str, help='Config file path')
args = parser.parse_args()


def main(config):
    print("Loading data...")
    dataloader = load_data(config.data_config)
    if config.data_config['dataset'] in ['AIFB', 'AM', 'BGS', 'MUTAG']:
        config.data_config['primary_type'] = dataloader.predict_category
        config.model_config['primary_type'] = dataloader.predict_category
    
    hg = dataloader.heter_graph
    
    # Load or process node attributes
    print("Processing node attributes...")
    node_attributes = None
    if config.data_config.get('use_node_attributes', False):
        node_attributes = load_node_attributes(config.data_config, hg)
        if node_attributes is None or config.data_config['primary_type'] not in node_attributes:
            print(f"Warning: No attributes found for primary type: {config.data_config['primary_type']}")
    
    # Load edges for training
    edges_data_dict = dataloader.load_train_k_context_edges(hg, config.data_config['K_length'],
                                                          config.data_config['primary_type'],
                                                          config.train_config['pos_num_for_each_hop'],
                                                          config.train_config['neg_num_for_each_hop'])

    # Load classification/link prediction data for evaluation
    CF_data = dataloader.load_classification_data()
    
    # Setup device
    device = torch.device('cuda:{}'.format(args.n) if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup dataloader with batching
    dataloader_dict = {key: DataLoader(dataset, batch_size=config.train_config['batch_size'],
                                     num_workers=config.train_config['sample_workers'], 
                                     collate_fn=dataset.collate,
                                     shuffle=True, pin_memory=True)
                     for key, dataset in edges_data_dict.items() if len(dataset) > 0}

    # Initialize the AE-CP-GNN model
    print("Initializing Attribute-enhanced CP-GNN model...")
    model = AttributeEnhancedCGNN(hg, config.model_config, node_attributes)
    model = model.to(device)

    # Load checkpoint if continuing training
    if config.train_config['continue']:
        model = load_latest_model(config.train_config['checkpoint_path'], model)

    # Setup early stopping
    stopper = EarlyStopping(checkpoint_path=config.train_config['checkpoint_path'], config=config,
                          patience=config.train_config['patience'])

    # Setup optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(),
                               lr=config.train_config['lr'], 
                               weight_decay=config.train_config['l2'])
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=config.train_config['factor'], 
                                patience=config.train_config['patience'] // 3, verbose=True)

    print("Start training...")
    for epoch in range(config.train_config['total_epoch']):
        model.train()
        running_loss = []
        
        for k_hop, dataloader in dataloader_dict.items():  # k_hop in [1, K+1]
            for pos_src, pos_dst, neg_src, neg_dst in dataloader:
                # Move data to device
                pos_src = pos_src.to(device)
                pos_dst = pos_dst.to(device)
                neg_src = neg_src.to(device)
                neg_dst = neg_dst.to(device)
                
                # Forward pass
                p_context_emb = model(k_hop)
                p_emb = model.base_model.primary_emb.weight
                
                # Calculate loss
                loss = model.get_loss(k_hop, pos_src, pos_dst, neg_src, neg_dst, p_emb, p_context_emb)
                running_loss.append(loss.item())
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Calculate mean loss for the epoch
        mean_loss = mean(running_loss)
        scheduler.step(mean_loss)  # Reduce learning rate if needed
        
        print(f"Epoch:{epoch}/{config.train_config['total_epoch']} Loss: {mean_loss:.6f}")
        
        # Check early stopping
        early_stop = stopper.step(mean_loss, model)
        
        # Evaluate the model
        model.eval()
        with torch.no_grad():
            p_emb = model.base_model.primary_emb.weight.detach().cpu().numpy()
            evaluate(p_emb, CF_data, None, config.evaluate_config['method'], 
                    metric=config.data_config['task'],
                    random_state=config.evaluate_config['random_state'],
                    max_iter=config.evaluate_config['max_iter'], 
                    n_jobs=config.evaluate_config['n_jobs'])
        
        if early_stop:
            print("Early stopping triggered.")
            break
    
    # Final evaluation using the best model
    checkpoint_path = stopper.filepath
    print(f"Loading best model from {checkpoint_path} for final evaluation")
    evaluate_task(config, checkpoint_path)
    
    return


if __name__ == "__main__":
    # Dynamically import config based on command line argument
    import importlib.util
    
    config_path = args.c
    if not os.path.exists(config_path):
        config_path = "config_AE_DBLP.py"  # Default fallback
    
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    
    print(f"Using config from: {config_path}")
    main(config) 