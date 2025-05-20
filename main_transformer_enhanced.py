#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Describe: Main script for running TransformerEnhancedCPGNN model

import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from statistics import mean
import os
import dgl

from model.CPGNN import TransformerEnhancedCPGNN, init_weights
from utils.transformer_enhanced import load_dblp_data_with_enhanced_features, save_features, load_features
from utils.preprocess import load_data
from utils.earlystop import EarlyStopping
from evaluate import evaluate_task

def main():
    parser = argparse.ArgumentParser(description='TransformerEnhancedCPGNN Training')
    parser.add_argument('--data_path', type=str, default='data/DBLP/', help='Path to the dataset')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID, -1 for CPU')
    parser.add_argument('--config_file', type=str, default='config.py', help='Configuration file')
    args = parser.parse_args()

    # Import config
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", args.config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    # Choose device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    print(f'Using device: {device}')

    # Load data with enhanced features
    print("Loading data with transformer-enhanced features...")
    data_config = config.data_config
    data_config['task'] = ['CF']  # Only classification task for now
    
    # Use transformer-enhanced features
    graph, node_features, labels, num_classes, train_idx, test_idx = load_dblp_data_with_enhanced_features(
        data_config, remove_self_loop=False
    )
    
    # Convert labels and indices to tensors if they're not already
    labels = torch.tensor(labels, dtype=torch.long)
    train_idx = torch.tensor(train_idx, dtype=torch.long)
    test_idx = torch.tensor(test_idx, dtype=torch.long)
    
    # Move data to device
    node_features = node_features.to(device)
    labels = labels.to(device)
    train_idx = train_idx.to(device)
    test_idx = test_idx.to(device)
    
    # Create homogeneous graph for the model
    # We focus on the author nodes (primary_type) based on the CP-GNN approach
    if isinstance(graph, dgl.DGLHeteroGraph):
        # If using heterogeneous graph, extract the author subgraph
        primary_type = data_config['primary_type'] 
        homo_g = dgl.to_homogeneous(graph, ndata=['feat'] if 'feat' in graph.nodes[primary_type].data else [])
        # Filter to only include primary type nodes
        primary_nodes = (homo_g.ndata['_TYPE'] == graph.ntypes.index(primary_type))
        g = dgl.node_subgraph(homo_g, primary_nodes)
    else:
        # If already homogeneous
        g = graph
    
    g = g.to(device)
    
    # Initialize model
    in_dim = node_features.shape[1]  # DistilBERT (768) + num_categories
    model = TransformerEnhancedCPGNN(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout
    )
    model.apply(init_weights)
    model = model.to(device)
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Setup checkpoint directory
    checkpoint_dir = os.path.join('checkpoint', 'transformer_enhanced')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Early stopping
    early_stopping = EarlyStopping(checkpoint_path=checkpoint_dir, patience=args.patience)
    
    # Training loop
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        node_embeddings = model(g, node_features)
        
        # Calculate loss for node classification
        logits = torch.nn.Linear(args.hidden_dim, num_classes).to(device)(node_embeddings[train_idx])
        loss = criterion(logits, labels[train_idx])
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_embeddings = model(g, node_features)
            val_logits = torch.nn.Linear(args.hidden_dim, num_classes).to(device)(val_embeddings[test_idx])
            val_loss = criterion(val_logits, labels[test_idx])
            
            # Calculate accuracy
            _, predicted = torch.max(val_logits, 1)
            accuracy = (predicted == labels[test_idx]).sum().item() / len(test_idx)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Print progress
        print(f'Epoch {epoch+1}/{args.epochs}, '
              f'Train Loss: {loss.item():.4f}, '
              f'Val Loss: {val_loss.item():.4f}, '
              f'Val Accuracy: {accuracy:.4f}')
        
        # Check early stopping
        early_stop = early_stopping.step(val_loss.item(), model)
        if early_stop:
            print("Early stopping triggered")
            break
    
    # Load best model
    best_model = TransformerEnhancedCPGNN(
        in_dim=in_dim, 
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout
    )
    
    best_checkpoint = os.path.join(checkpoint_dir, 'model.pth')
    if os.path.exists(best_checkpoint):
        best_model.load_state_dict(torch.load(best_checkpoint))
        best_model = best_model.to(device)
        
        # Final evaluation
        best_model.eval()
        with torch.no_grad():
            final_embeddings = best_model(g, node_features)
            final_logits = torch.nn.Linear(args.hidden_dim, num_classes).to(device)(final_embeddings[test_idx])
            _, predicted = torch.max(final_logits, 1)
            final_accuracy = (predicted == labels[test_idx]).sum().item() / len(test_idx)
            
        print(f"Final test accuracy: {final_accuracy:.4f}")
        
        # Save embeddings for further analysis
        embeddings_path = os.path.join('results', 'transformer_enhanced_embeddings.pkl')
        os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
        save_features(final_embeddings.cpu(), embeddings_path)
        print(f"Embeddings saved to {embeddings_path}")

if __name__ == '__main__':
    main() 