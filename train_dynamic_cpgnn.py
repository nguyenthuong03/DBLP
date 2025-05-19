#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Describe: Training script for Dynamic CP-GNN model

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from models.dynamic_cp_gnn import DynamicCPGNN, DynamicLinkPrediction, DynamicNodeClassification


def parse_args():
    parser = argparse.ArgumentParser(description='Train Dynamic CP-GNN model')
    
    # Model parameters
    parser.add_argument('--input_dim', type=int, default=128, help='Input feature dimension')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--num_channels', type=int, default=4, help='Number of channels in CP-GNN')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--ff_dim', type=int, default=256, help='Feed-forward dimension')
    parser.add_argument('--rnn_type', type=str, default='gru', choices=['gru', 'lstm'], help='Type of RNN for temporal modeling')
    parser.add_argument('--use_temporal_attention', action='store_true', help='Use temporal attention')
    parser.add_argument('--adaptive_channel', action='store_true', help='Use adaptive channel weighting')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--task', type=str, default='link_prediction', 
                        choices=['link_prediction', 'node_classification'], 
                        help='Task to perform')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes for node classification')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='data/dynamic_graph', help='Path to dataset')
    parser.add_argument('--time_steps', type=int, default=10, help='Number of time steps to use')
    parser.add_argument('--forecast_horizon', type=int, default=1, help='Number of future time steps to predict')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--save_path', type=str, default='checkpoints', help='Path to save models')
    parser.add_argument('--log_interval', type=int, default=10, help='Interval for logging')
    
    return parser.parse_args()


class GraphSequenceDataset:
    """Dataset wrapper for dynamic graph sequences"""
    
    def __init__(self, data_path, time_steps, forecast_horizon=1, task='link_prediction'):
        """
        Args:
            data_path: Path to dataset
            time_steps: Number of time steps to use
            forecast_horizon: Number of future time steps to predict
            task: Task to perform ('link_prediction' or 'node_classification')
        """
        self.data_path = data_path
        self.time_steps = time_steps
        self.forecast_horizon = forecast_horizon
        self.task = task
        
        # Load data
        self.load_data()
        
    def load_data(self):
        """Load dynamic graph data from files"""
        # This is a placeholder. In practice, you would load actual data here.
        # The example below simulates a dynamic graph sequence with random data.
        
        print(f"Loading data from {self.data_path}")
        
        # Simulate data loading (replace with actual data loading in real application)
        total_time_steps = self.time_steps + self.forecast_horizon
        num_nodes = 100  # Example, adjust based on your actual dataset
        num_channels = 4  # Example, adjust based on your actual dataset
        feat_dim = 128  # Example, adjust based on your actual dataset
        
        # Create dummy adjacency matrices for each time step and channel
        # In a real scenario, these would be loaded from files
        adjacency_matrices = []
        for t in range(total_time_steps):
            channels = []
            for c in range(num_channels):
                # Create sparse adjacency matrix for each channel
                # In practice, use real graph data with meaningful sparsity pattern
                adj = torch.zeros(num_nodes, num_nodes)
                # Add random edges with 5% density
                edges = torch.rand(num_nodes, num_nodes) < 0.05
                adj[edges] = 1
                # Make sure diagonal is 0 (no self-loops)
                adj.fill_diagonal_(0)
                channels.append(adj)
            adjacency_matrices.append(channels)
        
        # Create dummy node features for each time step
        node_features = []
        for t in range(total_time_steps):
            # Random node features
            features = torch.randn(num_nodes, feat_dim)
            node_features.append(features)
        
        # For link prediction task, create edges to predict
        if self.task == 'link_prediction':
            # Create positive examples (edges that exist)
            pos_edges = []
            for t in range(self.time_steps, total_time_steps):
                # Sample existing edges from the adjacency matrix of the first channel
                adj = adjacency_matrices[t][0]
                exist_edges = adj.nonzero()
                # Sample a subset of existing edges
                if exist_edges.size(0) > 0:
                    indices = torch.randperm(exist_edges.size(0))[:100]  # Sample 100 edges
                    pos_edges.append(exist_edges[indices])
                else:
                    pos_edges.append(torch.zeros(0, 2, dtype=torch.long))
            
            # Create negative examples (edges that don't exist)
            neg_edges = []
            for t in range(self.time_steps, total_time_steps):
                # Sample non-existing edges from the adjacency matrix of the first channel
                adj = adjacency_matrices[t][0]
                non_exist_edges = (1 - adj).nonzero()
                # Sample a subset of non-existing edges
                if non_exist_edges.size(0) > 0:
                    indices = torch.randperm(non_exist_edges.size(0))[:100]  # Sample 100 edges
                    neg_edges.append(non_exist_edges[indices])
                else:
                    neg_edges.append(torch.zeros(0, 2, dtype=torch.long))
            
            self.pos_edges = pos_edges
            self.neg_edges = neg_edges
        
        # For node classification task, create node labels
        elif self.task == 'node_classification':
            # Create random node labels for each time step
            node_labels = []
            for t in range(self.time_steps, total_time_steps):
                # Random node labels
                labels = torch.randint(0, 2, (num_nodes,))
                node_labels.append(labels)
            
            self.node_labels = node_labels
        
        # Store the data
        self.adjacency_matrices = adjacency_matrices
        self.node_features = node_features
        self.num_nodes = num_nodes
        self.feat_dim = feat_dim
        self.num_samples = 100  # Example, adjust based on your actual dataset
        
        print(f"Loaded {total_time_steps} time steps with {num_nodes} nodes each")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Return a sequence of graph snapshots
        # In a real scenario, you might have different samples or start points
        
        # Prepare input sequences
        input_adj = self.adjacency_matrices[:self.time_steps]
        input_feats = self.node_features[:self.time_steps]
        
        # Prepare target sequences
        target_adj = self.adjacency_matrices[self.time_steps:self.time_steps+self.forecast_horizon]
        target_feats = self.node_features[self.time_steps:self.time_steps+self.forecast_horizon]
        
        sample = {
            'input_adj': input_adj,
            'input_feats': input_feats,
            'target_adj': target_adj,
            'target_feats': target_feats,
        }
        
        # Add task-specific data
        if self.task == 'link_prediction':
            # Combine positive and negative edges with labels
            pos_edges = self.pos_edges[0]  # First time step in forecast horizon
            neg_edges = self.neg_edges[0]  # First time step in forecast horizon
            
            num_pos = pos_edges.size(0)
            num_neg = neg_edges.size(0)
            
            edges = torch.cat([pos_edges, neg_edges], dim=0)
            labels = torch.cat([torch.ones(num_pos), torch.zeros(num_neg)])
            
            # Shuffle edges and labels
            indices = torch.randperm(edges.size(0))
            edges = edges[indices]
            labels = labels[indices]
            
            sample['edges'] = edges
            sample['labels'] = labels
            
        elif self.task == 'node_classification':
            # Add node labels
            sample['node_labels'] = self.node_labels[0]  # First time step in forecast horizon
        
        return sample


def train_epoch(model, task_model, dataloader, optimizer, device, task):
    """Train for one epoch"""
    model.train()
    task_model.train()
    
    total_loss = 0
    total_samples = 0
    
    criterion = nn.BCELoss() if task == 'link_prediction' else nn.CrossEntropyLoss()
    
    for batch in tqdm(dataloader, desc='Training'):
        # Get batch data
        input_adj = [[[adj.to(device) for adj in time_step] for time_step in batch['input_adj']]]
        input_feats = [[feat.to(device) for feat in batch['input_feats']]]
        
        # Forward pass through Dynamic CP-GNN
        node_embeddings = model(input_adj, input_feats)
        
        # Task-specific forward pass and loss computation
        if task == 'link_prediction':
            edges = batch['edges'].to(device)
            labels = batch['labels'].to(device)
            
            # Predict links
            predictions = task_model(node_embeddings, edges.unsqueeze(0))
            loss = criterion(predictions, labels)
            
        elif task == 'node_classification':
            node_labels = batch['node_labels'].to(device)
            
            # Predict node classes
            logits = task_model(node_embeddings)
            loss = criterion(logits.squeeze(0), node_labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update statistics
        batch_size = 1  # In this example we have 1 graph per batch
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    
    return total_loss / total_samples


def evaluate(model, task_model, dataloader, device, task):
    """Evaluate model"""
    model.eval()
    task_model.eval()
    
    total_loss = 0
    total_samples = 0
    all_labels = []
    all_predictions = []
    
    criterion = nn.BCELoss() if task == 'link_prediction' else nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            # Get batch data
            input_adj = [[[adj.to(device) for adj in time_step] for time_step in batch['input_adj']]]
            input_feats = [[feat.to(device) for feat in batch['input_feats']]]
            
            # Forward pass through Dynamic CP-GNN
            node_embeddings = model(input_adj, input_feats)
            
            # Task-specific forward pass and metrics computation
            if task == 'link_prediction':
                edges = batch['edges'].to(device)
                labels = batch['labels'].to(device)
                
                # Predict links
                predictions = task_model(node_embeddings, edges.unsqueeze(0))
                loss = criterion(predictions, labels)
                
                # Store predictions and labels for metrics computation
                all_labels.append(labels.cpu().numpy())
                all_predictions.append(predictions.cpu().numpy())
                
            elif task == 'node_classification':
                node_labels = batch['node_labels'].to(device)
                
                # Predict node classes
                logits = task_model(node_embeddings)
                loss = criterion(logits.squeeze(0), node_labels)
                
                # Store predictions and labels for metrics computation
                all_labels.append(node_labels.cpu().numpy())
                all_predictions.append(torch.argmax(logits.squeeze(0), dim=1).cpu().numpy())
            
            # Update statistics
            batch_size = 1  # In this example we have 1 graph per batch
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    
    # Compute metrics
    all_labels = np.concatenate(all_labels)
    all_predictions = np.concatenate(all_predictions)
    
    metrics = {}
    metrics['loss'] = total_loss / total_samples
    
    if task == 'link_prediction':
        metrics['auc'] = roc_auc_score(all_labels, all_predictions)
        # Convert continuous predictions to binary
        binary_preds = (all_predictions > 0.5).astype(int)
        metrics['accuracy'] = accuracy_score(all_labels, binary_preds)
        metrics['f1'] = f1_score(all_labels, binary_preds)
    elif task == 'node_classification':
        metrics['accuracy'] = accuracy_score(all_labels, all_predictions)
        metrics['f1'] = f1_score(all_labels, all_predictions, average='macro')
    
    return metrics


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory if not exists
    os.makedirs(args.save_path, exist_ok=True)
    
    # Create datasets and dataloaders
    train_dataset = GraphSequenceDataset(
        os.path.join(args.data_path, 'train'),
        args.time_steps,
        args.forecast_horizon,
        args.task
    )
    
    val_dataset = GraphSequenceDataset(
        os.path.join(args.data_path, 'val'),
        args.time_steps,
        args.forecast_horizon,
        args.task
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    model = DynamicCPGNN(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_channels=args.num_channels,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        rnn_type=args.rnn_type,
        use_temporal_attention=args.use_temporal_attention,
        adaptive_channel=args.adaptive_channel,
        dropout_rate=args.dropout
    ).to(device)
    
    # Create task-specific model
    if args.task == 'link_prediction':
        task_model = DynamicLinkPrediction(args.hidden_dim).to(device)
    elif args.task == 'node_classification':
        task_model = DynamicNodeClassification(args.hidden_dim, args.num_classes).to(device)
    
    # Create optimizer
    optimizer = optim.Adam(
        list(model.parameters()) + list(task_model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    best_val_metric = float('inf') if args.task == 'link_prediction' else 0
    best_epoch = 0
    train_losses = []
    val_losses = []
    val_metrics = []
    
    print(f"Starting training for {args.epochs} epochs")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        # Train for one epoch
        train_loss = train_epoch(model, task_model, train_loader, optimizer, device, args.task)
        train_losses.append(train_loss)
        
        # Evaluate
        val_metrics_dict = evaluate(model, task_model, val_loader, device, args.task)
        val_loss = val_metrics_dict['loss']
        val_losses.append(val_loss)
        
        # Get primary metric for model selection
        if args.task == 'link_prediction':
            primary_metric = val_metrics_dict['auc']
            is_better = primary_metric > best_val_metric
        else:  # node_classification
            primary_metric = val_metrics_dict['accuracy']
            is_better = primary_metric > best_val_metric
        
        val_metrics.append(primary_metric)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if is_better:
            best_val_metric = primary_metric
            best_epoch = epoch
            
            # Save model
            checkpoint = {
                'model': model.state_dict(),
                'task_model': task_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
                'val_metrics': val_metrics_dict
            }
            
            torch.save(checkpoint, os.path.join(args.save_path, 'best_model.pth'))
            print(f"Saved best model at epoch {epoch}")
        
        # Log progress
        if (epoch + 1) % args.log_interval == 0:
            metric_name = 'AUC' if args.task == 'link_prediction' else 'Accuracy'
            print(f"Epoch {epoch+1}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val {metric_name}: {primary_metric:.4f}")
    
    train_time = time.time() - start_time
    print(f"Training finished in {train_time/60:.2f} minutes")
    print(f"Best validation {metric_name}: {best_val_metric:.4f} at epoch {best_epoch}")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_metrics, label=f'Val {metric_name}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} Curve')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_path, 'training_curves.png'))
    
    print(f"Training curves saved to {os.path.join(args.save_path, 'training_curves.png')}")
    
    # Load best model for final evaluation
    checkpoint = torch.load(os.path.join(args.save_path, 'best_model.pth'))
    model.load_state_dict(checkpoint['model'])
    task_model.load_state_dict(checkpoint['task_model'])
    
    # Final evaluation
    final_metrics = evaluate(model, task_model, val_loader, device, args.task)
    print("Final validation metrics:")
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == '__main__':
    main() 