#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Describe: Training script for Adaptive Path Length CP-GNN model

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

from models.adaptive_path_length_cpgnn import (
    AdaptivePathLengthCPGNN, 
    AdaptivePathLengthDynamicCPGNN
)

from utils.data_utils import load_data, load_dynamic_data, process_adjacency, normalize_features
from utils.train_utils import accuracy, link_prediction_metrics, early_stopping
from utils.visualization import (
    plot_training_curve, 
    plot_path_length_distribution, 
    visualize_adaptive_mlp_structure
)


def parse_args():
    parser = argparse.ArgumentParser(description='Training script for Adaptive Path Length CP-GNN')
    
    # Basic parameters
    parser.add_argument('--dataset', type=str, default='DBLP', help='Dataset name')
    parser.add_argument('--task', type=str, default='node_classification', 
                        choices=['node_classification', 'link_prediction'], 
                        help='Task to perform')
    parser.add_argument('--dynamic', action='store_true', help='Use dynamic graph model')
    parser.add_argument('--use_cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension size')
    parser.add_argument('--hidden_dims', type=str, default='64,64,64', 
                        help='Hidden dimensions for adaptive MLP (comma-separated)')
    parser.add_argument('--max_path_length', type=int, default=6, 
                        help='Maximum context path length')
    parser.add_argument('--num_channels', type=int, default=3, 
                        help='Number of propagation channels')
    parser.add_argument('--num_heads', type=int, default=4, 
                        help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=2, 
                        help='Number of Transformer layers (for dynamic model)')
    parser.add_argument('--ff_dim', type=int, default=128, 
                        help='Feed-forward dimension in Transformer (for dynamic model)')
    parser.add_argument('--rnn_type', type=str, default='gru', choices=['gru', 'lstm'],
                        help='Type of RNN for temporal modeling (for dynamic model)')
    parser.add_argument('--use_temporal_attention', action='store_true',
                        help='Use temporal attention (for dynamic model)')
    parser.add_argument('--use_pooling', action='store_true',
                        help='Use length-adaptive pooling')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping')
    parser.add_argument('--time_steps', type=int, default=10, 
                        help='Number of time steps (for dynamic model)')
    
    # Output parameters
    parser.add_argument('--log_interval', type=int, default=10, help='Logging interval')
    parser.add_argument('--save_dir', type=str, default='checkpoint', help='Save directory')
    parser.add_argument('--visualize', action='store_true', 
                        help='Visualize path length distribution and model structure')
    
    args = parser.parse_args()
    
    # Convert hidden_dims string to list
    args.hidden_dims = [int(dim) for dim in args.hidden_dims.split(',')]
    
    return args


def prepare_static_data(args, device):
    """Load and prepare static graph data"""
    print(f"Loading {args.dataset} dataset...")
    
    # Load data
    adj_matrices, features, labels, train_mask, val_mask, test_mask, num_classes = load_data(args.dataset)
    
    # Process adjacency matrices
    processed_adj = [process_adjacency(adj) for adj in adj_matrices]
    
    # Normalize features
    features = normalize_features(features)
    
    # Convert to PyTorch tensors
    processed_adj = [torch.FloatTensor(adj).to(device) for adj in processed_adj]
    features = torch.FloatTensor(features).to(device)
    labels = torch.LongTensor(labels).to(device)
    train_mask = torch.BoolTensor(train_mask).to(device)
    val_mask = torch.BoolTensor(val_mask).to(device)
    test_mask = torch.BoolTensor(test_mask).to(device)
    
    return processed_adj, features, labels, train_mask, val_mask, test_mask, num_classes


def prepare_dynamic_data(args, device):
    """Load and prepare dynamic graph data"""
    print(f"Loading dynamic {args.dataset} dataset...")
    
    # Load dynamic data
    graph_sequences, feature_sequences, labels, train_mask, val_mask, test_mask, num_classes = load_dynamic_data(
        args.dataset, args.time_steps
    )
    
    # Process each graph in the sequence
    processed_graphs = []
    for graph_seq in graph_sequences:
        processed_seq = [process_adjacency(adj) for adj in graph_seq]
        processed_graphs.append(processed_seq)
    
    # Normalize features in each sequence
    normalized_features = []
    for feature_seq in feature_sequences:
        normalized_seq = [normalize_features(feat) for feat in feature_seq]
        normalized_features.append(normalized_seq)
    
    # Convert to PyTorch tensors
    processed_graphs = [[torch.FloatTensor(adj).to(device) for adj in seq] for seq in processed_graphs]
    normalized_features = [[torch.FloatTensor(feat).to(device) for feat in seq] for seq in normalized_features]
    labels = torch.LongTensor(labels).to(device)
    train_mask = torch.BoolTensor(train_mask).to(device)
    val_mask = torch.BoolTensor(val_mask).to(device)
    test_mask = torch.BoolTensor(test_mask).to(device)
    
    return processed_graphs, normalized_features, labels, train_mask, val_mask, test_mask, num_classes


def create_model(args, input_dim, num_classes, device):
    """Create appropriate model based on arguments"""
    if args.dynamic:
        model = AdaptivePathLengthDynamicCPGNN(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            hidden_dims=args.hidden_dims,
            max_path_length=args.max_path_length,
            num_channels=args.num_channels,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            ff_dim=args.ff_dim,
            rnn_type=args.rnn_type,
            use_temporal_attention=args.use_temporal_attention,
            use_pooling=args.use_pooling,
            dropout_rate=args.dropout
        )
    else:
        model = AdaptivePathLengthCPGNN(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            hidden_dims=args.hidden_dims,
            max_path_length=args.max_path_length,
            num_channels=args.num_channels,
            num_heads=args.num_heads,
            use_pooling=args.use_pooling,
            dropout_rate=args.dropout
        )
    
    # Add task-specific output layer
    if args.task == 'node_classification':
        model.classifier = nn.Linear(args.hidden_dim, num_classes)
    else:  # link_prediction
        model.link_predictor = nn.Sequential(
            nn.Linear(args.hidden_dim * 2, args.hidden_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_dim, 1),
            nn.Sigmoid()
        )
    
    model = model.to(device)
    return model


def train_static_model(model, adj_matrices, features, labels, train_mask, optimizer, args):
    """Train the static model for one epoch"""
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    node_embeddings, path_attention = model(features, adj_matrices)
    
    # Task-specific loss
    if args.task == 'node_classification':
        # Classification loss
        logits = model.classifier(node_embeddings)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
    else:  # link_prediction
        # Generate positive and negative samples
        pos_edges, neg_edges = generate_edge_samples(adj_matrices[0], train_mask)
        loss = link_prediction_loss(model, node_embeddings, pos_edges, neg_edges)
    
    # Calculate loss and backpropagate
    loss.backward()
    optimizer.step()
    
    return loss.item(), path_attention


def train_dynamic_model(model, graph_sequences, feature_sequences, labels, train_mask, optimizer, args):
    """Train the dynamic model for one epoch"""
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    node_embeddings, path_attention = model(graph_sequences, feature_sequences, args.time_steps)
    
    # Use the embeddings from the last time step
    final_embeddings = node_embeddings[:, -1]
    
    # Task-specific loss
    if args.task == 'node_classification':
        # Classification loss
        logits = model.classifier(final_embeddings)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
    else:  # link_prediction
        # Generate positive and negative samples from the last graph
        pos_edges, neg_edges = generate_edge_samples(graph_sequences[-1][0], train_mask)
        loss = link_prediction_loss(model, final_embeddings, pos_edges, neg_edges)
    
    # Calculate loss and backpropagate
    loss.backward()
    optimizer.step()
    
    return loss.item(), path_attention


def evaluate_static_model(model, adj_matrices, features, labels, mask, args):
    """Evaluate the static model"""
    model.eval()
    
    with torch.no_grad():
        # Forward pass
        node_embeddings, path_attention = model(features, adj_matrices)
        
        # Task-specific metrics
        if args.task == 'node_classification':
            # Classification metrics
            logits = model.classifier(node_embeddings)
            loss = F.cross_entropy(logits[mask], labels[mask])
            acc = accuracy(logits[mask], labels[mask])
            return loss.item(), acc, path_attention
        else:  # link_prediction
            # Generate positive and negative samples
            pos_edges, neg_edges = generate_edge_samples(adj_matrices[0], mask)
            loss = link_prediction_loss(model, node_embeddings, pos_edges, neg_edges)
            auc, ap = link_prediction_metrics(model, node_embeddings, pos_edges, neg_edges)
            return loss.item(), (auc, ap), path_attention


def evaluate_dynamic_model(model, graph_sequences, feature_sequences, labels, mask, args):
    """Evaluate the dynamic model"""
    model.eval()
    
    with torch.no_grad():
        # Forward pass
        node_embeddings, path_attention = model(graph_sequences, feature_sequences, args.time_steps)
        
        # Use the embeddings from the last time step
        final_embeddings = node_embeddings[:, -1]
        
        # Task-specific metrics
        if args.task == 'node_classification':
            # Classification metrics
            logits = model.classifier(final_embeddings)
            loss = F.cross_entropy(logits[mask], labels[mask])
            acc = accuracy(logits[mask], labels[mask])
            return loss.item(), acc, path_attention
        else:  # link_prediction
            # Generate positive and negative samples from the last graph
            pos_edges, neg_edges = generate_edge_samples(graph_sequences[-1][0], mask)
            loss = link_prediction_loss(model, final_embeddings, pos_edges, neg_edges)
            auc, ap = link_prediction_metrics(model, final_embeddings, pos_edges, neg_edges)
            return loss.item(), (auc, ap), path_attention


def generate_edge_samples(adj, mask):
    """Generate positive and negative edge samples for link prediction"""
    # Extract positive edges from adjacency matrix
    pos_edges = adj[mask].nonzero(as_tuple=False)
    
    # Generate random negative edges (edges that don't exist)
    num_nodes = adj.size(0)
    neg_edges = []
    while len(neg_edges) < len(pos_edges):
        i, j = np.random.randint(0, num_nodes, 2)
        if adj[i, j] == 0 and i != j:
            neg_edges.append([i, j])
    
    neg_edges = torch.LongTensor(neg_edges).to(adj.device)
    
    return pos_edges, neg_edges


def link_prediction_loss(model, node_embeddings, pos_edges, neg_edges):
    """Calculate link prediction loss"""
    # Extract node pairs for positive edges
    pos_u = node_embeddings[pos_edges[:, 0]]
    pos_v = node_embeddings[pos_edges[:, 1]]
    pos_scores = model.link_predictor(torch.cat([pos_u, pos_v], dim=1))
    
    # Extract node pairs for negative edges
    neg_u = node_embeddings[neg_edges[:, 0]]
    neg_v = node_embeddings[neg_edges[:, 1]]
    neg_scores = model.link_predictor(torch.cat([neg_u, neg_v], dim=1))
    
    # Combine into one batch with targets
    scores = torch.cat([pos_scores, neg_scores], dim=0)
    targets = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=0)
    
    # Binary cross entropy loss
    loss = F.binary_cross_entropy(scores, targets)
    
    return loss


def visualize_results(model, path_attention, args):
    """Visualize model results and structure"""
    if not args.visualize:
        return
    
    # Create directory for visualizations
    viz_dir = os.path.join(args.save_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Plot path length distribution
    if path_attention is not None:
        plot_path_length_distribution(
            path_attention, 
            os.path.join(viz_dir, 'path_length_distribution.png')
        )
    
    # Visualize adaptive MLP structure
    visualize_adaptive_mlp_structure(
        model, 
        os.path.join(viz_dir, 'adaptive_mlp_structure.png'),
        args.max_path_length,
        args.hidden_dims
    )


def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load data
    if args.dynamic:
        processed_graphs, normalized_features, labels, train_mask, val_mask, test_mask, num_classes = prepare_dynamic_data(args, device)
        input_dim = normalized_features[0][0].size(1)
    else:
        processed_adj, features, labels, train_mask, val_mask, test_mask, num_classes = prepare_static_data(args, device)
        input_dim = features.size(1)
    
    # Create model
    model = create_model(args, input_dim, num_classes, device)
    print(model)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
    
    # Track best validation score
    best_val_score = 0.0
    best_epoch = 0
    patience_counter = 0
    
    # Training loop
    train_losses = []
    val_losses = []
    val_scores = []
    
    print(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # Training
        if args.dynamic:
            train_loss, path_attention = train_dynamic_model(
                model, processed_graphs, normalized_features, labels, train_mask, optimizer, args
            )
        else:
            train_loss, path_attention = train_static_model(
                model, processed_adj, features, labels, train_mask, optimizer, args
            )
        
        train_losses.append(train_loss)
        
        # Validation
        if args.dynamic:
            val_loss, val_score, path_attention = evaluate_dynamic_model(
                model, processed_graphs, normalized_features, labels, val_mask, args
            )
        else:
            val_loss, val_score, path_attention = evaluate_static_model(
                model, processed_adj, features, labels, val_mask, args
            )
        
        val_losses.append(val_loss)
        
        # Handle metrics based on task
        if args.task == 'node_classification':
            val_acc = val_score
            val_scores.append(val_acc)
            is_better = val_acc > best_val_score
            score_name = "Accuracy"
        else:  # link_prediction
            val_auc, val_ap = val_score
            val_scores.append(val_auc)
            is_better = val_auc > best_val_score
            score_name = "AUC"
        
        # Update best validation score
        if is_better:
            best_val_score = val_scores[-1]
            best_epoch = epoch
            patience_counter = 0
            
            # Save the model
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'best_model_{args.dataset}.pt'))
        else:
            patience_counter += 1
        
        # Update learning rate
        scheduler.step()
        
        # Print progress
        if epoch % args.log_interval == 0:
            if args.task == 'node_classification':
                print(f"Epoch {epoch}/{args.epochs}, Time: {time.time() - start_time:.2f}s, "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            else:  # link_prediction
                print(f"Epoch {epoch}/{args.epochs}, Time: {time.time() - start_time:.2f}s, "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}")
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}, best epoch: {best_epoch}")
            break
    
    # Plot training curves
    plot_training_curve(
        train_losses, val_losses, val_scores, 
        os.path.join(args.save_dir, f'training_curve_{args.dataset}.png'),
        score_name
    )
    
    # Load best model for testing
    model.load_state_dict(torch.load(os.path.join(args.save_dir, f'best_model_{args.dataset}.pt')))
    
    # Test
    if args.dynamic:
        test_loss, test_score, path_attention = evaluate_dynamic_model(
            model, processed_graphs, normalized_features, labels, test_mask, args
        )
    else:
        test_loss, test_score, path_attention = evaluate_static_model(
            model, processed_adj, features, labels, test_mask, args
        )
    
    # Print test results
    if args.task == 'node_classification':
        test_acc = test_score
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    else:  # link_prediction
        test_auc, test_ap = test_score
        print(f"Test Loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}")
    
    # Visualize results
    visualize_results(model, path_attention, args)
    
    print("Training completed!")


if __name__ == '__main__':
    main() 