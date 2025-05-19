#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Describe: Visualization utilities for Adaptive Path Length CP-GNN

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import networkx as nx


def plot_training_curve(train_losses, val_losses, val_scores, save_path, score_name="Accuracy"):
    """
    Plot training and validation curves
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        val_scores: List of validation scores (accuracy or AUC)
        save_path: Path to save the plot
        score_name: Name of the score metric
    """
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot scores
    plt.subplot(1, 2, 2)
    plt.plot(val_scores, label=f'Validation {score_name}', color='green')
    plt.xlabel('Epoch')
    plt.ylabel(score_name)
    plt.title(f'Validation {score_name}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_path_length_distribution(path_attention, save_path):
    """
    Plot distribution of path length attention weights
    
    Args:
        path_attention: Path attention weights [batch_size, num_nodes, max_path_length]
                       or [batch_size, time_steps, num_nodes, max_path_length]
        save_path: Path to save the plot
    """
    # Handle different dimensions (static vs. dynamic models)
    if path_attention.dim() == 4:  # Dynamic model
        # Use the last time step
        path_attention = path_attention[:, -1]
    
    # Convert to numpy for plotting
    if isinstance(path_attention, torch.Tensor):
        path_attention = path_attention.detach().cpu().numpy()
    
    # Average over batch and nodes
    avg_attention = np.mean(path_attention, axis=(0, 1))
    max_path_length = avg_attention.shape[0]
    
    plt.figure(figsize=(10, 6))
    
    # Bar plot for average attention
    plt.subplot(1, 2, 1)
    plt.bar(range(1, max_path_length + 1), avg_attention, alpha=0.7)
    plt.xlabel('Path Length')
    plt.ylabel('Average Attention Weight')
    plt.title('Average Path Length Attention')
    plt.xticks(range(1, max_path_length + 1))
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Heatmap for node-wise path length distribution
    plt.subplot(1, 2, 2)
    num_nodes = min(50, path_attention.shape[1])  # Limit to 50 nodes for visualization
    sns.heatmap(
        path_attention[0, :num_nodes, :], 
        cmap='viridis', 
        xticklabels=range(1, max_path_length + 1),
        yticklabels=range(1, num_nodes + 1),
        cbar_kws={'label': 'Attention Weight'}
    )
    plt.xlabel('Path Length')
    plt.ylabel('Node ID')
    plt.title('Path Length Attention per Node')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_adaptive_mlp_structure(model, save_path, max_path_length, hidden_dims):
    """
    Visualize the structure of adaptive MLPs for different path lengths
    
    Args:
        model: The model containing adaptive MLPs
        save_path: Path to save the plot
        max_path_length: Maximum path length
        hidden_dims: Original hidden dimensions list
    """
    plt.figure(figsize=(15, 10))
    
    # Find AdaptiveLengthMLP in the model
    adaptive_mlp = None
    for module in model.modules():
        if hasattr(module, 'path_mlps'):
            adaptive_mlp = module
            break
    
    if adaptive_mlp is None:
        print("No AdaptiveLengthMLP found in the model")
        return
    
    # Get network structures for each path length
    network_structures = []
    
    for length in range(max_path_length):
        mlp = adaptive_mlp.path_mlps[length]
        structure = []
        
        # Extract layer dimensions
        for i, layer in enumerate(mlp):
            if isinstance(layer, torch.nn.Linear):
                structure.append((layer.in_features, layer.out_features))
        
        network_structures.append(structure)
    
    # Plot network structures
    max_layers = max(len(struct) for struct in network_structures)
    colors = plt.cm.viridis(np.linspace(0, 1, max_path_length))
    
    for path_idx, structure in enumerate(network_structures):
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes for each layer
        layer_nodes = []
        for layer_idx, (in_dim, out_dim) in enumerate(structure):
            layer_name = f"Layer {layer_idx+1}"
            G.add_node(layer_name, size=out_dim)
            layer_nodes.append(layer_name)
            
            # Add additional node for input layer if this is the first layer
            if layer_idx == 0:
                input_name = "Input"
                G.add_node(input_name, size=in_dim)
                G.add_edge(input_name, layer_name)
                layer_nodes.insert(0, input_name)
        
        # Add edges between layers
        for i in range(len(layer_nodes) - 1):
            if i > 0:  # Skip the input to first layer edge as we added it above
                G.add_edge(layer_nodes[i], layer_nodes[i+1])
        
        # Position nodes in a hierarchical layout
        pos = {}
        for i, node in enumerate(layer_nodes):
            pos[node] = (i / (len(layer_nodes) - 1), 1 - path_idx / max_path_length)
        
        # Draw the graph in a subplot
        plt.subplot(max_path_length, 1, path_idx + 1)
        
        # Get node sizes based on their dimensions
        node_sizes = [G.nodes[node]['size'] * 10 for node in G.nodes]
        
        # Draw the graph
        nx.draw(
            G, pos, 
            with_labels=True, 
            node_size=node_sizes,
            node_color=[colors[path_idx]] * len(G.nodes),
            alpha=0.8,
            arrows=True,
            edge_color='gray'
        )
        
        plt.title(f"Path Length {path_idx+1} - MLP Structure")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_graph_with_path_lengths(graph, node_features, path_lengths, save_path):
    """
    Visualize graph with nodes colored by their predicted optimal path length
    
    Args:
        graph: Adjacency matrix [num_nodes, num_nodes]
        node_features: Node features [num_nodes, feature_dim]
        path_lengths: Predicted path lengths for each node [num_nodes]
        save_path: Path to save the plot
    """
    # Convert to numpy if tensor
    if isinstance(graph, torch.Tensor):
        graph = graph.detach().cpu().numpy()
    if isinstance(path_lengths, torch.Tensor):
        path_lengths = path_lengths.detach().cpu().numpy()
    
    # Create networkx graph
    G = nx.from_numpy_array(graph)
    
    # Create colormap for path lengths
    max_path_length = np.max(path_lengths) + 1
    cmap = plt.cm.get_cmap('viridis', max_path_length)
    colors = [cmap(path_lengths[i]) for i in range(len(path_lengths))]
    
    # Draw the graph
    plt.figure(figsize=(12, 10))
    
    # Apply force-directed layout
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, 
        node_color=colors,
        node_size=100,
        alpha=0.8
    )
    
    # Draw edges
    nx.draw_networkx_edges(
        G, pos, 
        alpha=0.5,
        width=0.5
    )
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_path_length - 1))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Optimal Path Length')
    
    plt.title('Graph with Nodes Colored by Optimal Path Length')
    plt.axis('off')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_attention_heatmap(attention_weights, save_path):
    """
    Plot attention weight heatmap
    
    Args:
        attention_weights: Attention weights
        save_path: Path to save the plot
    """
    # Convert to numpy for plotting
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attention_weights,
        cmap='viridis',
        cbar_kws={'label': 'Attention Weight'}
    )
    plt.title('Attention Weights Heatmap')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() 