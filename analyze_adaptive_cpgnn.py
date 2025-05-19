#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Describe: Analysis script for Adaptive CP-GNN results

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import pandas as pd
from matplotlib.colors import ListedColormap

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Adaptive CP-GNN results')
    
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model checkpoint')
    parser.add_argument('--output_dir', type=str, default='analysis_results', help='Directory to save analysis results')
    parser.add_argument('--num_clusters', type=int, default=3, help='Number of clusters for node clustering')
    parser.add_argument('--tsne_perplexity', type=int, default=30, help='Perplexity for t-SNE visualization')
    
    return parser.parse_args()

def load_checkpoint(checkpoint_path):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    return checkpoint

def visualize_path_attention_per_node(path_attentions, output_dir):
    """Visualize path attention weights for a sample of nodes"""
    # Select a few nodes to visualize
    num_nodes = path_attentions.shape[2]
    num_time_steps = path_attentions.shape[1]
    max_path_length = path_attentions.shape[3]
    
    # Select 5 nodes at random
    sample_nodes = np.random.choice(num_nodes, min(5, num_nodes), replace=False)
    
    plt.figure(figsize=(15, 10))
    
    # For each sampled node
    for i, node_idx in enumerate(sample_nodes):
        plt.subplot(len(sample_nodes), 1, i+1)
        
        # Get attention for this node across all time steps
        node_attention = path_attentions[0, :, node_idx, :]  # [time_steps, max_path_length]
        
        # Create heatmap
        sns.heatmap(node_attention, cmap='viridis', 
                   xticklabels=range(1, max_path_length+1),
                   yticklabels=range(1, num_time_steps+1),
                   cbar_kws={'label': 'Attention Weight'})
        
        plt.xlabel('Context Path Length')
        plt.ylabel('Time Step')
        plt.title(f'Path Attention Heatmap for Node {node_idx}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'path_attention_per_node.png'))
    plt.close()
    
    print(f"Node-specific attention visualization saved to {output_dir}")

def analyze_path_length_distribution(path_attentions, output_dir):
    """Analyze the distribution of preferred path lengths"""
    # Get preferred path length for each node at each time step
    preferred_lengths = np.argmax(path_attentions, axis=-1) + 1  # +1 to make it 1-indexed
    
    # Overall distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(preferred_lengths.flatten(), discrete=True, kde=True)
    plt.xlabel('Preferred Context Path Length')
    plt.ylabel('Count')
    plt.title('Distribution of Preferred Context Path Length')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'path_length_distribution.png'))
    plt.close()
    
    # Distribution over time
    plt.figure(figsize=(12, 6))
    
    # For each time step, compute the proportion of nodes preferring each path length
    time_steps = path_attentions.shape[1]
    max_path_length = path_attentions.shape[3]
    
    proportions = np.zeros((time_steps, max_path_length))
    for t in range(time_steps):
        for l in range(max_path_length):
            proportions[t, l] = np.mean(preferred_lengths[:, t, :] == l+1)
    
    # Plot as stacked area chart
    x = range(1, time_steps+1)
    plt.stackplot(x, proportions.T, labels=[f'Length {i+1}' for i in range(max_path_length)])
    plt.xlabel('Time Step')
    plt.ylabel('Proportion of Nodes')
    plt.title('Evolution of Preferred Path Length Over Time')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'path_length_evolution.png'))
    plt.close()
    
    print(f"Path length distribution analysis saved to {output_dir}")

def cluster_nodes_by_attention(path_attentions, num_clusters, output_dir):
    """Cluster nodes based on their attention patterns"""
    # Average over time to get per-node attention profile
    node_attention_profiles = path_attentions.mean(axis=1)  # [batch_size, num_nodes, max_path_length]
    
    # Reshape for clustering
    batch_size, num_nodes, max_path_length = node_attention_profiles.shape
    profiles_flat = node_attention_profiles.reshape(batch_size * num_nodes, max_path_length)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(profiles_flat)
    
    # Reshape clusters back to [batch_size, num_nodes]
    clusters = clusters.reshape(batch_size, num_nodes)
    
    # Visualize cluster centers (attention profiles for each cluster)
    plt.figure(figsize=(10, 6))
    cluster_centers = kmeans.cluster_centers_
    
    for i in range(num_clusters):
        plt.plot(range(1, max_path_length+1), cluster_centers[i], marker='o', label=f'Cluster {i+1}')
    
    plt.xlabel('Context Path Length')
    plt.ylabel('Average Attention Weight')
    plt.title('Attention Profiles by Node Cluster')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'node_clusters_attention.png'))
    plt.close()
    
    # Apply t-SNE to visualize nodes in 2D space based on attention profiles
    tsne = TSNE(n_components=2, perplexity=min(30, profiles_flat.shape[0]-1), random_state=42)
    embeddings_2d = tsne.fit_transform(profiles_flat)
    
    # Create a DataFrame for easier plotting
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'cluster': clusters.flatten()
    })
    
    # Plot the clusters
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='x', y='y', hue='cluster', data=df, palette='viridis')
    plt.title('t-SNE Visualization of Nodes Clustered by Attention Profile')
    plt.savefig(os.path.join(output_dir, 'tsne_node_clusters.png'))
    plt.close()
    
    print(f"Node clustering analysis saved to {output_dir}")

def visualize_attention_vs_structural(path_attentions, node_embeddings, output_dir):
    """Visualize relationship between structural properties and attention patterns"""
    # For simplicity, assume we have node structural properties like degree and centrality
    # In a real scenario, these would come from the actual graph
    
    # Generate synthetic structural properties for demonstration
    batch_size, time_steps, num_nodes, max_path_length = path_attentions.shape
    
    # Average attention weights over time
    avg_attention = path_attentions.mean(axis=1)  # [batch_size, num_nodes, max_path_length]
    
    # Get preferred path length for each node
    preferred_length = np.argmax(avg_attention, axis=2) + 1  # [batch_size, num_nodes]
    
    # Create synthetic structural properties
    # In a real analysis, you would use actual node properties
    node_degree = np.random.power(0.5, size=(batch_size, num_nodes)) * 30
    node_centrality = np.random.beta(2, 5, size=(batch_size, num_nodes))
    
    # Flatten arrays for plotting
    preferred_length_flat = preferred_length.flatten()
    node_degree_flat = node_degree.flatten()
    node_centrality_flat = node_centrality.flatten()
    
    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Degree vs Preferred Path Length
    axes[0].scatter(node_degree_flat, preferred_length_flat, alpha=0.5)
    axes[0].set_xlabel('Node Degree')
    axes[0].set_ylabel('Preferred Context Path Length')
    axes[0].set_title('Node Degree vs Preferred Path Length')
    axes[0].grid(True)
    
    # Centrality vs Preferred Path Length
    axes[1].scatter(node_centrality_flat, preferred_length_flat, alpha=0.5)
    axes[1].set_xlabel('Node Centrality')
    axes[1].set_ylabel('Preferred Context Path Length')
    axes[1].set_title('Node Centrality vs Preferred Path Length')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'structural_vs_attention.png'))
    plt.close()
    
    # Box plot of path length by degree bins
    degree_bins = pd.qcut(node_degree_flat, 4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=degree_bins, y=preferred_length_flat)
    plt.xlabel('Node Degree')
    plt.ylabel('Preferred Context Path Length')
    plt.title('Preferred Path Length Distribution by Node Degree')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'path_length_by_degree.png'))
    plt.close()
    
    print(f"Structural property analysis saved to {output_dir}")

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the checkpoint
    checkpoint = load_checkpoint(args.model_path)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Extract model parameters and validation metrics
    val_metrics = checkpoint['val_metrics']
    print("Validation metrics:")
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Extract args from the checkpoint to understand the model configuration
    model_args = checkpoint['args']
    print("\nModel configuration:")
    for k, v in vars(model_args).items():
        print(f"  {k}: {v}")
    
    # Generate synthetic path attention data for demonstration
    # In a real scenario, you would use the actual model to generate these
    # or load them from a previous run
    batch_size = 1
    time_steps = 5
    num_nodes = 100
    max_path_length = model_args.max_path_length
    
    # Synthetic path attentions with a pattern: smaller path lengths preferred for high-degree nodes
    path_attentions = np.zeros((batch_size, time_steps, num_nodes, max_path_length))
    
    # For each node, generate attention that depends on synthetic "degree"
    for n in range(num_nodes):
        # Synthetic node degree - higher for nodes with lower indices
        degree = 1 - (n / num_nodes)
        
        # Nodes with high degree (low index) prefer shorter paths
        # Nodes with low degree (high index) prefer longer paths
        mu = 0.5 + 3.5 * (1 - degree)  # mu between 0.5 and 4.0
        
        # Generate attention weights centered around preferred length
        for l in range(max_path_length):
            # Distance from preferred length
            dist = abs(l + 1 - mu)
            # Attention weight decreases with distance
            weight = np.exp(-dist**2)
            path_attentions[:, :, n, l] = weight
    
    # Normalize attention weights
    path_attentions = path_attentions / path_attentions.sum(axis=3, keepdims=True)
    
    # Add some time variation
    for t in range(time_steps):
        # Add time-dependent noise
        noise = np.random.normal(0, 0.1, (batch_size, num_nodes, max_path_length))
        path_attentions[:, t, :, :] += (t / time_steps) * noise
        # Re-normalize
        path_attentions[:, t, :, :] = path_attentions[:, t, :, :] / path_attentions[:, t, :, :].sum(axis=2, keepdims=True)
    
    # Create synthetic node embeddings
    node_embeddings = np.random.normal(0, 1, (batch_size, time_steps, num_nodes, 64))
    
    print("\nRunning analysis...")
    
    # Visualize path attention for specific nodes
    visualize_path_attention_per_node(path_attentions, args.output_dir)
    
    # Analyze path length distribution
    analyze_path_length_distribution(path_attentions, args.output_dir)
    
    # Cluster nodes by attention profile
    cluster_nodes_by_attention(path_attentions, args.num_clusters, args.output_dir)
    
    # Visualize relationship with structural properties
    visualize_attention_vs_structural(path_attentions, node_embeddings, args.output_dir)
    
    print(f"\nAll analyses completed. Results saved to {args.output_dir}")


if __name__ == '__main__':
    main() 