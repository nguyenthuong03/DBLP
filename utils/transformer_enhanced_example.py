#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Describe: Example script to demonstrate transformer-enhanced features

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score

# Import our transformer enhanced features module
from utils.transformer_enhanced import (
    load_dblp_data_with_enhanced_features, 
    save_features, 
    load_features
)

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def main():
    print("Loading DBLP dataset with transformer-enhanced features...")
    
    # Create a minimal data config
    data_config = {
        'data_path': 'data',
        'dataset': 'DBLP',
        'data_name': 'DBLP.mat',
        'primary_type': 'a',  # Author
        'test_ratio': 0.2,
        'random_seed': 42,
        'resample': False,
        'task': ['CF']
    }
    
    # Load the data with enhanced features
    graph, features, labels, num_classes, train_idx, test_idx = load_dblp_data_with_enhanced_features(
        data_config, remove_self_loop=False
    )
    
    print(f"Loaded graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    print(f"Enhanced features shape: {features.shape}")
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(train_idx)}")
    print(f"Testing samples: {len(test_idx)}")
    
    # Cache the features for faster loading next time
    os.makedirs('cache', exist_ok=True)
    features_path = 'cache/dblp_enhanced_features.pkl'
    save_features(features, features_path)
    print(f"Saved features to {features_path}")
    
    # Visualize the features using t-SNE
    print("Creating t-SNE visualization of features...")
    visualize_features(features, labels, num_classes, output_file='results/tsne_visualization.png')
    
    # Train a simple model
    print("Training a simple classifier on the enhanced features...")
    train_simple_classifier(features, labels, train_idx, test_idx, num_classes)

def visualize_features(features, labels, num_classes, output_file='tsne_visualization.png'):
    """Visualize features using t-SNE"""
    # Use a sample of features to speed up t-SNE
    max_samples = min(5000, len(features))
    indices = np.random.choice(len(features), max_samples, replace=False)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
    features_sample = features[indices].numpy() if isinstance(features, torch.Tensor) else features[indices]
    tsne_results = tsne.fit_transform(features_sample)
    
    # Prepare labels for visualization
    labels_sample = labels[indices] if isinstance(labels, torch.Tensor) else labels[indices]
    
    # Plot results
    plt.figure(figsize=(10, 8))
    for class_idx in range(num_classes):
        class_mask = labels_sample == class_idx
        plt.scatter(
            tsne_results[class_mask, 0],
            tsne_results[class_mask, 1],
            label=f'Class {class_idx}',
            alpha=0.6
        )
    
    plt.legend()
    plt.title("t-SNE visualization of transformer-enhanced features")
    plt.tight_layout()
    
    # Save the visualization
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file)
    print(f"Visualization saved to {output_file}")

def train_simple_classifier(features, labels, train_idx, test_idx, num_classes):
    """Train a simple classifier on the enhanced features"""
    # Convert to PyTorch tensors if not already
    if not isinstance(features, torch.Tensor):
        features = torch.tensor(features, dtype=torch.float32)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=torch.long)
    
    # Create a simple classifier
    input_dim = features.shape[1]
    model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(256, num_classes)
    )
    
    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(features[train_idx])
        loss = criterion(outputs, labels[train_idx])
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Evaluate
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                # Training accuracy
                _, predicted = torch.max(outputs, 1)
                train_acc = (predicted == labels[train_idx]).sum().item() / len(train_idx)
                
                # Test accuracy
                test_outputs = model(features[test_idx])
                _, predicted = torch.max(test_outputs, 1)
                test_acc = (predicted == labels[test_idx]).sum().item() / len(test_idx)
                
                # Calculate NMI score
                test_labels = labels[test_idx].numpy()
                test_preds = predicted.numpy()
                nmi = normalized_mutual_info_score(test_labels, test_preds)
                
                print(f"Epoch {epoch+1}/{num_epochs}, "
                      f"Loss: {loss.item():.4f}, "
                      f"Train Acc: {train_acc:.4f}, "
                      f"Test Acc: {test_acc:.4f}, "
                      f"NMI: {nmi:.4f}")

if __name__ == "__main__":
    main() 