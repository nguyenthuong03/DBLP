#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/12/19
# @Author  : Adaptive CP-GNN Implementation
# @File    : adaptive_mlp.py
# @Describe: MLP module for predicting adaptive context path weights

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveWeightMLP(nn.Module):
    """
    Multi-Layer Perceptron for predicting adaptive context path weights.
    
    Takes structural features of nodes as input and outputs normalized weights
    for different context path lengths.
    """
    
    def __init__(self, input_dim, K_length, hidden_dims=[64, 32], dropout=0.3):
        """
        Initialize the Adaptive Weight MLP.
        
        Args:
            input_dim: Dimension of input structural features
            K_length: Number of context path lengths (output dimension)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate for regularization
        """
        super(AdaptiveWeightMLP, self).__init__()
        
        self.input_dim = input_dim
        self.K_length = K_length
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer (no activation, will apply softmax later)
        layers.append(nn.Linear(prev_dim, K_length))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, structural_features):
        """
        Forward pass to predict context path weights.
        
        Args:
            structural_features: Tensor of shape (num_nodes, input_dim)
            
        Returns:
            weights: Tensor of shape (num_nodes, K_length) with softmax-normalized weights
        """
        # Pass through MLP
        logits = self.mlp(structural_features)
        
        # Apply softmax to ensure weights sum to 1 for each node
        weights = F.softmax(logits, dim=1)
        
        return weights
    
    def predict_single_node(self, node_features):
        """
        Predict weights for a single node.
        
        Args:
            node_features: Tensor of shape (input_dim,)
            
        Returns:
            weights: Tensor of shape (K_length,)
        """
        if node_features.dim() == 1:
            node_features = node_features.unsqueeze(0)
        
        weights = self.forward(node_features)
        return weights.squeeze(0)


class AdaptiveContextAggregator(nn.Module):
    """
    Module that combines context embeddings using adaptive weights.
    """
    
    def __init__(self, embedding_dim):
        """
        Initialize the adaptive context aggregator.
        
        Args:
            embedding_dim: Dimension of context embeddings
        """
        super(AdaptiveContextAggregator, self).__init__()
        self.embedding_dim = embedding_dim
    
    def forward(self, context_embeddings, adaptive_weights):
        """
        Aggregate context embeddings using adaptive weights.
        
        Args:
            context_embeddings: List of tensors, each of shape (num_nodes, embedding_dim)
                               representing context embeddings for different k-hop lengths
            adaptive_weights: Tensor of shape (num_nodes, K_length) with adaptive weights
            
        Returns:
            aggregated_embedding: Tensor of shape (num_nodes, embedding_dim)
        """
        # Stack context embeddings: (num_nodes, K_length, embedding_dim)
        stacked_embeddings = torch.stack(context_embeddings, dim=1)
        
        # Expand weights to match embedding dimensions: (num_nodes, K_length, 1)
        expanded_weights = adaptive_weights.unsqueeze(-1)
        
        # Weighted sum: (num_nodes, embedding_dim)
        aggregated_embedding = torch.sum(stacked_embeddings * expanded_weights, dim=1)
        
        return aggregated_embedding 