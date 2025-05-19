#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Describe: Adaptive CP-GNN with MLP for context path length adaptation

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ContextPathLengthPredictor(nn.Module):
    """MLP to predict appropriate context path length for each node"""
    
    def __init__(self, feature_dim, structural_dim, hidden_dim, max_path_length, use_structural=True, dropout_rate=0.1):
        """
        Args:
            feature_dim: Dimension of node features
            structural_dim: Dimension of structural properties (degree, centrality, etc.)
            hidden_dim: Hidden dimension for MLP
            max_path_length: Maximum context path length to consider
            use_structural: Whether to use structural properties as input
            dropout_rate: Dropout rate for MLP
        """
        super(ContextPathLengthPredictor, self).__init__()
        
        self.use_structural = use_structural
        self.max_path_length = max_path_length
        
        # Input dimension depends on whether structural properties are used
        input_dim = feature_dim
        if use_structural:
            input_dim += structural_dim
        
        # MLP to predict path length scores
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, max_path_length)
        )
        
        # Temperature parameter for softmax
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, node_features, structural_properties=None):
        """
        Predict context path length scores for each node
        
        Args:
            node_features: Node features [batch_size, num_nodes, feature_dim]
            structural_properties: Structural properties [batch_size, num_nodes, structural_dim]
        
        Returns:
            Path length scores [batch_size, num_nodes, max_path_length]
        """
        # Concatenate features and structural properties if available
        if self.use_structural and structural_properties is not None:
            inputs = torch.cat([node_features, structural_properties], dim=-1)
        else:
            inputs = node_features
        
        # Apply MLP
        path_length_scores = self.mlp(inputs)
        
        # Apply softmax with temperature to get attention weights
        attention_weights = F.softmax(path_length_scores / self.temperature, dim=-1)
        
        return attention_weights


class AdaptiveContextAggregation(nn.Module):
    """Aggregate context path embeddings with adaptive weights"""
    
    def __init__(self):
        super(AdaptiveContextAggregation, self).__init__()
    
    def forward(self, path_embeddings, attention_weights):
        """
        Aggregate embeddings from different path lengths using attention weights
        
        Args:
            path_embeddings: List of embeddings for each path length 
                             [max_path_length, batch_size, num_nodes, hidden_dim]
            attention_weights: Attention weights for each path length 
                              [batch_size, num_nodes, max_path_length]
        
        Returns:
            Aggregated embeddings [batch_size, num_nodes, hidden_dim]
        """
        # Stack path embeddings along a new dimension
        stacked_embeddings = torch.stack(path_embeddings, dim=1)  # [batch_size, max_path_length, num_nodes, hidden_dim]
        
        # Transpose to align with attention weights
        stacked_embeddings = stacked_embeddings.transpose(1, 2)  # [batch_size, num_nodes, max_path_length, hidden_dim]
        
        # Apply attention weights (broadcasting along hidden_dim)
        attention_weights = attention_weights.unsqueeze(-1)  # [batch_size, num_nodes, max_path_length, 1]
        weighted_embeddings = stacked_embeddings * attention_weights  # [batch_size, num_nodes, max_path_length, hidden_dim]
        
        # Sum along path length dimension
        aggregated_embeddings = weighted_embeddings.sum(dim=2)  # [batch_size, num_nodes, hidden_dim]
        
        return aggregated_embeddings


class StructuralFeatureExtractor(nn.Module):
    """Extract structural features from graph"""
    
    def __init__(self, output_dim):
        super(StructuralFeatureExtractor, self).__init__()
        
        self.output_dim = output_dim
        
        # Linear transformation to get fixed-size output
        self.transform = nn.Linear(3, output_dim)  # 3 features: degree, clustering, centrality
    
    def forward(self, graph):
        """
        Extract structural features from graph
        
        Args:
            graph: Graph structure with adjacency matrices
        
        Returns:
            Structural features [batch_size, num_nodes, output_dim]
        """
        batch_size = graph[0].size(0)
        num_nodes = graph[0].size(1)
        
        # Extract basic structural features
        # 1. Node degree (sum of row in adjacency matrix)
        adj_sum = torch.sum(graph[0], dim=2)  # [batch_size, num_nodes]
        degree = adj_sum / (num_nodes - 1)  # Normalize by max possible degree
        
        # 2. Approximate clustering coefficient (using powers of adjacency)
        adj_sq = torch.bmm(graph[0], graph[0])  # [batch_size, num_nodes, num_nodes]
        # Get diagonal elements (number of paths of length 2 starting and ending at node)
        clustering = torch.diagonal(adj_sq, dim1=1, dim2=2) / (adj_sum * (adj_sum - 1) + 1e-6)  # [batch_size, num_nodes]
        
        # 3. Approximate centrality (using row and column sums as proxy)
        centrality = (torch.sum(graph[0], dim=1) + adj_sum) / (2 * (num_nodes - 1))  # [batch_size, num_nodes]
        
        # Stack features
        features = torch.stack([degree, clustering, centrality], dim=2)  # [batch_size, num_nodes, 3]
        
        # Transform to desired dimension
        transformed_features = self.transform(features)  # [batch_size, num_nodes, output_dim]
        
        return transformed_features


class AdaptiveCPGNN(nn.Module):
    """CP-GNN with adaptive context path length using MLP"""
    
    def __init__(self, input_dim, hidden_dim, max_path_length, num_channels, 
                 structural_dim=16, use_structural=True, dropout_rate=0.1):
        super(AdaptiveCPGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_path_length = max_path_length
        self.num_channels = num_channels
        
        # Node embedding layer
        self.node_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Structural feature extractor
        if use_structural:
            self.structural_extractor = StructuralFeatureExtractor(structural_dim)
        self.use_structural = use_structural
        self.structural_dim = structural_dim
        
        # Context Path Length Predictor
        self.path_length_predictor = ContextPathLengthPredictor(
            feature_dim=hidden_dim,
            structural_dim=structural_dim,
            hidden_dim=hidden_dim,
            max_path_length=max_path_length,
            use_structural=use_structural,
            dropout_rate=dropout_rate
        )
        
        # Adaptive Context Aggregation
        self.context_aggregation = AdaptiveContextAggregation()
        
        # Message passing layers for each path length
        self.message_layers = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim) for _ in range(num_channels)
            ]) for _ in range(max_path_length)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, node_features, adj_matrices):
        """
        Forward pass through Adaptive CP-GNN
        
        Args:
            node_features: Node features [batch_size, num_nodes, input_dim]
            adj_matrices: List of adjacency matrices for each channel 
                         [num_channels, batch_size, num_nodes, num_nodes]
        
        Returns:
            Node embeddings [batch_size, num_nodes, hidden_dim]
        """
        batch_size, num_nodes, _ = node_features.size()
        
        # Initial node embeddings
        node_emb = self.node_embedding(node_features)  # [batch_size, num_nodes, hidden_dim]
        node_emb = F.relu(node_emb)
        
        # Extract structural features if enabled
        if self.use_structural:
            structural_features = self.structural_extractor(adj_matrices)  # [batch_size, num_nodes, structural_dim]
        else:
            structural_features = None
        
        # Predict path length attention weights
        path_attention = self.path_length_predictor(node_emb, structural_features)  # [batch_size, num_nodes, max_path_length]
        
        # Propagate messages for each path length
        path_embeddings = []
        
        for path_idx in range(self.max_path_length):
            path_emb = node_emb
            
            # Apply message passing up to current path length
            for hop in range(path_idx + 1):
                channel_embeddings = []
                
                # Process each channel
                for ch in range(self.num_channels):
                    # Get adjacency matrix for current channel
                    adj = adj_matrices[ch]  # [batch_size, num_nodes, num_nodes]
                    
                    # Apply message passing
                    messages = torch.bmm(adj, path_emb)  # [batch_size, num_nodes, hidden_dim]
                    messages = self.message_layers[path_idx][ch](messages)  # [batch_size, num_nodes, hidden_dim]
                    
                    channel_embeddings.append(messages)
                
                # Combine channel embeddings (simple mean for now)
                channel_combined = torch.stack(channel_embeddings, dim=0).mean(dim=0)  # [batch_size, num_nodes, hidden_dim]
                
                # Update embeddings
                path_emb = channel_combined
            
            path_embeddings.append(path_emb)
        
        # Aggregate embeddings with attention weights
        aggregated_emb = self.context_aggregation(path_embeddings, path_attention)  # [batch_size, num_nodes, hidden_dim]
        
        # Apply output projection and dropout
        output = self.output_proj(aggregated_emb)  # [batch_size, num_nodes, hidden_dim]
        output = self.dropout(output)
        
        return output, path_attention


class PathLengthRegularization(nn.Module):
    """Regularization for path length prediction"""
    
    def __init__(self, entropy_weight=0.1, diversity_weight=0.1):
        super(PathLengthRegularization, self).__init__()
        self.entropy_weight = entropy_weight
        self.diversity_weight = diversity_weight
    
    def forward(self, path_attention):
        """
        Compute regularization loss for path attention weights
        
        Args:
            path_attention: Path attention weights [batch_size, num_nodes, max_path_length]
        
        Returns:
            Regularization loss
        """
        # Entropy regularization to encourage exploration
        entropy = -torch.sum(path_attention * torch.log(path_attention + 1e-6), dim=-1).mean()
        entropy_loss = -self.entropy_weight * entropy  # Negative because we want to maximize entropy
        
        # Diversity regularization: encourage different nodes to use different path lengths
        mean_attention = path_attention.mean(dim=1, keepdim=True)  # [batch_size, 1, max_path_length]
        diversity = -torch.sum((path_attention - mean_attention) ** 2, dim=-1).mean()
        diversity_loss = -self.diversity_weight * diversity  # Negative because we want to maximize diversity
        
        return entropy_loss + diversity_loss


class AdaptiveDynamicCPGNN(nn.Module):
    """Dynamic CP-GNN with adaptive context path length"""
    
    def __init__(self, input_dim, hidden_dim, max_path_length, num_channels, 
                 num_layers, num_heads, ff_dim, rnn_type='gru', 
                 use_temporal_attention=True, structural_dim=16, 
                 use_structural=True, dropout_rate=0.1):
        super(AdaptiveDynamicCPGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_path_length = max_path_length
        self.num_channels = num_channels
        self.rnn_type = rnn_type
        self.use_temporal_attention = use_temporal_attention
        
        # Static Adaptive CP-GNN for each time step
        self.adaptive_cpgnn = AdaptiveCPGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            max_path_length=max_path_length,
            num_channels=num_channels,
            structural_dim=structural_dim,
            use_structural=use_structural,
            dropout_rate=dropout_rate
        )
        
        # Temporal module (GRU or LSTM)
        if rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                batch_first=True
            )
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                batch_first=True
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
        
        # Temporal attention if enabled
        if use_temporal_attention:
            self.temporal_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout_rate
            )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Regularization loss for path length prediction
        self.path_regularizer = PathLengthRegularization()
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, graph_sequence, attr_sequence, time_steps=None):
        """
        Process a sequence of dynamic graphs
        
        Args:
            graph_sequence: List of graph structures at each time step
                            Each element contains adjacency matrices for each channel
            attr_sequence: List of node attributes at each time step
            time_steps: Number of time steps to process (default: all)
        
        Returns:
            Sequence of node embeddings at each time step and path attention weights
        """
        if time_steps is None:
            time_steps = len(graph_sequence)
        
        batch_size = attr_sequence[0].size(0)
        num_nodes = attr_sequence[0].size(1)
        
        # Initialize hidden states for RNN
        h0 = torch.zeros(1, batch_size * num_nodes, self.hidden_dim, device=attr_sequence[0].device)
        if self.rnn_type == 'lstm':
            c0 = torch.zeros(1, batch_size * num_nodes, self.hidden_dim, device=attr_sequence[0].device)
            hidden = (h0, c0)
        else:
            hidden = h0
        
        # Storage for output embeddings and attention weights
        outputs = []
        all_path_attentions = []
        
        # Process each time step
        for t in range(time_steps):
            # Get current graph and attributes
            current_graph = graph_sequence[t]  # List of adjacency matrices for each channel
            current_attr = attr_sequence[t]  # Node attributes
            
            # Process with Adaptive CP-GNN
            node_emb, path_attention = self.adaptive_cpgnn(current_attr, current_graph)
            
            # Store path attention
            all_path_attentions.append(path_attention)
            
            # Reshape for RNN: [batch_size * num_nodes, 1, hidden_dim]
            reshaped_emb = node_emb.reshape(batch_size * num_nodes, 1, self.hidden_dim)
            
            # Process with RNN
            if self.rnn_type == 'lstm':
                rnn_out, hidden = self.rnn(reshaped_emb, hidden)
            else:
                rnn_out, hidden = self.rnn(reshaped_emb, hidden)
            
            # Apply temporal attention if enabled
            if self.use_temporal_attention and t > 0:
                # Reshape output for attention: [1, batch_size * num_nodes, hidden_dim]
                query = rnn_out.transpose(0, 1)
                
                # Stack previous outputs: [t, batch_size * num_nodes, hidden_dim]
                keys = torch.cat([out.transpose(0, 1) for out in outputs], dim=0)
                values = keys
                
                # Apply attention
                attn_out, _ = self.temporal_attention(query, keys, values)
                
                # Add attention output to current hidden state
                rnn_out = rnn_out + attn_out.transpose(0, 1)
            
            # Reshape back: [batch_size, num_nodes, hidden_dim]
            output = rnn_out.reshape(batch_size, num_nodes, self.hidden_dim)
            
            # Apply output projection and dropout
            output = self.output_proj(output)
            output = self.dropout(output)
            
            # Store output
            outputs.append(output)
        
        # Stack outputs into a single tensor [batch_size, time_steps, num_nodes, hidden_dim]
        stacked_outputs = torch.stack(outputs, dim=1)
        
        # Stack path attentions [batch_size, time_steps, num_nodes, max_path_length]
        stacked_attentions = torch.stack(all_path_attentions, dim=1)
        
        # Compute regularization loss
        reg_loss = sum(self.path_regularizer(attn) for attn in all_path_attentions)
        
        return stacked_outputs, stacked_attentions, reg_loss 