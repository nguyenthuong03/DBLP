#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Describe: Adaptive CP-GNN with Path Length-Aware MLP

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AdaptiveLengthMLP(nn.Module):
    """MLP that adapts its architecture based on context path length"""
    
    def __init__(self, input_dim, hidden_dims, max_path_length, dropout_rate=0.1):
        """
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden dimensions for base MLP
            max_path_length: Maximum path length to support
            dropout_rate: Dropout rate
        """
        super(AdaptiveLengthMLP, self).__init__()
        
        self.input_dim = input_dim
        self.max_path_length = max_path_length
        
        # Create different MLP configurations for different path lengths
        self.path_mlps = nn.ModuleList([
            self._create_mlp_for_length(input_dim, hidden_dims, length+1, dropout_rate) 
            for length in range(max_path_length)
        ])
    
    def _create_mlp_for_length(self, input_dim, hidden_dims, path_length, dropout_rate):
        """Create an MLP with architecture tailored to a specific path length"""
        # Scale hidden dimensions based on path length
        # Short paths: smaller network, long paths: deeper network
        if path_length <= 2:  # Short path
            dims = [max(dim // 2, input_dim // 2) for dim in hidden_dims[:2]]
        elif path_length <= 4:  # Medium path
            dims = hidden_dims[:3]
        else:  # Long path
            dims = hidden_dims + [hidden_dims[-1]]  # Add an extra layer for long paths
        
        layers = []
        prev_dim = input_dim
        
        for dim in dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim
        
        # Final output layer (same output dimension across all path lengths)
        layers.append(nn.Linear(prev_dim, hidden_dims[-1]))
        
        return nn.Sequential(*layers)
    
    def forward(self, x, path_lengths):
        """
        Apply appropriate MLP based on path length
        
        Args:
            x: Input features [batch_size, num_nodes, input_dim]
            path_lengths: Path length tensor (integer) [batch_size, num_nodes]
                         or one-hot encoded [batch_size, num_nodes, max_path_length]
        
        Returns:
            Processed features [batch_size, num_nodes, output_dim]
        """
        batch_size, num_nodes, _ = x.size()
        
        # If path_lengths is one-hot encoded, convert to indices
        if path_lengths.dim() == 3:
            path_lengths = torch.argmax(path_lengths, dim=2)  # [batch_size, num_nodes]
        
        # Clamp path lengths to valid range
        path_lengths = torch.clamp(path_lengths, 0, self.max_path_length - 1)
        
        # Initialize output tensor
        output = torch.zeros(batch_size, num_nodes, self.path_mlps[0][-1].out_features, 
                            device=x.device)
        
        # Process each node with its appropriate path-length MLP
        for length in range(self.max_path_length):
            # Create mask for nodes with this path length
            mask = (path_lengths == length)  # [batch_size, num_nodes]
            
            if not mask.any():
                continue
                
            # Get nodes with this path length
            masked_indices = mask.nonzero(as_tuple=True)
            masked_x = x[masked_indices[0], masked_indices[1]]  # [num_selected, input_dim]
            
            # Apply corresponding MLP
            masked_output = self.path_mlps[length](masked_x)  # [num_selected, output_dim]
            
            # Put results back in output tensor
            output[masked_indices[0], masked_indices[1]] = masked_output
        
        return output


class PathLengthAttention(nn.Module):
    """Attention mechanism that weights path embeddings based on their length"""
    
    def __init__(self, hidden_dim, max_path_length, num_heads=4, dropout_rate=0.1):
        """
        Args:
            hidden_dim: Hidden dimension
            max_path_length: Maximum path length
            num_heads: Number of attention heads
            dropout_rate: Dropout rate
        """
        super(PathLengthAttention, self).__init__()
        
        self.max_path_length = max_path_length
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Attention layers
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(max_path_length)
        ])
        self.value_proj = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(max_path_length)
        ])
        
        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Path importance predictor (learns importance weights for each path length)
        self.path_importance = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, max_path_length),
            nn.Softmax(dim=-1)
        )
        
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, node_features, path_embeddings):
        """
        Apply path-length-aware attention
        
        Args:
            node_features: Node features [batch_size, num_nodes, hidden_dim]
            path_embeddings: List of path embeddings for each length
                            [max_path_length, batch_size, num_nodes, hidden_dim]
        
        Returns:
            Aggregated embeddings [batch_size, num_nodes, hidden_dim]
            Path attention weights [batch_size, num_nodes, max_path_length]
        """
        batch_size, num_nodes, _ = node_features.size()
        
        # Get query from node features
        queries = self.query_proj(node_features)  # [batch_size, num_nodes, hidden_dim]
        
        # Path importance weights - global importance of each path length
        path_weights = self.path_importance(node_features)  # [batch_size, num_nodes, max_path_length]
        
        # Process each path length with its dedicated key/value projections
        keys = []
        values = []
        
        for path_idx in range(self.max_path_length):
            # Get embeddings for current path length
            path_emb = path_embeddings[path_idx]  # [batch_size, num_nodes, hidden_dim]
            
            # Project to keys and values
            key = self.key_proj[path_idx](path_emb)  # [batch_size, num_nodes, hidden_dim]
            value = self.value_proj[path_idx](path_emb)  # [batch_size, num_nodes, hidden_dim]
            
            # Apply path importance weights
            path_weight = path_weights[:, :, path_idx].unsqueeze(-1)  # [batch_size, num_nodes, 1]
            weighted_key = key * path_weight
            weighted_value = value * path_weight
            
            keys.append(weighted_key)
            values.append(weighted_value)
        
        # Concatenate along a new dimension
        keys = torch.stack(keys, dim=1)  # [batch_size, max_path_length, num_nodes, hidden_dim]
        values = torch.stack(values, dim=1)  # [batch_size, max_path_length, num_nodes, hidden_dim]
        
        # Reshape for multi-head attention
        # Flatten max_path_length and num_nodes dimensions
        keys = keys.view(batch_size, self.max_path_length * num_nodes, -1)
        values = values.view(batch_size, self.max_path_length * num_nodes, -1)
        
        # Repeat queries for each path length
        queries_repeated = queries.unsqueeze(1).expand(-1, self.max_path_length, -1, -1)
        queries_repeated = queries_repeated.reshape(batch_size, self.max_path_length * num_nodes, -1)
        
        # Apply multi-head attention
        attn_output, _ = self.multihead_attn(
            query=queries_repeated,
            key=keys,
            value=values
        )
        
        # Reshape back and aggregate across path lengths
        attn_output = attn_output.view(batch_size, self.max_path_length, num_nodes, -1)
        attn_output = attn_output.sum(dim=1)  # [batch_size, num_nodes, hidden_dim]
        
        # Apply output projection and dropout
        output = self.output_proj(attn_output)
        output = self.dropout(output)
        
        return output, path_weights


class LengthAdaptivePooling(nn.Module):
    """Pooling mechanism that adapts based on path length"""
    
    def __init__(self, hidden_dim, max_path_length):
        """
        Args:
            hidden_dim: Hidden dimension
            max_path_length: Maximum path length
        """
        super(LengthAdaptivePooling, self).__init__()
        
        self.max_path_length = max_path_length
        
        # Different pooling strategies for different path lengths
        # Short paths: simple pooling
        self.short_pool = nn.AdaptiveMaxPool1d(1)
        
        # Medium paths: attention-based pooling
        self.medium_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Long paths: hierarchical pooling
        self.long_pool1 = nn.AdaptiveAvgPool1d(hidden_dim // 2)
        self.long_pool2 = nn.AdaptiveMaxPool1d(1)
    
    def forward(self, embeddings, path_lengths):
        """
        Apply appropriate pooling based on path length
        
        Args:
            embeddings: Node embeddings [batch_size, num_nodes, hidden_dim]
            path_lengths: Path length tensor [batch_size, num_nodes]
        
        Returns:
            Pooled embeddings [batch_size, num_nodes, hidden_dim]
        """
        batch_size, num_nodes, hidden_dim = embeddings.size()
        
        # Initialize output tensor
        output = torch.zeros_like(embeddings)
        
        # Apply different pooling strategies based on path length
        # Short paths (length 1-2)
        short_mask = (path_lengths < 3)
        if short_mask.any():
            short_indices = short_mask.nonzero(as_tuple=True)
            short_emb = embeddings[short_indices[0], short_indices[1]]  # [num_short, hidden_dim]
            
            # Apply max pooling
            short_emb = short_emb.unsqueeze(2)  # [num_short, hidden_dim, 1]
            short_emb = self.short_pool(short_emb).squeeze(2)  # [num_short, hidden_dim]
            
            output[short_indices[0], short_indices[1]] = short_emb
        
        # Medium paths (length 3-4)
        medium_mask = (path_lengths >= 3) & (path_lengths < 5)
        if medium_mask.any():
            medium_indices = medium_mask.nonzero(as_tuple=True)
            medium_emb = embeddings[medium_indices[0], medium_indices[1]]  # [num_medium, hidden_dim]
            
            # Apply attention pooling
            attention_scores = self.medium_attention(medium_emb)  # [num_medium, 1]
            attention_weights = F.softmax(attention_scores, dim=0)
            medium_pooled = (medium_emb * attention_weights).sum(dim=0, keepdim=True)
            
            output[medium_indices[0], medium_indices[1]] = medium_pooled
        
        # Long paths (length 5+)
        long_mask = (path_lengths >= 5)
        if long_mask.any():
            long_indices = long_mask.nonzero(as_tuple=True)
            long_emb = embeddings[long_indices[0], long_indices[1]]  # [num_long, hidden_dim]
            
            # Apply hierarchical pooling
            long_emb = long_emb.transpose(1, 2)  # [num_long, hidden_dim, hidden_dim]
            long_emb = self.long_pool1(long_emb)  # [num_long, hidden_dim, hidden_dim//2]
            long_emb = self.long_pool2(long_emb).squeeze(2)  # [num_long, hidden_dim]
            
            output[long_indices[0], long_indices[1]] = long_emb
        
        return output


class AdaptivePathLengthCPGNN(nn.Module):
    """CP-GNN with MLP adapting to context path length"""
    
    def __init__(self, input_dim, hidden_dim, hidden_dims, max_path_length, num_channels, 
                 num_heads=4, use_pooling=True, dropout_rate=0.1):
        """
        Args:
            input_dim: Input dimension of node features
            hidden_dim: Hidden dimension
            hidden_dims: List of hidden dimensions for adaptive MLP
            max_path_length: Maximum context path length
            num_channels: Number of propagation channels
            num_heads: Number of attention heads
            use_pooling: Whether to use length-adaptive pooling
            dropout_rate: Dropout rate
        """
        super(AdaptivePathLengthCPGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_path_length = max_path_length
        self.num_channels = num_channels
        self.use_pooling = use_pooling
        
        # Initial node embedding
        self.node_embedding = nn.Linear(input_dim, hidden_dim)
        
        # MLPs adaptive to path length
        self.adaptive_mlps = nn.ModuleList([
            AdaptiveLengthMLP(
                input_dim=hidden_dim, 
                hidden_dims=hidden_dims,
                max_path_length=max_path_length,
                dropout_rate=dropout_rate
            ) for _ in range(num_channels)
        ])
        
        # Message passing layers for each path length and channel
        self.message_layers = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim) for _ in range(num_channels)
            ]) for _ in range(max_path_length)
        ])
        
        # Path length attention mechanism
        self.path_attention = PathLengthAttention(
            hidden_dim=hidden_dim,
            max_path_length=max_path_length,
            num_heads=num_heads,
            dropout_rate=dropout_rate
        )
        
        # Length-adaptive pooling if enabled
        if use_pooling:
            self.length_pooling = LengthAdaptivePooling(
                hidden_dim=hidden_dim,
                max_path_length=max_path_length
            )
        
        # Path length predictor (predicts optimal path length for each node)
        self.path_length_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, max_path_length),
            nn.Softmax(dim=-1)
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Fusion layer to combine results from different channels
        self.channel_fusion = nn.Linear(num_channels * hidden_dim, hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, node_features, adj_matrices):
        """
        Forward pass through Adaptive Path Length CP-GNN
        
        Args:
            node_features: Node features [batch_size, num_nodes, input_dim]
            adj_matrices: List of adjacency matrices for each channel 
                         [num_channels, batch_size, num_nodes, num_nodes]
        
        Returns:
            Node embeddings [batch_size, num_nodes, hidden_dim]
            Path attention weights [batch_size, num_nodes, max_path_length]
        """
        batch_size, num_nodes, _ = node_features.size()
        
        # Initial node embeddings
        node_emb = self.node_embedding(node_features)  # [batch_size, num_nodes, hidden_dim]
        node_emb = F.relu(node_emb)
        
        # Predict optimal path length for each node
        path_lengths_prob = self.path_length_predictor(node_emb)  # [batch_size, num_nodes, max_path_length]
        path_lengths = torch.argmax(path_lengths_prob, dim=2)  # [batch_size, num_nodes]
        
        # Storage for path embeddings
        path_embeddings = [[] for _ in range(self.max_path_length)]
        
        # Process each channel separately
        channel_embeddings = []
        
        for ch in range(self.num_channels):
            # Get adjacency matrix for current channel
            adj = adj_matrices[ch]  # [batch_size, num_nodes, num_nodes]
            
            # Start with initial embeddings
            ch_emb = node_emb
            
            # Propagate for each path length
            for path_idx in range(self.max_path_length):
                # Apply message passing
                messages = torch.bmm(adj, ch_emb)  # [batch_size, num_nodes, hidden_dim]
                messages = self.message_layers[path_idx][ch](messages)  # [batch_size, num_nodes, hidden_dim]
                
                # Apply adaptive MLP based on path length
                processed_messages = self.adaptive_mlps[ch](messages, path_lengths)
                
                # Update embeddings
                ch_emb = processed_messages
                
                # Store embeddings for this path length
                path_embeddings[path_idx].append(ch_emb)
        
        # Combine embeddings from different channels for each path length
        combined_path_embeddings = []
        for path_idx in range(self.max_path_length):
            path_embs = torch.stack(path_embeddings[path_idx], dim=1)  # [batch_size, num_channels, num_nodes, hidden_dim]
            path_embs = path_embs.transpose(1, 2)  # [batch_size, num_nodes, num_channels, hidden_dim]
            path_embs = path_embs.reshape(batch_size, num_nodes, -1)  # [batch_size, num_nodes, num_channels*hidden_dim]
            
            # Apply channel fusion
            fused_emb = self.channel_fusion(path_embs)  # [batch_size, num_nodes, hidden_dim]
            fused_emb = F.relu(fused_emb)
            
            combined_path_embeddings.append(fused_emb)
        
        # Apply path-length-aware attention
        aggregated_emb, path_attention = self.path_attention(
            node_features=node_emb,
            path_embeddings=combined_path_embeddings
        )  # [batch_size, num_nodes, hidden_dim], [batch_size, num_nodes, max_path_length]
        
        # Apply length-adaptive pooling if enabled
        if self.use_pooling:
            aggregated_emb = self.length_pooling(aggregated_emb, path_lengths)
        
        # Apply output projection and dropout
        output = self.output_proj(aggregated_emb)  # [batch_size, num_nodes, hidden_dim]
        output = self.dropout(output)
        
        return output, path_attention


class AdaptivePathLengthDynamicCPGNN(nn.Module):
    """Dynamic CP-GNN with path length adaptive MLP"""
    
    def __init__(self, input_dim, hidden_dim, hidden_dims, max_path_length, num_channels, 
                 num_layers, num_heads, ff_dim, rnn_type='gru', 
                 use_temporal_attention=True, use_pooling=True, dropout_rate=0.1):
        """
        Args:
            input_dim: Input dimension of node features
            hidden_dim: Hidden dimension
            hidden_dims: List of hidden dimensions for adaptive MLP
            max_path_length: Maximum context path length
            num_channels: Number of propagation channels
            num_layers: Number of Transformer layers
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension in Transformer
            rnn_type: Type of RNN ('gru' or 'lstm')
            use_temporal_attention: Whether to use temporal attention
            use_pooling: Whether to use length-adaptive pooling
            dropout_rate: Dropout rate
        """
        super(AdaptivePathLengthDynamicCPGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_path_length = max_path_length
        self.num_channels = num_channels
        self.rnn_type = rnn_type
        self.use_temporal_attention = use_temporal_attention
        
        # Static Adaptive Path Length CP-GNN for each time step
        self.adaptive_cpgnn = AdaptivePathLengthCPGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            hidden_dims=hidden_dims,
            max_path_length=max_path_length,
            num_channels=num_channels,
            num_heads=num_heads,
            use_pooling=use_pooling,
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
                dropout=dropout_rate,
                batch_first=True
            )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
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
            
            # Process with Adaptive Path Length CP-GNN
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
        
        return stacked_outputs, stacked_attentions 