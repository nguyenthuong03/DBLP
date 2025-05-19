#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Describe: Dynamic CP-GNN model with GRU/LSTM for dynamic graphs

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.attribute_transformer import NodeAttributeTransformer, AttributeFusion


class TemporalAttention(nn.Module):
    """Temporal attention mechanism to focus on important time steps"""
    
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5
    
    def forward(self, query, keys, values):
        """
        Args:
            query: Current hidden state [batch_size, num_nodes, hidden_dim]
            keys: Hidden states from previous time steps [batch_size, time_steps, num_nodes, hidden_dim]
            values: Hidden states from previous time steps [batch_size, time_steps, num_nodes, hidden_dim]
        
        Returns:
            Weighted context vector
        """
        # Project query, keys, and values
        query = self.query_proj(query).unsqueeze(1)  # [batch_size, 1, num_nodes, hidden_dim]
        keys = self.key_proj(keys)  # [batch_size, time_steps, num_nodes, hidden_dim]
        values = self.value_proj(values)  # [batch_size, time_steps, num_nodes, hidden_dim]
        
        # Calculate attention scores
        scores = torch.matmul(query, keys.transpose(-1, -2)) / self.scale  # [batch_size, 1, num_nodes, time_steps]
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)  # [batch_size, 1, num_nodes, time_steps]
        
        # Apply attention weights to values
        context = torch.matmul(attn_weights, values)  # [batch_size, 1, num_nodes, hidden_dim]
        
        return context.squeeze(1)  # [batch_size, num_nodes, hidden_dim]


class GRUCell(nn.Module):
    """GRU cell for node embeddings"""
    
    def __init__(self, input_dim, hidden_dim):
        super(GRUCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # GRU gates
        self.reset_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.new_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
    
    def forward(self, x, h_prev):
        """
        Args:
            x: Input features [batch_size, num_nodes, input_dim]
            h_prev: Previous hidden state [batch_size, num_nodes, hidden_dim]
        
        Returns:
            Updated hidden state
        """
        # Concatenate input and previous hidden state
        combined = torch.cat([x, h_prev], dim=-1)
        
        # Compute reset gate
        r = torch.sigmoid(self.reset_gate(combined))
        
        # Compute update gate
        z = torch.sigmoid(self.update_gate(combined))
        
        # Compute candidate hidden state
        combined_reset = torch.cat([x, r * h_prev], dim=-1)
        h_tilde = torch.tanh(self.new_gate(combined_reset))
        
        # Compute new hidden state
        h_new = (1 - z) * h_prev + z * h_tilde
        
        return h_new


class LSTMCell(nn.Module):
    """LSTM cell for node embeddings"""
    
    def __init__(self, input_dim, hidden_dim):
        super(LSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # LSTM gates
        self.forget_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.cell_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
    
    def forward(self, x, states):
        """
        Args:
            x: Input features [batch_size, num_nodes, input_dim]
            states: Tuple of (h_prev, c_prev) where:
                    h_prev: Previous hidden state [batch_size, num_nodes, hidden_dim]
                    c_prev: Previous cell state [batch_size, num_nodes, hidden_dim]
        
        Returns:
            Tuple of (h_new, c_new)
        """
        h_prev, c_prev = states
        
        # Concatenate input and previous hidden state
        combined = torch.cat([x, h_prev], dim=-1)
        
        # Compute forget gate
        f = torch.sigmoid(self.forget_gate(combined))
        
        # Compute input gate
        i = torch.sigmoid(self.input_gate(combined))
        
        # Compute cell gate
        g = torch.tanh(self.cell_gate(combined))
        
        # Compute output gate
        o = torch.sigmoid(self.output_gate(combined))
        
        # Update cell state
        c_new = f * c_prev + i * g
        
        # Update hidden state
        h_new = o * torch.tanh(c_new)
        
        return h_new, c_new


class AdaptiveChannelWeight(nn.Module):
    """Adaptive weighting of channels based on temporal information"""
    
    def __init__(self, hidden_dim, num_channels):
        super(AdaptiveChannelWeight, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_channels),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        """
        Args:
            x: Node embeddings [batch_size, num_nodes, hidden_dim]
        
        Returns:
            Channel weights [batch_size, num_nodes, num_channels]
        """
        return self.channel_attention(x)


class DynamicCPGNN(nn.Module):
    """Dynamic CP-GNN with temporal modeling for dynamic graphs"""
    
    def __init__(self, input_dim, hidden_dim, num_channels, num_layers, 
                 num_heads, ff_dim, rnn_type='gru', use_temporal_attention=True,
                 adaptive_channel=True, dropout_rate=0.1):
        super(DynamicCPGNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type
        self.use_temporal_attention = use_temporal_attention
        self.adaptive_channel = adaptive_channel
        
        # Assume CP-GNN is defined elsewhere and accessible
        # This would be the static encoder for each time step
        # self.cp_gnn = CPGNN(input_dim, hidden_dim, num_channels)
        
        # Node attribute transformer
        self.attribute_transformer = NodeAttributeTransformer(
            input_dim, hidden_dim, num_layers, num_heads, ff_dim, dropout_rate
        )
        
        # Fusion module to combine structural and attribute embeddings
        self.fusion = AttributeFusion(hidden_dim, fusion_type='cross_attention')
        
        # Temporal module (GRU or LSTM)
        if rnn_type == 'gru':
            self.rnn_cell = GRUCell(hidden_dim, hidden_dim)
        elif rnn_type == 'lstm':
            self.rnn_cell = LSTMCell(hidden_dim, hidden_dim)
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
        
        # Temporal attention if enabled
        if use_temporal_attention:
            self.temporal_attention = TemporalAttention(hidden_dim)
        
        # Adaptive channel weighting if enabled
        if adaptive_channel:
            self.channel_weight = AdaptiveChannelWeight(hidden_dim, num_channels)
        
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
            Sequence of node embeddings at each time step
        """
        if time_steps is None:
            time_steps = len(graph_sequence)
        
        batch_size = graph_sequence[0][0].size(0)  # Assuming graph_sequence[t][c] has shape [batch_size, num_nodes, num_nodes]
        num_nodes = graph_sequence[0][0].size(1)
        
        # Initialize hidden states
        if self.rnn_type == 'gru':
            h = torch.zeros(batch_size, num_nodes, self.hidden_dim, device=graph_sequence[0][0].device)
            states = h
        else:  # lstm
            h = torch.zeros(batch_size, num_nodes, self.hidden_dim, device=graph_sequence[0][0].device)
            c = torch.zeros(batch_size, num_nodes, self.hidden_dim, device=graph_sequence[0][0].device)
            states = (h, c)
        
        # Storage for output embeddings at each time step
        outputs = []
        
        # Storage for hidden states for temporal attention
        hidden_states = []
        
        # Process each time step
        for t in range(time_steps):
            # Get current graph and attributes
            current_graph = graph_sequence[t]  # List of adjacency matrices for each channel
            current_attr = attr_sequence[t]  # Node attributes
            
            # Process node attributes with transformer
            attr_emb, _ = self.attribute_transformer(current_attr)
            
            # Assume we have a CP-GNN function that processes the graph structure
            # struct_emb = self.cp_gnn(current_graph)
            
            # For demonstration, we'll just use a placeholder
            # In practice, you would call your actual CP-GNN model here
            struct_emb = torch.rand_like(attr_emb)  # Placeholder
            
            # Apply adaptive channel weighting if enabled
            if self.adaptive_channel:
                channel_weights = self.channel_weight(states if self.rnn_type == 'gru' else states[0])
                # Apply weights to each channel's output in struct_emb
                # This would depend on how your CP-GNN outputs channel information
            
            # Fuse structural and attribute embeddings
            fused_emb = self.fusion(struct_emb, attr_emb)
            
            # Apply temporal attention if enabled
            if self.use_temporal_attention and t > 0:
                # Convert hidden_states list to tensor
                prev_hidden = torch.stack([h for h in hidden_states], dim=1)
                # Current query is either h or h from (h,c)
                current_query = states if self.rnn_type == 'gru' else states[0]
                # Apply temporal attention
                context = self.temporal_attention(current_query, prev_hidden, prev_hidden)
                # Combine with current embeddings
                fused_emb = fused_emb + context
            
            # Update RNN state
            if self.rnn_type == 'gru':
                states = self.rnn_cell(fused_emb, states)
                hidden_states.append(states)
            else:  # lstm
                states = self.rnn_cell(fused_emb, states)
                hidden_states.append(states[0])  # Only store h, not c
            
            # Apply output projection
            output = self.output_proj(states if self.rnn_type == 'gru' else states[0])
            output = self.dropout(output)
            
            # Store output
            outputs.append(output)
        
        # Stack outputs into a single tensor [batch_size, time_steps, num_nodes, hidden_dim]
        return torch.stack(outputs, dim=1)


class DynamicLinkPrediction(nn.Module):
    """Dynamic link prediction task based on Dynamic CP-GNN embeddings"""
    
    def __init__(self, hidden_dim):
        super(DynamicLinkPrediction, self).__init__()
        
        # MLP for link prediction
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, node_embeddings, edge_indices):
        """
        Predict link probability
        
        Args:
            node_embeddings: Node embeddings from Dynamic CP-GNN [batch_size, time_steps, num_nodes, hidden_dim]
            edge_indices: Indices of edges to predict [batch_size, num_edges, 2]
        
        Returns:
            Predicted link probabilities
        """
        # Get the last time step embeddings
        last_embeddings = node_embeddings[:, -1]  # [batch_size, num_nodes, hidden_dim]
        
        batch_size = last_embeddings.size(0)
        num_edges = edge_indices.size(1)
        
        # Get embeddings for source and target nodes
        source_embeddings = torch.gather(
            last_embeddings, 1, 
            edge_indices[:, :, 0].unsqueeze(-1).expand(-1, -1, last_embeddings.size(-1))
        )  # [batch_size, num_edges, hidden_dim]
        
        target_embeddings = torch.gather(
            last_embeddings, 1, 
            edge_indices[:, :, 1].unsqueeze(-1).expand(-1, -1, last_embeddings.size(-1))
        )  # [batch_size, num_edges, hidden_dim]
        
        # Concatenate source and target embeddings
        edge_embeddings = torch.cat([source_embeddings, target_embeddings], dim=-1)  # [batch_size, num_edges, hidden_dim*2]
        
        # Predict link probability
        link_scores = self.link_predictor(edge_embeddings).squeeze(-1)  # [batch_size, num_edges]
        
        return link_scores


class DynamicNodeClassification(nn.Module):
    """Dynamic node classification task based on Dynamic CP-GNN embeddings"""
    
    def __init__(self, hidden_dim, num_classes):
        super(DynamicNodeClassification, self).__init__()
        
        # MLP for node classification
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, node_embeddings):
        """
        Classify nodes
        
        Args:
            node_embeddings: Node embeddings from Dynamic CP-GNN [batch_size, time_steps, num_nodes, hidden_dim]
        
        Returns:
            Node class logits
        """
        # Get the last time step embeddings
        last_embeddings = node_embeddings[:, -1]  # [batch_size, num_nodes, hidden_dim]
        
        # Apply classifier
        logits = self.classifier(last_embeddings)  # [batch_size, num_nodes, num_classes]
        
        return logits


class TemporalNodeEmbeddingDecoder(nn.Module):
    """Decoder to predict future node embeddings based on temporal patterns"""
    
    def __init__(self, hidden_dim, forecast_horizon):
        super(TemporalNodeEmbeddingDecoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon
        
        # Decoder GRU
        self.decoder_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, node_embeddings):
        """
        Predict future node embeddings
        
        Args:
            node_embeddings: Node embeddings from Dynamic CP-GNN [batch_size, time_steps, num_nodes, hidden_dim]
        
        Returns:
            Predicted future node embeddings [batch_size, forecast_horizon, num_nodes, hidden_dim]
        """
        batch_size = node_embeddings.size(0)
        num_nodes = node_embeddings.size(2)
        
        # Reshape for GRU: [batch_size * num_nodes, time_steps, hidden_dim]
        reshaped_embeddings = node_embeddings.transpose(1, 2).reshape(-1, node_embeddings.size(1), self.hidden_dim)
        
        # Get last hidden state from GRU
        _, h_n = self.decoder_gru(reshaped_embeddings)  # h_n: [1, batch_size * num_nodes, hidden_dim]
        
        # Initialize predictions with last hidden state
        predictions = []
        
        # Autoregressive decoding
        current_input = reshaped_embeddings[:, -1].unsqueeze(1)  # [batch_size * num_nodes, 1, hidden_dim]
        
        for _ in range(self.forecast_horizon):
            # Predict next embedding
            output, h_n = self.decoder_gru(current_input, h_n)
            
            # Apply output projection
            next_embedding = self.output_proj(output.squeeze(1))  # [batch_size * num_nodes, hidden_dim]
            
            # Store prediction
            predictions.append(next_embedding)
            
            # Update input for next step
            current_input = next_embedding.unsqueeze(1)
        
        # Stack predictions and reshape back to [batch_size, forecast_horizon, num_nodes, hidden_dim]
        stacked_predictions = torch.stack(predictions, dim=1)  # [batch_size * num_nodes, forecast_horizon, hidden_dim]
        future_embeddings = stacked_predictions.reshape(batch_size, num_nodes, self.forecast_horizon, self.hidden_dim).transpose(1, 2)
        
        return future_embeddings 