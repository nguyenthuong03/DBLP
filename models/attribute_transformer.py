#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Describe: Transformer modules for node attributes

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention module"""
    
    def __init__(self, hidden_dim, num_heads, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        # Linear projections and reshape for multi-head attention
        query = self.query_proj(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_proj(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_proj(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attn_weights, value)
        
        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        output = self.output_proj(context)
        
        return output, attn_weights


class PositionwiseFeedForward(nn.Module):
    """Feed-forward network with residual connection"""
    
    def __init__(self, hidden_dim, ff_dim, dropout_rate=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        output = self.linear2(self.dropout(F.relu(self.linear1(x))))
        return output


class TransformerLayer(nn.Module):
    """Single transformer encoder layer with self-attention and feed-forward network"""
    
    def __init__(self, hidden_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerLayer, self).__init__()
        self.self_attn = MultiHeadAttention(hidden_dim, num_heads, dropout_rate)
        self.feed_forward = PositionwiseFeedForward(hidden_dim, ff_dim, dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer normalization
        attn_output, attn_weights = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        output = self.norm2(x + self.dropout(ff_output))
        
        return output, attn_weights


class NodeAttributeTransformer(nn.Module):
    """Transformer encoder for processing node attributes"""
    
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, ff_dim, dropout_rate=0.1):
        super(NodeAttributeTransformer, self).__init__()
        
        # Embedding layer to project input attributes to hidden dimension
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_dim, num_heads, ff_dim, dropout_rate)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x, mask=None):
        # Input shape: [batch_size, seq_len, input_dim]
        
        # Project input to hidden dimension
        x = self.embedding(x)
        x = self.dropout(x)
        
        # Apply transformer layers
        attn_weights_list = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attn_weights_list.append(attn_weights)
        
        return x, attn_weights_list


class AttributeEncoder(nn.Module):
    """Encoder for different types of node attributes (text, numeric, categorical)"""
    
    def __init__(self, attr_types, attr_dims, hidden_dim):
        super(AttributeEncoder, self).__init__()
        
        self.attr_types = attr_types
        self.encoders = nn.ModuleDict()
        
        for attr_type, dim in zip(attr_types, attr_dims):
            if attr_type == 'text':
                # For text attributes, we'll use a dense layer
                self.encoders[attr_type] = nn.Linear(dim, hidden_dim)
            elif attr_type == 'numeric':
                # For numeric attributes, simple linear projection
                self.encoders[attr_type] = nn.Linear(dim, hidden_dim)
            elif attr_type == 'categorical':
                # For categorical features, use embedding
                self.encoders[attr_type] = nn.Linear(dim, hidden_dim)
    
    def forward(self, attrs_dict):
        """
        Process different types of attributes and concatenate them
        
        Args:
            attrs_dict: Dictionary with attribute types as keys and tensor attributes as values
        
        Returns:
            Tensor with encoded attributes
        """
        encoded_attrs = []
        
        for attr_type in self.attr_types:
            if attr_type in attrs_dict:
                encoded = self.encoders[attr_type](attrs_dict[attr_type])
                encoded_attrs.append(encoded)
        
        # Concatenate all encoded attributes
        if len(encoded_attrs) > 0:
            return torch.cat(encoded_attrs, dim=-1)
        else:
            return None


class AttributeFusion(nn.Module):
    """Fusion module to combine structural and attribute embeddings"""
    
    def __init__(self, hidden_dim, fusion_type='concat'):
        super(AttributeFusion, self).__init__()
        self.fusion_type = fusion_type
        
        if fusion_type == 'concat':
            self.fusion_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        elif fusion_type == 'gate':
            self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        elif fusion_type == 'cross_attention':
            self.cross_attn = MultiHeadAttention(hidden_dim, num_heads=4)
            self.norm = nn.LayerNorm(hidden_dim)
        else:
            raise ValueError(f"Unsupported fusion type: {fusion_type}")
    
    def forward(self, struct_emb, attr_emb):
        """
        Fuse structural and attribute embeddings
        
        Args:
            struct_emb: Structural embeddings from CP-GNN
            attr_emb: Attribute embeddings from transformer
        
        Returns:
            Fused embeddings
        """
        if self.fusion_type == 'concat':
            # Concatenate and project
            concat_emb = torch.cat([struct_emb, attr_emb], dim=-1)
            return self.fusion_layer(concat_emb)
        
        elif self.fusion_type == 'gate':
            # Gating mechanism
            concat_emb = torch.cat([struct_emb, attr_emb], dim=-1)
            gate = torch.sigmoid(self.gate(concat_emb))
            return gate * struct_emb + (1 - gate) * attr_emb
        
        elif self.fusion_type == 'cross_attention':
            # Cross-attention mechanism
            # Use structural embeddings as query and attribute embeddings as key/value
            output, _ = self.cross_attn(struct_emb.unsqueeze(0), attr_emb.unsqueeze(0))
            output = output.squeeze(0)
            
            # Add residual connection and normalization
            return self.norm(struct_emb + output) 