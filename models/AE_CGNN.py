#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Describe: Attribute-enhanced CP-GNN with Transformer for node attributes

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from .CGNN import ContextGNN
from .attribute_transformer import NodeAttributeTransformer, AttributeFusion


class ConditionalAttention(nn.Module):
    """Conditional attention module that adjusts attention weights based on node attributes"""
    
    def __init__(self, context_dim, attr_dim, hidden_dim):
        super(ConditionalAttention, self).__init__()
        self.attr_proj = nn.Linear(attr_dim, hidden_dim)
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        self.attention = nn.Linear(hidden_dim, 1)
    
    def forward(self, context_embeds, attr_embeds):
        """
        Args:
            context_embeds: Context path embeddings [num_paths, context_dim]
            attr_embeds: Node attribute embeddings [attr_dim]
        
        Returns:
            Attention weights for context paths
        """
        # Project attribute embeddings
        attr_proj = self.attr_proj(attr_embeds)  # [hidden_dim]
        
        # Project context embeddings
        context_proj = self.context_proj(context_embeds)  # [num_paths, hidden_dim]
        
        # Expand attribute projection to match context paths
        attr_proj = attr_proj.unsqueeze(0).expand_as(context_proj)  # [num_paths, hidden_dim]
        
        # Combine and compute attention
        combined = torch.tanh(context_proj + attr_proj)  # [num_paths, hidden_dim]
        attn_weights = F.softmax(self.attention(combined).squeeze(-1), dim=0)  # [num_paths]
        
        return attn_weights


class AttributeEnhancedCGNN(nn.Module):
    """Attribute-enhanced Context Path GNN with Transformer for node attributes"""
    
    def __init__(self, hg, config, node_attributes=None):
        super(AttributeEnhancedCGNN, self).__init__()
        
        # Initialize the base CP-GNN model
        self.base_model = ContextGNN(hg, config['base_model_config'])
        self.hg = hg
        self.config = config
        self.node_attributes = node_attributes
        
        # Get the primary node type from config
        self.primary_type = config['primary_type']
        self.embedding_dim = config['embedding_dim']
        
        # Create attribute transformer for primary node type if node attributes are provided
        if node_attributes is not None and self.primary_type in node_attributes:
            attr_config = config['attribute_config']
            
            # Get input dimension for node attributes
            attr_dim = node_attributes[self.primary_type].shape[1]
            
            # Create transformer for node attributes
            self.attribute_transformer = NodeAttributeTransformer(
                input_dim=attr_dim,
                hidden_dim=attr_config['hidden_dim'],
                num_layers=attr_config['num_layers'],
                num_heads=attr_config['num_heads'],
                ff_dim=attr_config['ff_dim'],
                dropout_rate=attr_config['dropout_rate']
            )
            
            # Create fusion module for combining structural and attribute embeddings
            self.fusion = AttributeFusion(
                hidden_dim=self.embedding_dim,
                fusion_type=attr_config['fusion_type']
            )
            
            # Create conditional attention module if enabled
            if attr_config['conditional_attention']:
                self.cond_attention = ConditionalAttention(
                    context_dim=self.embedding_dim,
                    attr_dim=attr_config['hidden_dim'],
                    hidden_dim=self.embedding_dim
                )
            else:
                self.cond_attention = None
            
            # Attribute reconstruction task if enabled
            if attr_config['attr_reconstruction']:
                self.attr_decoder = nn.Linear(self.embedding_dim, attr_dim)
            else:
                self.attr_decoder = None
        
    def forward(self, k_hop=None):
        """
        Forward pass through AE-CP-GNN
        
        Args:
            k_hop: Which k-hop path to generate embeddings for
        
        Returns:
            Context path embeddings enhanced with attribute information
        """
        # Get the base CP-GNN embeddings
        base_embeddings = self.base_model(k_hop)
        
        # If no node attributes are available, return base embeddings
        if self.node_attributes is None or self.primary_type not in self.node_attributes:
            return base_embeddings
        
        # Process node attributes with transformer
        node_attrs = self.node_attributes[self.primary_type]
        
        # Prepare node attributes for transformer (add batch dimension if needed)
        if len(node_attrs.shape) == 2:
            node_attrs = node_attrs.unsqueeze(0)  # [1, num_nodes, attr_dim]
        
        # Process attributes with transformer
        attr_embeds, _ = self.attribute_transformer(node_attrs)
        attr_embeds = attr_embeds.squeeze(0)  # [num_nodes, hidden_dim]
        
        # Apply conditional attention if enabled
        if self.cond_attention is not None and k_hop is not None:
            # Get context path embeddings for the specific k_hop
            context_embeds = self.base_model.get_context_embeddings(k_hop)
            
            # Get attribute embeddings for the primary nodes
            primary_attr_embeds = attr_embeds[self.hg.nodes(self.primary_type)]
            
            # Apply conditional attention to adjust weights based on attributes
            cond_attn_weights = self.cond_attention(context_embeds, primary_attr_embeds)
            
            # Apply the conditional attention weights to context embeddings
            context_embeds = context_embeds * cond_attn_weights.unsqueeze(1)
            
            # Update the embeddings in the base model
            self.base_model.update_context_embeddings(k_hop, context_embeds)
        
        # Fuse structural and attribute embeddings
        primary_embeds = self.base_model.primary_emb.weight
        primary_attr_embeds = attr_embeds[self.hg.nodes(self.primary_type)]
        
        # Ensure dimensions match
        if primary_embeds.shape != primary_attr_embeds.shape:
            primary_attr_embeds = F.pad(
                primary_attr_embeds, 
                (0, primary_embeds.shape[1] - primary_attr_embeds.shape[1], 0, 0)
            )
        
        # Fuse the embeddings
        fused_embeds = self.fusion(primary_embeds, primary_attr_embeds)
        
        # Replace the primary embeddings with fused embeddings
        self.base_model.primary_emb.weight.data = fused_embeds.data
        
        # Return the context embeddings for the specified k_hop
        if k_hop is not None:
            return self.base_model.context_path_emb[k_hop]
        else:
            return self.base_model.primary_emb.weight
    
    def get_loss(self, k_hop, pos_src, pos_dst, neg_src, neg_dst, p_emb, p_context_emb):
        """
        Get the combined loss for link prediction and attribute reconstruction (if enabled)
        
        Args:
            k_hop: Which k-hop path to use
            pos_src, pos_dst: Positive source and destination nodes
            neg_src, neg_dst: Negative source and destination nodes
            p_emb: Primary node embeddings
            p_context_emb: Context embeddings for primary nodes
        
        Returns:
            Combined loss
        """
        # Get the base CP-GNN loss for link prediction
        base_loss = self.base_model.get_loss(k_hop, pos_src, pos_dst, neg_src, neg_dst, p_emb, p_context_emb)
        
        # If attribute reconstruction is enabled, add the reconstruction loss
        if hasattr(self, 'attr_decoder') and self.attr_decoder is not None:
            # Get node attributes for the primary nodes
            node_attrs = self.node_attributes[self.primary_type]
            
            # Get primary node embeddings
            primary_embeds = p_emb
            
            # Reconstruct node attributes
            reconstructed_attrs = self.attr_decoder(primary_embeds)
            
            # Compute reconstruction loss (mean squared error)
            attr_loss = F.mse_loss(reconstructed_attrs, node_attrs)
            
            # Combine losses with weight from config
            lambda_attr = self.config['attribute_config']['lambda_attr']
            combined_loss = base_loss + lambda_attr * attr_loss
            
            return combined_loss
        else:
            return base_loss 