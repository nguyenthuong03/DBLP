#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/12/19
# @Author  : AE-CP-GNN Extension
# @File    : AE_CGNN.py
# @Describe: Attribute Enhanced Context-aware Graph Neural Network

import torch
import torch.nn as nn
import torch.nn.functional as F
from .CGNN import ContextGNN
from .attribute_encoder import AttributeEncoder, NodeAttributeManager
import os


class AE_ContextGNN(nn.Module):
    """
    Attribute Enhanced Context-aware Graph Neural Network.
    Extends the original ContextGNN by incorporating node text attributes 
    encoded with DistilBERT.
    """
    
    def __init__(self, g, model_config, data_config):
        super(AE_ContextGNN, self).__init__()
        
        self.g = g
        self.primary_type = model_config['primary_type']
        self.embedding_dim = model_config['embedding_dim']
        self.alpha = model_config.get('alpha', 0.5)  # Weighting parameter for combining embeddings
        
        # Original ContextGNN for structural embeddings
        self.context_gnn = ContextGNN(g, model_config)
        
        # Attribute encoder for text attributes
        self.attribute_encoder = AttributeEncoder(
            output_dim=self.embedding_dim,
            max_length=model_config.get('max_text_length', 128),
            freeze_bert=model_config.get('freeze_bert', True)
        )
        
        # Node attribute manager
        self.attribute_manager = NodeAttributeManager(
            data_config['data_path'], 
            data_config['dataset']
        )
        
        # Cache for attribute embeddings to avoid recomputation
        self.attribute_cache = {}
        self.cache_valid = False
        
        # Linear projection layer for dimension alignment (if needed)
        self.attr_projection = nn.Linear(self.embedding_dim, self.embedding_dim)
        
        print(f"AE-CP-GNN initialized with alpha={self.alpha}")
        print(f"Available node attributes: {list(self.attribute_manager.node_attributes.keys())}")
    
    def _get_node_ids_mapping(self):
        """
        Get mapping from DGL node indices to original node IDs.
        This reads the actual ID mappings created during DBLP data loading.
        """
        mappings = {}
        
        for ntype in self.g.ntypes:
            num_nodes = self.g.number_of_nodes(ntype)
            
            # Map node type abbreviation to full name
            node_type_map = {'a': 'author', 'p': 'paper', 'c': 'conf', 't': 'term'}
            node_type_name = node_type_map.get(ntype, ntype)
            
            # Load the mapping created during data loading
            data_path = os.path.join(self.attribute_manager.data_path, self.attribute_manager.dataset_name)
            node_file = os.path.join(data_path, f"{node_type_name}.txt")
            
            if os.path.exists(node_file):
                original_ids = []
                try:
                    # Try different encodings
                    encodings = ['utf-8', 'gbk', 'latin-1']
                    for encoding in encodings:
                        try:
                            with open(node_file, 'r', encoding=encoding) as f:
                                for i, line in enumerate(f):
                                    if i >= num_nodes:  # Only read up to the number of nodes in graph
                                        break
                                    line = line.strip()
                                    if line:
                                        parts = line.split('\t', 1)
                                        if len(parts) >= 1:
                                            original_id = int(parts[0])
                                            original_ids.append(original_id)
                            break
                        except UnicodeDecodeError:
                            continue
                    
                    # If we have fewer IDs than nodes, fill with consecutive IDs
                    while len(original_ids) < num_nodes:
                        if original_ids:
                            original_ids.append(original_ids[-1] + 1)
                        else:
                            original_ids.append(0)
                    
                    mappings[ntype] = original_ids[:num_nodes]
                    
                except Exception as e:
                    print(f"Warning: Could not load node mapping for {ntype}: {e}")
                    # Fallback to consecutive indexing
                    mappings[ntype] = list(range(num_nodes))
            else:
                # Fallback to consecutive indexing
                mappings[ntype] = list(range(num_nodes))
        
        return mappings
    
    def _compute_attribute_embeddings(self):
        """
        Compute attribute embeddings for all nodes and cache them.
        """
        if self.cache_valid and self.attribute_cache:
            return self.attribute_cache
        
        self.attribute_cache = {}
        node_mappings = self._get_node_ids_mapping()
        
        for ntype in self.g.ntypes:
            num_nodes = self.g.number_of_nodes(ntype)
            
            if self.attribute_manager.has_attributes(ntype):
                # Get original node IDs
                original_ids = node_mappings[ntype]
                
                # Get text attributes
                texts = self.attribute_manager.get_node_attributes(ntype, original_ids)
                
                # Process texts in batches to avoid memory issues
                batch_size = 32
                attr_embeddings = []
                
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    batch_embeddings = self.attribute_encoder.encode_texts(batch_texts)
                    attr_embeddings.append(batch_embeddings)
                
                if attr_embeddings:
                    attr_embeddings = torch.cat(attr_embeddings, dim=0)
                else:
                    attr_embeddings = torch.zeros(num_nodes, self.embedding_dim)
                
                # Apply projection if needed
                attr_embeddings = self.attr_projection(attr_embeddings)
                
            else:
                # No text attributes available, use zero embeddings
                attr_embeddings = torch.zeros(num_nodes, self.embedding_dim)
            
            # Move to same device as model
            device = next(self.parameters()).device
            self.attribute_cache[ntype] = attr_embeddings.to(device)
        
        self.cache_valid = True
        return self.attribute_cache
    
    def forward(self, k):
        """
        Forward pass that combines structural and attribute embeddings.
        
        Args:
            k: k-hop for context
            
        Returns:
            Combined embeddings for primary type nodes
        """
        # Get structural embeddings from original ContextGNN
        structural_emb = self.context_gnn(k)
        
        # Get attribute embeddings
        attr_embeddings = self._compute_attribute_embeddings()
        primary_attr_emb = attr_embeddings[self.primary_type]
        
        # Ensure same device and shape
        device = structural_emb.device
        primary_attr_emb = primary_attr_emb.to(device)
        
        # Handle size mismatch if any
        if structural_emb.size(0) != primary_attr_emb.size(0):
            min_size = min(structural_emb.size(0), primary_attr_emb.size(0))
            structural_emb = structural_emb[:min_size]
            primary_attr_emb = primary_attr_emb[:min_size]
        
        # Combine embeddings using weighted sum: z_i^ae = α·z_i + (1-α)·h_i^attr
        combined_emb = self.alpha * structural_emb + (1 - self.alpha) * primary_attr_emb
        
        return combined_emb
    
    def get_loss(self, k_hop, pos_src, pos_dst, neg_src, neg_dst, p_emb, p_context_emb):
        """
        Compute loss using combined embeddings.
        This delegates to the original ContextGNN loss function.
        """
        return self.context_gnn.get_loss(k_hop, pos_src, pos_dst, neg_src, neg_dst, p_emb, p_context_emb)
    
    def _context_score(self, src, dst, p_emb, context_emb):
        """Context score computation (delegates to original)."""
        return self.context_gnn._context_score(src, dst, p_emb, context_emb)
    
    def reset_attribute_cache(self):
        """Reset the attribute cache (call when node attributes change)."""
        self.cache_valid = False
        self.attribute_cache = {}
    
    @property
    def primary_emb(self):
        """Access to primary embeddings for compatibility."""
        return self.context_gnn.primary_emb
    
    @property
    def loss_weight(self):
        """Access to loss weights for compatibility."""
        return self.context_gnn.loss_weight
    
    def dump_cgnn_attention_matrix(self, path):
        """Dump attention matrices (delegates to original)."""
        return self.context_gnn.dump_cgnn_attention_matrix(path)


def create_ae_model_config(base_config, alpha=0.5, max_text_length=128, freeze_bert=True):
    """
    Create model configuration for AE-CP-GNN.
    
    Args:
        base_config: Base model configuration
        alpha: Weighting parameter for combining structural and attribute embeddings
        max_text_length: Maximum text length for DistilBERT
        freeze_bert: Whether to freeze DistilBERT parameters
        
    Returns:
        Updated model configuration
    """
    config = base_config.copy()
    config['alpha'] = alpha
    config['max_text_length'] = max_text_length
    config['freeze_bert'] = freeze_bert
    
    # Ensure embedding dimension is consistent
    if config['embedding_dim'] != 128:
        print(f"Warning: Changing embedding_dim from {config['embedding_dim']} to 128 for AE-CP-GNN")
        config['embedding_dim'] = 128
        config['in_dim'] = 128
        config['out_dim'] = 128
        config['kq_linear_out_dim'] = 128
    
    return config 