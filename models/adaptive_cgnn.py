#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/12/19
# @Author  : Adaptive CP-GNN Implementation
# @File    : adaptive_cgnn.py
# @Describe: Adaptive Context-aware Graph Neural Network (A-CP-GNN)

import torch
import torch.nn as nn
import torch.nn.functional as F
from .CGNN import ContextGNN, MultiHeadCGNN, EmbTransformer
from .adaptive_mlp import AdaptiveWeightMLP, AdaptiveContextAggregator
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.structural_features import StructuralFeatureExtractor


class AdaptiveContextGNN(nn.Module):
    """
    Adaptive Context-aware Graph Neural Network (A-CP-GNN).
    
    Extends the original CP-GNN by using an MLP to predict adaptive weights
    for different context path lengths based on structural features of nodes.
    """
    
    def __init__(self, g, model_config):
        super(AdaptiveContextGNN, self).__init__()
        
        self.g = g
        self.primary_type = model_config['primary_type']
        self.auxiliary_embedding = model_config['auxiliary_embedding']
        self.K_length = model_config['K_length']
        self.embedding_dim = model_config['embedding_dim']
        self.enable_gru = model_config['gru']
        self.enable_add_init = model_config['add_init']
        self.ntypes = g.ntypes
        self.etypes = g.etypes
        
        # Primary embeddings
        self.primary_emb = nn.Embedding(g.number_of_nodes(self.primary_type), 
                                       model_config['embedding_dim'])
        
        # Auxiliary embeddings or transformers
        if self.auxiliary_embedding == "emb":
            self.auxiliary_emb = nn.ModuleDict(
                {ntype: nn.Embedding(g.number_of_nodes(ntype), model_config['embedding_dim']) 
                 for ntype in g.ntypes if ntype != self.primary_type})
        else:
            self.auxiliary_tans_fuc = nn.ModuleDict({
                etype: EmbTransformer(model_config['in_dim'], model_config['out_dim'], 
                                    self.auxiliary_embedding) 
                for etype in g.etypes
            })
        
        # Multi-head CGNN layers for each hop
        self.multihead_cgnn = nn.ModuleList([
            MultiHeadCGNN(model_config['in_dim'], model_config['out_dim'],
                         model_config['num_heads'],
                         merge=model_config['merge'], 
                         g_agg_type=model_config['g_agg_type'],
                         drop_out=model_config['drop_out'],
                         cgnn_non_linear=model_config['cgnn_non_linear'],
                         multi_attn_linear=model_config['multi_attn_linear'],
                         ntypes=g.ntypes,
                         etypes=g.etypes,
                         graph_attention=model_config['graph_attention'],
                         kq_linear_out_dim=model_config['kq_linear_out_dim'],
                         path_attention=model_config['path_attention'],
                         c_linear_out_dim=model_config['c_linear_out_dim'],
                         enable_bilinear=model_config['enable_bilinear']) 
            for hop in range(self.K_length + 1)
        ])
        
        # GRU gate for temporal modeling
        if self.enable_gru:
            self.gru_gate = nn.GRUCell(model_config['out_dim'], model_config['out_dim'])
        
        # Structural feature extractor
        device = next(self.parameters()).device if len(list(self.parameters())) > 0 else 'cpu'
        self.feature_extractor = StructuralFeatureExtractor(
            g, self.primary_type, self.K_length, device=device
        )
        
        # Adaptive weight prediction MLP
        feature_dim = self.feature_extractor.get_feature_dim()
        self.adaptive_mlp = AdaptiveWeightMLP(
            input_dim=feature_dim,
            K_length=self.K_length,
            hidden_dims=model_config.get('adaptive_hidden_dims', [64, 32]),
            dropout=model_config.get('adaptive_dropout', 0.3)
        )
        
        # Context aggregator
        self.context_aggregator = AdaptiveContextAggregator(model_config['out_dim'])
        
        # Loss weights (kept for compatibility, but will be replaced by adaptive weights)
        self.loss_weight = nn.Parameter(torch.zeros(self.K_length + 1))
        
        # Cache for structural features
        self._structural_features = None
        self._adaptive_weights = None
        
        self.reset_parameter()
    
    def reset_parameter(self):
        """Initialize parameters."""
        if self.auxiliary_embedding == "emb":
            [nn.init.normal_(emb.weight) for emb in self.auxiliary_emb.values()]
        else:
            [nn.init.normal_(func.linear.weight) for func in self.auxiliary_tans_fuc.values()]
        
        # Initialize primary embeddings
        nn.init.normal_(self.primary_emb.weight)
    
    def embedding_transformer(self, primary_feature):
        """
        Get HIN feature dict for each node type.
        
        Args:
            primary_feature: Primary graph feature
            
        Returns:
            h_dict: Dictionary {ntype: feature}
        """
        if self.auxiliary_embedding == "emb":
            h_dict = {ntype: emb.weight for ntype, emb in self.auxiliary_emb.items()}
            h_dict[self.primary_type] = primary_feature
        else:
            h_dict = {}
            non_init_graph = self.ntypes[:]  # Copy list
            h_dict[self.primary_type] = primary_feature
            non_init_graph.remove(self.primary_type)
            while non_init_graph:
                for srctype, etype, dsttype in self.g.canonical_etypes:
                    if srctype not in non_init_graph and dsttype in non_init_graph:
                        h_dict[dsttype] = self.auxiliary_tans_fuc[etype](
                            self.g, h_dict[srctype], srctype, etype, dsttype)
                        non_init_graph.remove(dsttype)
        return h_dict
    
    def get_adaptive_weights(self):
        """
        Compute adaptive weights for context paths based on structural features.
        
        Returns:
            adaptive_weights: Tensor of shape (num_primary_nodes, K_length)
        """
        if self._adaptive_weights is not None:
            return self._adaptive_weights
        
        # Extract structural features if not cached
        if self._structural_features is None:
            device = next(self.parameters()).device
            self.feature_extractor.device = device
            self._structural_features = self.feature_extractor.extract_features()
        
        # Predict adaptive weights using MLP
        self._adaptive_weights = self.adaptive_mlp(self._structural_features)
        
        return self._adaptive_weights
    
    def forward_all_hops(self):
        """
        Forward pass through all K hops to get context embeddings.
        
        Returns:
            context_embeddings: List of embeddings for each hop length
        """
        h_dict = self.embedding_transformer(self.primary_emb.weight)
        context_embeddings = []
        
        # Store initial embedding (0-hop)
        context_embeddings.append(h_dict[self.primary_type])
        
        # Forward through each hop
        for hop in range(self.K_length):
            new_h_dict = self.multihead_cgnn[hop](self.g, h_dict)
            
            if self.enable_gru:
                new_primary_feature = self.gru_gate(
                    new_h_dict[self.primary_type], h_dict[self.primary_type])
                new_h_dict[self.primary_type] = new_primary_feature
            
            if self.enable_add_init:
                new_h_dict[self.primary_type] += self.primary_emb.weight
            
            h_dict = new_h_dict
            context_embeddings.append(h_dict[self.primary_type])
        
        return context_embeddings
    
    def forward(self, k=None):
        """
        Forward pass with adaptive context aggregation.
        
        Args:
            k: If specified, return k-hop embedding (for compatibility).
               If None, return adaptively aggregated embedding.
               
        Returns:
            Embedding of primary nodes
        """
        if k is not None:
            # Compatibility mode: return specific k-hop embedding
            return self._forward_single_hop(k)
        
        # Adaptive mode: aggregate all hops with adaptive weights
        context_embeddings = self.forward_all_hops()
        adaptive_weights = self.get_adaptive_weights()
        
        # Use only the first K_length embeddings (excluding 0-hop for aggregation)
        context_embeddings_for_agg = context_embeddings[1:self.K_length+1]
        
        # Aggregate using adaptive weights
        aggregated_embedding = self.context_aggregator(context_embeddings_for_agg, adaptive_weights)
        
        return aggregated_embedding
    
    def _forward_single_hop(self, k):
        """
        Forward pass for a specific k-hop (compatibility with original CP-GNN).
        
        Args:
            k: k-hop length
            
        Returns:
            k-hop embedding of primary nodes
        """
        h_dict = self.embedding_transformer(self.primary_emb.weight)
        
        for hop in range(k):
            new_h_dict = self.multihead_cgnn[hop](self.g, h_dict)
            
            if self.enable_gru:
                new_primary_feature = self.gru_gate(
                    new_h_dict[self.primary_type], h_dict[self.primary_type])
                new_h_dict[self.primary_type] = new_primary_feature
            
            if self.enable_add_init:
                new_h_dict[self.primary_type] += self.primary_emb.weight
            
            h_dict = new_h_dict
        
        return h_dict[self.primary_type]
    
    def _context_score(self, src, dst, p_emb, context_emb):
        """Compute context score between source and destination nodes."""
        logits = torch.sum((p_emb[src] * context_emb[src]) * 
                          (p_emb[dst] * context_emb[dst]), dim=1)
        return logits
    
    def get_loss(self, k_hop, pos_src, pos_dst, neg_src, neg_dst, p_emb, p_context_emb):
        """
        Compute loss for a specific k-hop (compatibility with original training).
        
        Args:
            k_hop: k-hop length
            pos_src, pos_dst: Positive edge source and destination
            neg_src, neg_dst: Negative edge source and destination  
            p_emb: Primary embeddings
            p_context_emb: Context embeddings
            
        Returns:
            Loss value
        """
        k_length = k_hop - 1
        pos_score = self._context_score(pos_src, pos_dst, p_emb, p_context_emb)
        neg_score = self._context_score(neg_src, neg_dst, p_emb, p_context_emb)
        
        pos_loss = -F.logsigmoid(pos_score).view(-1).mean()
        neg_loss = -F.logsigmoid(-neg_score).view(-1).mean()
        
        # Use adaptive weights for loss weighting
        adaptive_weights = self.get_adaptive_weights()
        # Average adaptive weight for this k-hop across all nodes
        avg_weight = adaptive_weights[:, k_length].mean()
        
        return (pos_loss + neg_loss) * avg_weight + self.loss_weight[k_length]
    
    def get_adaptive_loss(self, pos_src, pos_dst, neg_src, neg_dst):
        """
        Compute adaptive loss using all context paths with adaptive weights.
        
        Args:
            pos_src, pos_dst: Positive edge source and destination
            neg_src, neg_dst: Negative edge source and destination
            
        Returns:
            Adaptive loss value
        """
        # Get adaptive aggregated embeddings
        adaptive_emb = self.forward()  # This uses adaptive weights
        p_emb = self.primary_emb.weight
        
        # Compute scores
        pos_score = self._context_score(pos_src, pos_dst, p_emb, adaptive_emb)
        neg_score = self._context_score(neg_src, neg_dst, p_emb, adaptive_emb)
        
        pos_loss = -F.logsigmoid(pos_score).view(-1).mean()
        neg_loss = -F.logsigmoid(-neg_score).view(-1).mean()
        
        return pos_loss + neg_loss
    
    def dump_cgnn_attention_matrix(self, path):
        """Dump attention matrix (compatibility with original CP-GNN)."""
        import json
        with open(path, 'w') as f:
            multi_head_attention_matrix = {}
            for k, layer in enumerate(self.multihead_cgnn):
                hop_multi_head_matrix = layer.dump_multi_head_attention_matrix()
                multi_head_attention_matrix["length_{}".format(k)] = hop_multi_head_matrix
            json.dump(multi_head_attention_matrix, f)
    
    def clear_cache(self):
        """Clear cached features and weights (useful for training)."""
        self._structural_features = None
        self._adaptive_weights = None 