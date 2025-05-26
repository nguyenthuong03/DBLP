#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/12/19
# @Author  : Dynamic CP-GNN Implementation
# @File    : DynamicCGNN.py
# @Software: PyCharm
# @Describe: Dynamic Context-aware Graph Neural Network with GRU temporal modeling

import torch
import torch.nn as nn
import torch.nn.functional as F
from .CGNN import ContextGNN, MultiHeadCGNN, EmbTransformer


class DynamicContextGNN(nn.Module):
    """
    Dynamic Context-aware Graph Neural Network (D-CP-GNN)
    Extends CP-GNN with GRU-based temporal modeling for dynamic heterogeneous graphs
    """
    
    def __init__(self, snapshots, model_config):
        super(DynamicContextGNN, self).__init__()
        
        # Store snapshots and configuration
        self.snapshots = snapshots
        self.num_snapshots = len(snapshots)
        self.model_config = model_config
        
        # Get graph structure from the first snapshot
        self.g_template = snapshots[0]
        self.primary_type = model_config['primary_type']
        self.auxiliary_embedding = model_config['auxiliary_embedding']
        self.K_length = model_config['K_length']
        self.embedding_dim = model_config['embedding_dim']
        
        # Node type and edge type information
        self.ntypes = self.g_template.ntypes
        self.etypes = self.g_template.etypes
        
        # Initialize primary embeddings for the largest snapshot (final one)
        max_primary_nodes = max([g.number_of_nodes(self.primary_type) for g in snapshots])
        self.primary_emb = nn.Embedding(max_primary_nodes, model_config['embedding_dim'])
        
        # GRU configuration
        self.enable_gru = model_config['gru']
        self.enable_add_init = model_config['add_init']
        
        # Auxiliary embedding transformers
        if self.auxiliary_embedding == "emb":
            # For embedding-based auxiliary nodes, we need separate embeddings for each snapshot
            self.auxiliary_emb = nn.ModuleDict()
            for ntype in self.ntypes:
                if ntype != self.primary_type:
                    max_aux_nodes = max([g.number_of_nodes(ntype) for g in snapshots])
                    self.auxiliary_emb[ntype] = nn.Embedding(max_aux_nodes, model_config['embedding_dim'])
        else:
            self.auxiliary_tans_fuc = nn.ModuleDict({
                etype: EmbTransformer(model_config['in_dim'], model_config['out_dim'], self.auxiliary_embedding) 
                for etype in self.etypes
            })
        
        # Multi-head CGNN layers for each hop
        self.multihead_cgnn = nn.ModuleList([
            MultiHeadCGNN(
                model_config['in_dim'], model_config['out_dim'],
                model_config['num_heads'],
                merge=model_config['merge'], 
                g_agg_type=model_config['g_agg_type'],
                drop_out=model_config['drop_out'],
                cgnn_non_linear=model_config['cgnn_non_linear'],
                multi_attn_linear=model_config['multi_attn_linear'],
                ntypes=self.ntypes,
                etypes=self.etypes,
                graph_attention=model_config['graph_attention'],
                kq_linear_out_dim=model_config['kq_linear_out_dim'],
                path_attention=model_config['path_attention'],
                c_linear_out_dim=model_config['c_linear_out_dim'],
                enable_bilinear=model_config['enable_bilinear']
            ) for hop in range(self.K_length + 1)
        ])
        
        # Loss weights for different hops
        self.loss_weight = nn.Parameter(torch.zeros(self.K_length + 1))
        
        # GRU for temporal modeling
        if self.enable_gru:
            self.temporal_gru = nn.GRUCell(model_config['out_dim'], model_config['out_dim'])
        
        # Initialize parameters
        self.reset_parameter()
    
    def reset_parameter(self):
        """Initialize model parameters"""
        nn.init.normal_(self.primary_emb.weight)
        
        if self.auxiliary_embedding == "emb":
            for emb in self.auxiliary_emb.values():
                nn.init.normal_(emb.weight)
        else:
            for func in self.auxiliary_tans_fuc.values():
                nn.init.normal_(func.linear.weight)
    
    def embedding_transformer(self, primary_feature, snapshot):
        """
        Generate HIN feature dict for each node type in a specific snapshot
        """
        if self.auxiliary_embedding == "emb":
            h_dict = {}
            for ntype in snapshot.ntypes:
                if ntype == self.primary_type:
                    h_dict[ntype] = primary_feature
                else:
                    # Use only the nodes present in this snapshot
                    num_nodes = snapshot.number_of_nodes(ntype)
                    h_dict[ntype] = self.auxiliary_emb[ntype].weight[:num_nodes]
        else:
            h_dict = {}
            non_init_graph = snapshot.ntypes[:]
            h_dict[self.primary_type] = primary_feature
            non_init_graph.remove(self.primary_type)
            
            while non_init_graph:
                for srctype, etype, dsttype in snapshot.canonical_etypes:
                    if srctype not in non_init_graph and dsttype in non_init_graph:
                        h_dict[dsttype] = self.auxiliary_tans_fuc[etype](
                            snapshot, h_dict[srctype], srctype, etype, dsttype
                        )
                        non_init_graph.remove(dsttype)
        
        return h_dict
    
    def forward_snapshot(self, snapshot, k, primary_nodes_slice=None):
        """
        Forward pass for a single snapshot
        
        Args:
            snapshot: DGL heterogeneous graph for this time step
            k: k-hop context length
            primary_nodes_slice: slice of primary embeddings to use for this snapshot
        
        Returns:
            h_dict: Node embeddings for this snapshot
        """
        # Get primary node embeddings for this snapshot
        if primary_nodes_slice is None:
            num_primary_nodes = snapshot.number_of_nodes(self.primary_type)
            primary_feature = self.primary_emb.weight[:num_primary_nodes]
        else:
            primary_feature = primary_nodes_slice
        
        # Generate embeddings for all node types
        h_dict = self.embedding_transformer(primary_feature, snapshot)
        
        # Apply k-hop CGNN layers
        for hop in range(k):
            new_h_dict = self.multihead_cgnn[hop](snapshot, h_dict)
            
            # Apply residual connection if enabled
            if self.enable_add_init:
                new_h_dict[self.primary_type] += primary_feature
            
            h_dict = new_h_dict
        
        return h_dict
    
    def forward_temporal_sequence(self, k, device=None):
        """
        Forward pass through the entire temporal sequence
        
        Args:
            k: k-hop context length
            device: device to run computations on
        
        Returns:
            temporal_embeddings: List of dynamic embeddings for each snapshot
            final_embedding: Final dynamic embedding from the last snapshot
        """
        if device is None:
            device = next(self.parameters()).device
        
        temporal_embeddings = []
        hidden_state = None
        
        for t, snapshot in enumerate(self.snapshots):
            # Move snapshot to device
            snapshot = snapshot.to(device)
            
            # Get structural embeddings from CP-GNN
            num_primary_nodes = snapshot.number_of_nodes(self.primary_type)
            primary_feature = self.primary_emb.weight[:num_primary_nodes]
            
            # Forward through CP-GNN for this snapshot
            h_dict = self.forward_snapshot(snapshot, k, primary_feature)
            structural_embedding = h_dict[self.primary_type]  # z_i^(t)
            
            # Apply GRU for temporal modeling
            if self.enable_gru and hidden_state is not None:
                # Ensure hidden state has the same number of nodes as current snapshot
                if hidden_state.size(0) != structural_embedding.size(0):
                    # Pad or truncate hidden state to match current snapshot size
                    if hidden_state.size(0) < structural_embedding.size(0):
                        # Pad with zeros for new nodes
                        padding = torch.zeros(
                            structural_embedding.size(0) - hidden_state.size(0),
                            hidden_state.size(1),
                            device=device
                        )
                        hidden_state = torch.cat([hidden_state, padding], dim=0)
                    else:
                        # Truncate for smaller snapshots (shouldn't happen in cumulative approach)
                        hidden_state = hidden_state[:structural_embedding.size(0)]
                
                # Update with GRU: h_i^(t) = GRU(z_i^(t), h_i^(t-1))
                dynamic_embedding = self.temporal_gru(structural_embedding, hidden_state)
            else:
                # First snapshot or GRU disabled
                dynamic_embedding = structural_embedding
            
            # Update hidden state for next iteration
            hidden_state = dynamic_embedding
            temporal_embeddings.append(dynamic_embedding)
        
        return temporal_embeddings, temporal_embeddings[-1]
    
    def forward(self, k, device=None):
        """
        Main forward pass - returns final dynamic embeddings
        """
        _, final_embedding = self.forward_temporal_sequence(k, device)
        return final_embedding
    
    def _context_score(self, src, dst, p_emb, context_emb):
        """Compute context score for link prediction"""
        logits = torch.sum((p_emb[src] * context_emb[src]) * (p_emb[dst] * context_emb[dst]), dim=1)
        return logits
    
    def get_loss(self, k_hop, pos_src, pos_dst, neg_src, neg_dst, p_emb, p_context_emb):
        """
        Compute loss using dynamic embeddings
        """
        k_length = k_hop - 1
        pos_score = self._context_score(pos_src, pos_dst, p_emb, p_context_emb)
        neg_score = self._context_score(neg_src, neg_dst, p_emb, p_context_emb)
        
        pos_loss = -F.logsigmoid(pos_score).view(-1).mean()
        neg_loss = -F.logsigmoid(-neg_score).view(-1).mean()
        
        # Multi-loss with learnable weights
        weight = torch.exp(-self.loss_weight[k_length])
        return (pos_loss + neg_loss) * weight + self.loss_weight[k_length]
    
    def dump_cgnn_attention_matrix(self, path):
        """
        Dump multi-head attention matrix to json (inherited functionality)
        """
        import json
        with open(path, 'w') as f:
            multi_head_attention_matrix = {}
            for k, layer in enumerate(self.multihead_cgnn):
                hop_multi_head_matrix = layer.dump_multi_head_attention_matrix()
                multi_head_attention_matrix[f"length_{k}"] = hop_multi_head_matrix
            json.dump(multi_head_attention_matrix, f)


class DynamicEdgesDataset:
    """
    Dataset for dynamic graph edges across multiple snapshots
    """
    def __init__(self, snapshot_edges_datasets):
        """
        Args:
            snapshot_edges_datasets: List of EdgesDataset for each snapshot
        """
        self.snapshot_datasets = snapshot_edges_datasets
        self.num_snapshots = len(snapshot_edges_datasets)
    
    def __len__(self):
        # Return length of the largest snapshot dataset
        return max(len(dataset) for dataset in self.snapshot_datasets)
    
    def __getitem__(self, idx):
        # Return edges from all snapshots for this index
        snapshot_edges = []
        for dataset in self.snapshot_datasets:
            if idx < len(dataset):
                snapshot_edges.append(dataset[idx])
            else:
                # If snapshot is smaller, repeat the last item or use a default
                snapshot_edges.append(dataset[len(dataset) - 1])
        return snapshot_edges
    
    @staticmethod
    def collate(batches):
        """
        Collate function for dynamic edges
        """
        # batches is a list of snapshot_edges lists
        num_snapshots = len(batches[0])
        collated_snapshots = []
        
        for t in range(num_snapshots):
            # Collect edges for snapshot t across all batches
            snapshot_batch = [batch[t] for batch in batches]
            # Use the original collate function
            from utils.preprocess import EdgesDataset
            collated_snapshot = EdgesDataset.collate(snapshot_batch)
            collated_snapshots.append(collated_snapshot)
        
        return collated_snapshots 