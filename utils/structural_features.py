#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/12/19
# @Author  : Adaptive CP-GNN Implementation
# @File    : structural_features.py
# @Describe: Extract structural features for adaptive context path weighting

import torch
import dgl
import numpy as np
from sklearn.preprocessing import StandardScaler


class StructuralFeatureExtractor:
    """
    Extract structural features for nodes in heterogeneous graphs.
    Features include:
    1. Node degree (number of direct connections)
    2. Context neighbors count for each k-hop
    3. Average shortest path length (approximated)
    """
    
    def __init__(self, hg, primary_type, K_length, device='cpu'):
        """
        Initialize the structural feature extractor.
        
        Args:
            hg: DGL heterogeneous graph
            primary_type: Primary node type (e.g., 'a' for authors in DBLP)
            K_length: Maximum context path length
            device: Device to run computations on
        """
        self.hg = hg
        self.primary_type = primary_type
        self.K_length = K_length
        self.device = device
        self.scaler = StandardScaler()
        self._features_cache = None
        
    def extract_features(self, normalize=True):
        """
        Extract structural features for all primary nodes.
        
        Args:
            normalize: Whether to normalize features using StandardScaler
            
        Returns:
            torch.Tensor: Feature matrix of shape (num_primary_nodes, num_features)
        """
        if self._features_cache is not None:
            return self._features_cache
            
        num_primary_nodes = self.hg.number_of_nodes(self.primary_type)
        
        # Feature 1: Node degree
        degrees = self._compute_node_degrees()
        
        # Feature 2: Context neighbors count for each k-hop
        context_neighbors = self._compute_context_neighbors()
        
        # Feature 3: Average clustering coefficient (proxy for local connectivity)
        clustering_coeffs = self._compute_clustering_coefficient()
        
        # Combine all features
        features = np.column_stack([
            degrees,
            context_neighbors,
            clustering_coeffs
        ])
        
        # Normalize features
        if normalize:
            features = self.scaler.fit_transform(features)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features).to(self.device)
        
        # Cache the features
        self._features_cache = features_tensor
        
        return features_tensor
    
    def _compute_node_degrees(self):
        """Compute the degree of each primary node."""
        # Convert to homogeneous graph to compute degrees easily
        homo_g = dgl.to_homogeneous(self.hg)
        
        # Find primary node indices in homogeneous graph
        ntype_id = self.hg.ntypes.index(self.primary_type)
        primary_mask = (homo_g.ndata['_TYPE'] == ntype_id)
        primary_indices = torch.where(primary_mask)[0]
        
        # Compute degrees
        degrees = homo_g.in_degrees(primary_indices) + homo_g.out_degrees(primary_indices)
        
        return degrees.cpu().numpy()
    
    def _compute_context_neighbors(self):
        """
        Compute the number of context neighbors reachable within K hops.
        Returns a matrix where each row is a node and each column is k-hop neighbor count.
        """
        homo_g = dgl.to_homogeneous(self.hg)
        ntype_id = self.hg.ntypes.index(self.primary_type)
        primary_mask = (homo_g.ndata['_TYPE'] == ntype_id)
        primary_indices = torch.where(primary_mask)[0]
        
        num_primary_nodes = len(primary_indices)
        context_neighbors = np.zeros((num_primary_nodes, self.K_length))
        
        # Convert to adjacency matrix for efficient k-hop computation
        adj_matrix = homo_g.adjacency_matrix().to_dense().float()
        
        # Compute k-hop reachability
        current_adj = adj_matrix
        for k in range(self.K_length):
            if k > 0:
                current_adj = torch.matmul(current_adj, adj_matrix)
            
            # Count reachable nodes for each primary node
            for i, node_idx in enumerate(primary_indices):
                # Count non-zero entries (reachable nodes) excluding self
                reachable = (current_adj[node_idx] > 0).sum().item()
                if k == 0:  # For 1-hop, exclude self-connection
                    reachable = max(0, reachable - 1)
                context_neighbors[i, k] = reachable
        
        return context_neighbors
    
    def _compute_clustering_coefficient(self):
        """
        Compute a proxy for clustering coefficient using local triangle count.
        For heterogeneous graphs, we approximate this using the homogeneous projection.
        """
        homo_g = dgl.to_homogeneous(self.hg)
        ntype_id = self.hg.ntypes.index(self.primary_type)
        primary_mask = (homo_g.ndata['_TYPE'] == ntype_id)
        primary_indices = torch.where(primary_mask)[0]
        
        # Convert to undirected for clustering coefficient
        homo_g = dgl.to_bidirected(homo_g)
        
        clustering_coeffs = []
        
        for node_idx in primary_indices:
            # Get neighbors
            neighbors = homo_g.successors(node_idx)
            degree = len(neighbors)
            
            if degree < 2:
                clustering_coeffs.append(0.0)
                continue
            
            # Count triangles (edges between neighbors)
            triangle_count = 0
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if homo_g.has_edges_between(neighbors[i], neighbors[j]):
                        triangle_count += 1
            
            # Clustering coefficient = 2 * triangles / (degree * (degree - 1))
            max_triangles = degree * (degree - 1) // 2
            clustering_coeff = triangle_count / max_triangles if max_triangles > 0 else 0.0
            clustering_coeffs.append(clustering_coeff)
        
        return np.array(clustering_coeffs)
    
    def get_feature_dim(self):
        """Return the dimension of extracted features."""
        return 1 + self.K_length + 1  # degree + context_neighbors + clustering_coeff 