#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Describe: Transformer Enhanced Context Path Graph Neural Network

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv

class TransformerEnhancedCPGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, num_heads, dropout=0.1):
        super(TransformerEnhancedCPGNN, self).__init__()
        # Transformer Encoder để mã hóa thuộc tính
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=in_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout
            ),
            num_layers=num_layers
        )
        # Giảm chiều từ in_dim (768 + num_categories) về hidden_dim
        self.fc = nn.Linear(in_dim, hidden_dim)
        
        # Các tầng GNN từ CP-GNN gốc
        self.gnn_layers = nn.ModuleList([
            GraphConv(hidden_dim, hidden_dim, activation=F.relu)
            for _ in range(2)  # Giả định 2 tầng GNN như CP-GNN gốc
        ])
        
        # Cơ chế chú ý cho context path (giữ nguyên từ CP-GNN gốc)
        self.path_attention = nn.Linear(hidden_dim, 1)
        self.edge_attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, g, node_features):
        # Mã hóa thuộc tính bằng Transformer
        h = self.transformer(node_features.unsqueeze(0)).squeeze(0)  # [num_nodes, in_dim]
        h = F.relu(self.fc(h))  # [num_nodes, hidden_dim]
        
        # Tầng GNN (giữ nguyên từ CP-GNN gốc)
        for layer in self.gnn_layers:
            h = layer(g, h)
        
        # Tính chú ý cho context path (giả định từ CP-GNN gốc)
        path_scores = torch.sigmoid(self.path_attention(h))
        edge_scores = torch.sigmoid(self.edge_attention(h))
        
        # Tổng hợp embedding (giả định từ CP-GNN gốc)
        h = h * path_scores  # Ví dụ: nhân với trọng số chú ý
        return h

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias) 