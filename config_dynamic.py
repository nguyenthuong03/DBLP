#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/12/19
# @Author  : Dynamic CP-GNN Implementation
# @File    : config_dynamic.py
# @Software: PyCharm
# @Describe: Configuration for Dynamic CP-GNN (D-CP-GNN)

import os

config_path = os.path.dirname(__file__)

# Data configuration for dynamic graphs
data_config = {
    'data_path': os.path.join(config_path, 'data'),
    'dataset': 'DBLP',
    'data_name': 'DBLP.mat',
    'primary_type': 'a',  # Author nodes as primary type
    'task': ['CF', 'CL'],  # Classification and Clustering tasks
    'K_length': 4,  # 3-hop context as specified
    'ignore_edge_type': True,
    'resample': False,
    'random_seed': 123,
    'test_ratio': 0.8,
    'num_snapshots': 5  # 5 temporal snapshots (T=5)
}

# Model configuration for D-CP-GNN
model_config = {
    'primary_type': data_config['primary_type'],
    'auxiliary_embedding': 'non_linear',  # Non-linear auxiliary embedding
    'K_length': data_config['K_length'],
    'embedding_dim': 128,  # Reduced to 128 as specified in the requirements
    'in_dim': 128,
    'out_dim': 128,
    'num_heads': 8,  # 8 attention heads as specified
    'merge': 'linear',  # Multi-head attention merge method
    'g_agg_type': 'mean',  # Graph representation encoder
    'drop_out': 0.3,  # Node dropout rate as specified
    'cgnn_non_linear': True,  # Enable non-linear activation for CGNN
    'multi_attn_linear': True,  # Enable attention K/Q-linear for each type
    'graph_attention': True,
    'kq_linear_out_dim': 128,
    'path_attention': True,  # Enable context path attention
    'c_linear_out_dim': 8,
    'enable_bilinear': True,  # Enable bilinear for context attention
    'gru': True,  # Enable GRU for temporal modeling
    'add_init': False
}

# Training configuration for D-CP-GNN
train_config = {
    'continue': False,
    'lr': 0.05,  # Learning rate as specified
    'l2': 0,
    'factor': 0.2,
    'total_epoch': 10000000,
    'batch_size': 1024,
    'pos_num_for_each_hop': [20, 20, 20, 20, 20, 20, 20, 20, 20],
    'neg_num_for_each_hop': [3, 3, 3, 3, 3, 3, 3, 3, 3],
    'sample_workers': 8,
    'patience': 15,
    'checkpoint_path': os.path.join(config_path, 'checkpoint', f"dynamic_{data_config['dataset']}")
}

# Evaluation configuration
evaluate_config = {
    'method': 'LR',  # Logistic Regression for evaluation
    'save_heat_map': True,
    'result_path': os.path.join('result', f"dynamic_{data_config['dataset']}"),
    'random_state': 123,
    'max_iter': 500,
    'n_jobs': 1,
    'save_heat_map': False
}

# Dynamic-specific configuration
dynamic_config = {
    'num_snapshots': data_config['num_snapshots'],
    'temporal_modeling': True,
    'gru_enabled': model_config['gru'],
    'snapshot_years': list(range(2010, 2010 + data_config['num_snapshots'])),  # 2010-2014
    'cumulative_snapshots': True,  # Each snapshot includes previous data
    'final_snapshot_evaluation': True  # Evaluate on final snapshot
} 
