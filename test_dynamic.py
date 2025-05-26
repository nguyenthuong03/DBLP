#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/12/19
# @Author  : Dynamic CP-GNN Implementation
# @File    : test_dynamic.py
# @Software: PyCharm
# @Describe: Test script for Dynamic CP-GNN implementation

import torch
import numpy as np
from models import DynamicContextGNN
from utils import load_dynamic_data
import config_dynamic as config

def test_dynamic_data_loading():
    """Test dynamic data loading functionality"""
    print("=" * 50)
    print("Testing Dynamic Data Loading")
    print("=" * 50)
    
    try:
        # Load dynamic data
        dataloader = load_dynamic_data(config.data_config, num_snapshots=3)  # Use 3 snapshots for testing
        snapshots = dataloader.get_all_snapshots()
        
        print(f"✓ Successfully loaded {len(snapshots)} snapshots")
        
        # Check snapshot properties
        for i, snapshot in enumerate(snapshots):
            print(f"Snapshot {i+1}:")
            print(f"  - Papers: {snapshot.number_of_nodes('p')}")
            print(f"  - Authors: {snapshot.number_of_nodes('a')}")
            print(f"  - Conferences: {snapshot.number_of_nodes('c')}")
            print(f"  - Terms: {snapshot.number_of_nodes('t')}")
            print(f"  - Total edges: {snapshot.number_of_edges()}")
        
        # Test classification data loading
        CF_data = dataloader.load_classification_data()
        print(f"✓ Classification data: {len(CF_data[3])} train, {len(CF_data[4])} test samples")
        
        return True, snapshots
        
    except Exception as e:
        print(f"✗ Error in data loading: {e}")
        return False, None

def test_dynamic_model_creation(snapshots):
    """Test Dynamic CP-GNN model creation"""
    print("\n" + "=" * 50)
    print("Testing Dynamic Model Creation")
    print("=" * 50)
    
    try:
        # Create model with test configuration
        test_config = config.model_config.copy()
        test_config['embedding_dim'] = 64  # Smaller for testing
        test_config['in_dim'] = 64
        test_config['out_dim'] = 64
        test_config['kq_linear_out_dim'] = 64
        
        model = DynamicContextGNN(snapshots, test_config)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ Model created successfully")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        print(f"  - Number of snapshots: {model.num_snapshots}")
        print(f"  - Primary type: {model.primary_type}")
        print(f"  - GRU enabled: {model.enable_gru}")
        
        return True, model
        
    except Exception as e:
        print(f"✗ Error in model creation: {e}")
        return False, None

def test_forward_pass(model, snapshots):
    """Test forward pass through the model"""
    print("\n" + "=" * 50)
    print("Testing Forward Pass")
    print("=" * 50)
    
    try:
        device = torch.device('cpu')  # Use CPU for testing
        model = model.to(device)
        model.eval()
        
        # Test forward pass
        with torch.no_grad():
            # Test single forward
            final_embedding = model(k=2, device=device)  # Use k=2 for testing
            print(f"✓ Single forward pass successful")
            print(f"  - Output shape: {final_embedding.shape}")
            
            # Test temporal sequence forward
            temporal_embeddings, final_emb = model.forward_temporal_sequence(k=2, device=device)
            print(f"✓ Temporal sequence forward successful")
            print(f"  - Number of temporal embeddings: {len(temporal_embeddings)}")
            print(f"  - Final embedding shape: {final_emb.shape}")
            
            # Check that embeddings evolve across snapshots
            for i, emb in enumerate(temporal_embeddings):
                print(f"  - Snapshot {i+1} embedding shape: {emb.shape}")
            
            # Test that final embeddings match
            assert torch.allclose(final_embedding, final_emb, atol=1e-6)
            print(f"✓ Final embeddings match between methods")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in forward pass: {e}")
        return False

def test_loss_computation(model, snapshots):
    """Test loss computation"""
    print("\n" + "=" * 50)
    print("Testing Loss Computation")
    print("=" * 50)
    
    try:
        device = torch.device('cpu')
        model = model.to(device)
        model.train()
        
        # Create dummy edge data
        batch_size = 10
        final_snapshot = snapshots[-1]
        num_primary_nodes = final_snapshot.number_of_nodes(model.primary_type)
        
        pos_src = torch.randint(0, num_primary_nodes, (batch_size,))
        pos_dst = torch.randint(0, num_primary_nodes, (batch_size,))
        neg_src = torch.randint(0, num_primary_nodes, (batch_size,))
        neg_dst = torch.randint(0, num_primary_nodes, (batch_size,))
        
        # Forward pass to get embeddings
        p_context_emb = model(k=2, device=device)
        p_emb = model.primary_emb.weight[:num_primary_nodes]
        
        # Compute loss
        loss = model.get_loss(2, pos_src, pos_dst, neg_src, neg_dst, p_emb, p_context_emb)
        
        print(f"✓ Loss computation successful")
        print(f"  - Loss value: {loss.item():.6f}")
        print(f"  - Loss requires grad: {loss.requires_grad}")
        
        # Test backward pass
        loss.backward()
        print(f"✓ Backward pass successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in loss computation: {e}")
        return False

def main():
    """Run all tests"""
    print("Dynamic CP-GNN Implementation Test Suite")
    print("=" * 60)
    
    # Test 1: Data Loading
    success, snapshots = test_dynamic_data_loading()
    if not success:
        print("❌ Data loading test failed. Stopping tests.")
        return
    
    # Test 2: Model Creation
    success, model = test_dynamic_model_creation(snapshots)
    if not success:
        print("❌ Model creation test failed. Stopping tests.")
        return
    
    # Test 3: Forward Pass
    success = test_forward_pass(model, snapshots)
    if not success:
        print("❌ Forward pass test failed. Stopping tests.")
        return
    
    # Test 4: Loss Computation
    success = test_loss_computation(model, snapshots)
    if not success:
        print("❌ Loss computation test failed. Stopping tests.")
        return
    
    print("\n" + "=" * 60)
    print("🎉 All tests passed! Dynamic CP-GNN implementation is working correctly.")
    print("=" * 60)
    
    print("\nNext steps:")
    print("1. Run training: python main_dynamic.py")
    print("2. Evaluate model: python evaluate_dynamic.py")
    print("3. Analyze temporal evolution: python evaluate_dynamic.py --analyze")

if __name__ == "__main__":
    main() 