#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/12/19
# @Author  : AE-CP-GNN Extension
# @File    : test_AE.py
# @Software: PyCharm
# @Describe: Test script for AE-CP-GNN implementation

import torch
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_attribute_encoder():
    """Test the AttributeEncoder functionality."""
    print("=== Testing AttributeEncoder ===")
    
    try:
        from models.attribute_encoder import AttributeEncoder, NodeAttributeManager
        
        # Test AttributeEncoder
        encoder = AttributeEncoder(output_dim=128, max_length=64)
        
        # Test with sample texts
        sample_texts = [
            "Graph Neural Networks for Node Classification",
            "Attention Mechanisms in Deep Learning",
            "Transformer-based Graph Representation Learning",
            None,  # Test None handling
            "",    # Test empty string
        ]
        
        print(f"Input texts: {sample_texts}")
        
        # Encode texts
        embeddings = encoder.encode_texts(sample_texts)
        print(f"Output embeddings shape: {embeddings.shape}")
        print(f"Output embeddings dtype: {embeddings.dtype}")
        print(f"Embeddings range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
        
        # Test NodeAttributeManager
        print("\n=== Testing NodeAttributeManager ===")
        try:
            manager = NodeAttributeManager('./data', 'DBLP')
            print(f"Available node types: {list(manager.node_attributes.keys())}")
            
            for ntype in manager.node_attributes.keys():
                count = len(manager.node_attributes[ntype])
                print(f"Node type '{ntype}': {count} nodes with attributes")
                
                # Show sample attributes
                if count > 0:
                    sample_ids = list(manager.node_attributes[ntype].keys())[:3]
                    for node_id in sample_ids:
                        text = manager.node_attributes[ntype][node_id]
                        print(f"  {node_id}: {text[:50]}..." if len(text) > 50 else f"  {node_id}: {text}")
        
        except Exception as e:
            print(f"Error testing NodeAttributeManager: {e}")
            return False
        
        print("AttributeEncoder tests completed.\n")
        return True
        
    except Exception as e:
        print(f"Error testing AttributeEncoder: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ae_model():
    """Test the AE-CP-GNN model initialization and forward pass."""
    print("=== Testing AE-ContextGNN Model ===")
    
    try:
        from models import AE_ContextGNN, create_ae_model_config
        from utils import load_data
        import config_AE
        
        # Create test config
        test_config = create_ae_model_config(
            config_AE.model_config,
            alpha=0.5,
            max_text_length=64,
            freeze_bert=True
        )
        
        print(f"Model config: {test_config}")
        
        # Load data
        print("Loading DBLP data...")
        dataloader = load_data(config_AE.data_config)
        hg = dataloader.heter_graph
        
        print(f"Graph info:")
        print(f"  Node types: {hg.ntypes}")
        print(f"  Edge types: {hg.etypes}")
        for ntype in hg.ntypes:
            print(f"  {ntype}: {hg.number_of_nodes(ntype)} nodes")
        
        # Initialize model
        print("\nInitializing AE-CP-GNN model...")
        model = AE_ContextGNN(hg, test_config, config_AE.data_config)
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        
        # Test forward pass
        print("\nTesting forward pass...")
        with torch.no_grad():
            embeddings = model(1)  # 1-hop
            print(f"Output embeddings shape: {embeddings.shape}")
            print(f"Output embeddings dtype: {embeddings.dtype}")
            print(f"Embeddings range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
        
        print("AE-ContextGNN model tests completed.\n")
        return True
        
    except Exception as e:
        print(f"Error testing AE-ContextGNN model: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loading():
    """Test data loading and preprocessing."""
    print("=== Testing Data Loading ===")
    
    try:
        import config_AE
        from utils import load_data
        
        # Load data
        dataloader = load_data(config_AE.data_config)
        print("Data loading successful!")
        
        # Test classification data
        CF_data = dataloader.load_classification_data()
        features, labels, num_classes, train_idx, test_idx = CF_data
        
        print(f"Classification data:")
        print(f"  Features shape: {features.shape if features is not None else 'None'}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Number of classes: {num_classes}")
        print(f"  Train samples: {len(train_idx)}")
        print(f"  Test samples: {len(test_idx)}")
        print(f"  Label distribution: {[int((labels == i).sum()) for i in range(num_classes)]}")
        
        # Test edge data loading
        print("\nTesting edge data loading...")
        edges_data_dict = dataloader.load_train_k_context_edges(
            dataloader.heter_graph,
            config_AE.data_config['K_length'],
            config_AE.data_config['primary_type'],
            [5, 5, 5],  # Smaller sample for testing
            [2, 2, 2]
        )
        
        print(f"Edge datasets: {list(edges_data_dict.keys())}")
        for k, dataset in edges_data_dict.items():
            print(f"  K={k}: {len(dataset)} samples")
        
        print("Data loading tests completed.\n")
        return True
        
    except Exception as e:
        print(f"Error testing data loading: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=== AE-CP-GNN Implementation Tests ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Check if transformers is available
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
    except ImportError:
        print("Warning: transformers library not installed!")
        print("Please install: pip install transformers")
        return
    
    # Run tests
    tests = [
        ("Data Loading", test_data_loading),
        ("Attribute Encoder", test_attribute_encoder),
        ("AE-ContextGNN Model", test_ae_model),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running {test_name} test...")
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            print(f"Error in {test_name} test: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*50}")
    print("Test Summary:")
    for test_name, success in results.items():
        status = "PASSED" if success else "FAILED"
        print(f"  {test_name}: {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n🎉 AE-CP-GNN implementation is ready for training!")
        print("You can now run: python main_AE.py")
    else:
        print("\n❌ Please fix the failing tests before training.")


if __name__ == "__main__":
    main() 