#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Quick test script for Adaptive CP-GNN implementation

import sys
import os

def test_imports():
    """Test if all modules can be imported correctly."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import dgl
        print(f"✓ DGL {dgl.__version__}")
    except ImportError as e:
        print(f"✗ DGL import failed: {e}")
        return False
    
    try:
        from models import AdaptiveContextGNN
        print("✓ AdaptiveContextGNN import successful")
    except ImportError as e:
        print(f"✗ AdaptiveContextGNN import failed: {e}")
        return False
    
    try:
        from utils import StructuralFeatureExtractor
        print("✓ StructuralFeatureExtractor import successful")
    except ImportError as e:
        print(f"✗ StructuralFeatureExtractor import failed: {e}")
        return False
    
    try:
        import config_adaptive
        print("✓ config_adaptive import successful")
    except ImportError as e:
        print(f"✗ config_adaptive import failed: {e}")
        return False
    
    return True

def test_model_creation():
    """Test if the model can be created with dummy data."""
    print("\nTesting model creation...")
    
    try:
        import torch
        import dgl
        from models import AdaptiveContextGNN
        import config_adaptive as config
        
        # Create a simple heterogeneous graph for testing
        data_dict = {
            ('a', 'ap', 'p'): ([0, 1, 2], [0, 1, 2]),
            ('p', 'pa', 'a'): ([0, 1, 2], [0, 1, 2]),
            ('p', 'pt', 't'): ([0, 1, 2], [0, 1, 2]),
            ('t', 'tp', 'p'): ([0, 1, 2], [0, 1, 2]),
        }
        
        hg = dgl.heterograph(data_dict)
        print(f"✓ Created test heterograph: {hg}")
        
        # Test model creation
        model = AdaptiveContextGNN(hg, config.model_config)
        print(f"✓ AdaptiveContextGNN model created successfully")
        
        # Test forward pass
        with torch.no_grad():
            # Test compatibility mode
            emb_k1 = model(k=1)
            print(f"✓ Forward pass (k=1): {emb_k1.shape}")
            
            # Test adaptive mode
            emb_adaptive = model()
            print(f"✓ Forward pass (adaptive): {emb_adaptive.shape}")
            
            # Test adaptive weights
            weights = model.get_adaptive_weights()
            print(f"✓ Adaptive weights: {weights.shape}")
            print(f"  Sample weights: {weights[0]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loading():
    """Test if data can be loaded."""
    print("\nTesting data loading...")
    
    try:
        from utils import load_data
        import config_adaptive as config
        
        # Check if data file exists
        data_path = os.path.join(config.data_config['data_path'], 
                                config.data_config['dataset'], 
                                config.data_config['data_name'])
        
        if os.path.exists(data_path):
            print(f"✓ Data file found: {data_path}")
            
            # Try to load data
            dataloader = load_data(config.data_config)
            print(f"✓ Data loaded successfully")
            print(f"  Graph: {dataloader.heter_graph}")
            print(f"  Node types: {dataloader.heter_graph.ntypes}")
            print(f"  Edge types: {dataloader.heter_graph.etypes}")
            
            return True
        else:
            print(f"✗ Data file not found: {data_path}")
            print("  Please ensure DBLP.mat is in the correct location")
            return False
            
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=== Adaptive CP-GNN Implementation Test ===\n")
    
    tests = [
        ("Import Test", test_imports),
        ("Model Creation Test", test_model_creation),
        ("Data Loading Test", test_data_loading),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running {test_name}")
        print('='*50)
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print('='*50)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Your Adaptive CP-GNN implementation is ready.")
        print("\nNext steps:")
        print("1. Run: python main_adaptive.py --mode adaptive -n 0")
        print("2. Evaluate: python evaluate_adaptive.py --adaptive")
        print("3. Compare: python analyze_results.py")
    else:
        print(f"\n⚠️  {len(results) - passed} test(s) failed. Please fix the issues before running experiments.")

if __name__ == "__main__":
    main() 