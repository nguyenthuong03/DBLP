#!/usr/bin/env python
# Minimal test to check basic functionality

def test_basic():
    """Test basic functionality."""
    print("=== Basic Test ===")
    try:
        # Test basic imports
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        # Test if we can create a simple tensor
        x = torch.randn(3, 4)
        print(f"Created tensor with shape: {x.shape}")
        
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_transformers():
    """Test transformers import."""
    print("=== Transformers Test ===")
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
        return True
    except Exception as e:
        print(f"Transformers error: {e}")
        return False

def test_attribute_encoder_simple():
    """Test simple attribute encoder import."""
    print("=== Simple AttributeEncoder Test ===")
    try:
        from models.attribute_encoder import AttributeEncoder
        print("AttributeEncoder imported successfully")
        return True
    except Exception as e:
        print(f"AttributeEncoder error: {e}")
        return False

if __name__ == "__main__":
    tests = [
        ("Basic", test_basic),
        ("Transformers", test_transformers),
        ("AttributeEncoder", test_attribute_encoder_simple),
    ]
    
    for name, test_func in tests:
        print(f"\nRunning {name} test...")
        result = test_func()
        print(f"{name}: {'PASSED' if result else 'FAILED'}") 