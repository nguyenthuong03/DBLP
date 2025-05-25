#!/usr/bin/env python
# Simple test script

print("=== Simple Test ===")

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
except ImportError as e:
    print(f"✗ PyTorch import failed: {e}")

try:
    import transformers
    print(f"✓ Transformers: {transformers.__version__}")
except ImportError as e:
    print(f"✗ Transformers import failed: {e}")

try:
    import dgl
    print(f"✓ DGL: {dgl.__version__}")
except ImportError as e:
    print(f"✗ DGL import failed: {e}")

try:
    from models.attribute_encoder import AttributeEncoder
    print("✓ AttributeEncoder import successful")
except ImportError as e:
    print(f"✗ AttributeEncoder import failed: {e}")

try:
    import config_AE
    print("✓ config_AE import successful")
except ImportError as e:
    print(f"✗ config_AE import failed: {e}")

print("=== Test Complete ===") 