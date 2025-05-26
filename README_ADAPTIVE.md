# Adaptive CP-GNN (A-CP-GNN) Implementation

## Overview

This repository implements **Adaptive Context-aware Graph Neural Network (A-CP-GNN)**, an enhancement to the original CP-GNN model that automatically determines optimal context path weights for each node using structural features and an MLP predictor.

## Key Improvements

### Adaptive Context Path Weighting
- **Problem**: Original CP-GNN uses fixed context path length K, requiring manual tuning
- **Solution**: A-CP-GNN uses an MLP to predict adaptive weights α_k for each node based on structural features
- **Benefit**: Eliminates hyperparameter tuning and adapts to node-specific neighborhood structures

### Structural Feature Extraction
- **Node Degree**: Number of direct connections
- **Context Neighbors**: Reachable nodes at each k-hop distance  
- **Clustering Coefficient**: Local connectivity measure
- **Normalization**: StandardScaler for stable training

### Adaptive Aggregation
- **Formula**: `z_i = Σ(α_k^(i) * c_i^k)` where α_k^(i) are node-specific weights
- **MLP Prediction**: `[α_1, α_2, ..., α_K] = softmax(MLP(f_i))`
- **Dynamic Weighting**: Each node gets personalized context path importance

## Usage

### Training Adaptive CP-GNN

```bash
# Pure adaptive training
python main_adaptive.py --mode adaptive -n 0

# Hybrid training (mix adaptive + traditional)
python main_adaptive.py --mode hybrid -n 0
```

### Evaluation

```bash
# Evaluate with adaptive embeddings
python evaluate_adaptive.py --adaptive

# Compare adaptive vs traditional
python evaluate_adaptive.py --compare

# Evaluate specific checkpoint
python evaluate_adaptive.py -path checkpoint/DBLP_adaptive/model.pth --adaptive
```

## Dependencies

- PyTorch 2.4.0+
- DGL 0.4.3+
- scikit-learn
- NumPy
- SciPy 