# Dynamic CP-GNN (D-CP-GNN) Implementation

## Overview

This repository implements **Dynamic Context-aware Graph Neural Network (D-CP-GNN)**, an extension of the original CP-GNN model that handles dynamic heterogeneous graphs through temporal modeling with GRU (Gated Recurrent Unit).

## Key Features

### 🚀 Dynamic Graph Modeling
- **Temporal Snapshots**: Represents dynamic graphs as sequences of static snapshots {G₁, G₂, ..., Gₜ}
- **GRU-based Temporal Modeling**: Uses GRU to capture temporal dependencies between snapshots
- **Cumulative Growth**: Each snapshot includes all previous data plus new additions

### 🧠 Advanced Architecture
- **Context-aware Attention**: Multi-head attention mechanisms for heterogeneous graphs
- **Path Attention**: Context path attention for multi-hop relationships
- **Bilinear Attention**: Enhanced attention with bilinear transformations
- **Multi-hop Learning**: Configurable K-hop context aggregation

### 📊 Comprehensive Evaluation
- **Node Classification**: Author classification on DBLP dataset
- **Clustering**: Community detection with temporal evolution
- **Temporal Analysis**: Evolution tracking across snapshots

## Architecture

### D-CP-GNN Process Flow

1. **Graph Snapshot Division**: 
   - Divide dynamic graph into T temporal snapshots
   - Each snapshot Gₜ = (Vₜ, Eₜ) represents the graph state at time t

2. **Structural Embedding with CP-GNN**:
   - Apply original CP-GNN to each snapshot
   - Generate structural embeddings z_i^(t) for each node

3. **Temporal Modeling with GRU**:
   - Update embeddings across time: h_i^(t) = GRU(z_i^(t), h_i^(t-1))
   - Integrate structural and temporal information

4. **Training and Evaluation**:
   - Use dynamic embeddings h_i^(t) for downstream tasks
   - Evaluate on final snapshot for classification/clustering

### Model Configuration

```python
model_config = {
    'embedding_dim': 128,        # Reduced from 256 for efficiency
    'num_heads': 8,              # Multi-head attention
    'drop_out': 0.3,             # Node dropout rate
    'K_length': 3,               # 3-hop context
    'gru': True,                 # Enable GRU temporal modeling
    'path_attention': True,      # Context path attention
    'graph_attention': True,     # Graph-level attention
    'enable_bilinear': True      # Bilinear attention
}
```

## Installation

### Prerequisites
- Python 3.12
- PyTorch 2.4.0
- DGL (Deep Graph Library)
- CUDA 12.4 (optional, for GPU acceleration)

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd DBLP

# Install dependencies
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
pip install scikit-learn matplotlib scipy pandas numpy
```

## Usage

### Training Dynamic CP-GNN

```bash
# Train with default 5 snapshots
python main_dynamic.py

# Train with custom number of snapshots
python main_dynamic.py -s 7

# Train on specific GPU
python main_dynamic.py -n 1 -s 5
```

### Evaluation

```bash
# Evaluate latest model
python evaluate_dynamic.py

# Evaluate specific checkpoint
python evaluate_dynamic.py -path checkpoint/dynamic_DBLP/model_epoch_100

# Analyze temporal evolution
python evaluate_dynamic.py --analyze
```

### Configuration

The dynamic configuration is in `config_dynamic.py`:

```python
# Data configuration
data_config = {
    'dataset': 'DBLP',
    'primary_type': 'a',         # Author nodes
    'task': ['CF', 'CL'],        # Classification & Clustering
    'K_length': 3,               # 3-hop context
    'num_snapshots': 5           # 5 temporal snapshots
}

# Dynamic-specific settings
dynamic_config = {
    'num_snapshots': 5,
    'temporal_modeling': True,
    'gru_enabled': True,
    'snapshot_years': [2010, 2011, 2012, 2013, 2014],
    'cumulative_snapshots': True
}
```

## Dataset

### DBLP Dataset Structure
- **Papers (p)**: Academic publications
- **Authors (a)**: Paper authors (primary node type)
- **Conferences (c)**: Publication venues
- **Terms (t)**: Paper keywords/topics

### Temporal Snapshots
Since the original DBLP.mat doesn't contain temporal information, we simulate it by:
1. Dividing papers into time periods
2. Creating cumulative snapshots (each includes previous data)
3. Mapping connected authors, conferences, and terms

### Example Snapshot Division (T=5)
- **Snapshot 1 (2010)**: Papers 0-2875, connected entities
- **Snapshot 2 (2011)**: Papers 0-5750, connected entities  
- **Snapshot 3 (2012)**: Papers 0-8625, connected entities
- **Snapshot 4 (2013)**: Papers 0-11500, connected entities
- **Snapshot 5 (2014)**: Papers 0-14377, connected entities

## Results

### Performance Metrics
- **Node Classification**: Micro-F1, Macro-F1 scores
- **Clustering**: NMI, ARI, Purity, Silhouette scores
- **Temporal Analysis**: Cosine similarity between consecutive snapshots

### Expected Improvements
D-CP-GNN should show improvements over static CP-GNN in:
- **Temporal Consistency**: Better handling of evolving relationships
- **Community Evolution**: Tracking of community changes over time
- **Robustness**: More stable embeddings through temporal modeling

## File Structure

```
DBLP/
├── models/
│   ├── CGNN.py                 # Original CP-GNN implementation
│   ├── DynamicCGNN.py          # Dynamic CP-GNN implementation
│   └── __init__.py
├── utils/
│   ├── preprocess.py           # Original data preprocessing
│   ├── dynamic_preprocess.py   # Dynamic data preprocessing
│   ├── evluator.py            # Evaluation utilities
│   └── helper.py              # Helper functions
├── data/DBLP/                 # DBLP dataset files
├── config.py                  # Original configuration
├── config_dynamic.py          # Dynamic configuration
├── main.py                    # Original training script
├── main_dynamic.py            # Dynamic training script
├── evaluate.py                # Original evaluation
├── evaluate_dynamic.py        # Dynamic evaluation
└── README_Dynamic_CPGNN.md    # This file
```

## Key Differences from Original CP-GNN

| Aspect | Original CP-GNN | Dynamic CP-GNN |
|--------|----------------|----------------|
| **Graph Type** | Static | Dynamic (temporal snapshots) |
| **Temporal Modeling** | None | GRU-based |
| **Embedding Dimension** | 256 | 128 (optimized) |
| **Training Data** | Single graph | Sequence of snapshots |
| **Memory Usage** | Lower | Higher (multiple snapshots) |
| **Complexity** | O(N) | O(T×N) where T = snapshots |

## Hyperparameters

### Optimized Settings
- **Embedding Dimension**: 128 (reduced for efficiency)
- **Learning Rate**: 0.05 (as per original paper)
- **Attention Heads**: 8
- **Dropout Rate**: 0.3
- **K-hop Context**: 3
- **Temporal Snapshots**: 5 (T=5)

### GRU Configuration
- **Hidden Size**: 128 (matches embedding dimension)
- **Temporal Sequence**: Forward through all snapshots
- **Hidden State**: Carried between consecutive snapshots

## Troubleshooting

### Common Issues

1. **Memory Issues**
   ```bash
   # Reduce batch size or number of snapshots
   python main_dynamic.py -s 3  # Use 3 snapshots instead of 5
   ```

2. **CUDA Out of Memory**
   ```bash
   # Use CPU or smaller model
   python main_dynamic.py -n -1  # Force CPU usage
   ```

3. **Slow Training**
   ```bash
   # Reduce workers or batch size
   # Edit config_dynamic.py: 'sample_workers': 4, 'batch_size': 512
   ```

## Future Enhancements

### Potential Improvements
1. **Real Temporal Data**: Use actual publication timestamps
2. **Adaptive Snapshots**: Variable-length time windows
3. **Attention Visualization**: Temporal attention heatmaps
4. **Multi-task Learning**: Joint optimization across tasks
5. **Incremental Learning**: Online updates for new snapshots

### Research Directions
- **Temporal Attention**: Learn temporal importance weights
- **Graph Evolution Prediction**: Forecast future graph states
- **Dynamic Community Detection**: Real-time community tracking
- **Scalability**: Handling larger temporal sequences

## Citation

If you use this Dynamic CP-GNN implementation, please cite:

```bibtex
@article{dynamic_cpgnn_2024,
  title={Dynamic Context-aware Graph Neural Network for Heterogeneous Information Networks},
  author={[Your Name]},
  journal={[Conference/Journal]},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original CP-GNN paper and implementation
- DGL (Deep Graph Library) team
- PyTorch community
- DBLP dataset providers 