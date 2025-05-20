# Transformer-Enhanced Node Attributes for DBLP

This extension enhances the original CP-GNN model with transformer-based node attribute extraction. We use DistilBERT to encode semantic information from paper titles and categories to create rich author node attributes.

## Overview

The extension consists of:

1. **Transformer-Enhanced Feature Extraction** - Uses DistilBERT and YAKE to extract and encode node attributes
2. **TransformerEnhancedCPGNN Model** - A model that combines transformers with graph neural networks
3. **Example Scripts** - Both a simple example and a full training pipeline

## File Structure

```
DBLP/
├── utils/
│   └── transformer_enhanced.py    # Enhanced feature extraction
├── model/
│   └── CPGNN.py                   # TransformerEnhancedCPGNN model
├── main_transformer_enhanced.py   # Main training script
└── utils/transformer_enhanced_example.py  # Simple usage example
```

## Feature Extraction

The feature extraction pipeline works as follows:

1. Extract paper titles and categories for each author
2. Use YAKE to extract keywords from paper titles
3. Encode keywords with DistilBERT to get semantic embeddings
4. Encode categories with one-hot encoding
5. Combine these embeddings to create rich author node attributes

These enhanced attributes capture both the semantic meaning of research topics and the research areas, which can improve node classification and link prediction tasks.

## Usage

### Quick Start

To try out the transformer-enhanced features:

```bash
python utils/transformer_enhanced_example.py
```

This will:
- Load the DBLP dataset
- Extract enhanced features
- Create a t-SNE visualization
- Train a simple classifier on the enhanced features

### Full Model Training

To train the full TransformerEnhancedCPGNN model:

```bash
python main_transformer_enhanced.py --gpu 0
```

Command-line arguments:
- `--data_path`: Path to the dataset (default: `data/DBLP/`)
- `--hidden_dim`: Hidden dimension size (default: 128)
- `--num_layers`: Number of transformer layers (default: 2)
- `--num_heads`: Number of attention heads (default: 4)
- `--dropout`: Dropout rate (default: 0.1)
- `--lr`: Learning rate (default: 0.001)
- `--epochs`: Number of epochs (default: 100)
- `--gpu`: GPU ID, -1 for CPU (default: 0)
- `--config_file`: Configuration file (default: config.py)

## Model Architecture

The TransformerEnhancedCPGNN model combines:
1. A transformer encoder for initial node attribute processing
2. Traditional GNN layers for graph structure learning
3. Attention mechanisms for context path learning

The model architecture is designed to:
- Better capture semantic meaning in node attributes using transformers
- Preserve the context path learning capabilities of the original CP-GNN
- Enable better node classification and link prediction performance

## Requirements

Additional requirements beyond the original CP-GNN:
- transformers>=4.10.0
- yake>=0.4.8

These have been added to requirements.txt.

## Caching

To avoid recomputing features, the extracted features are cached by default. The cache location is specified as:
```
{data_config['data_path']}/{data_config['dataset']}/enhanced_features/author_features.pkl
```

You can manually load and save features using the provided functions:
```python
from utils.transformer_enhanced import save_features, load_features

# Save features
save_features(node_features, "path/to/cache.pkl")

# Load features
node_features = load_features("path/to/cache.pkl")
```

## Performance

The transformer-enhanced model typically provides improved performance on node classification and link prediction tasks compared to the original CP-GNN, especially when working with text-based attributes. 