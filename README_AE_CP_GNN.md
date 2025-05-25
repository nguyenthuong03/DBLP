# AE-CP-GNN: Attribute Enhanced Context-aware Graph Neural Network

This implementation extends the original CP-GNN model by incorporating DistilBERT-based text attribute encoding.

## Key Features

- DistilBERT integration for text attribute encoding
- Hybrid embeddings combining structural and semantic information  
- Configurable weighting parameter α
- Grid search for hyperparameter optimization
- Support for DBLP dataset with paper titles, author names, etc.

## Installation

```bash
pip install -r requirements_AE.txt
```

## Usage

Basic training:
```bash
python main_AE.py
```

Grid search:
```bash
python main_AE.py --grid_search
```

Custom alpha:
```bash
python main_AE.py -alpha 0.7
```

## Architecture

The model combines:
1. Structural embeddings from CP-GNN: z_i
2. Attribute embeddings from DistilBERT: h_i^attr
3. Combined embeddings: z_i^ae = α·z_i + (1-α)·h_i^attr

## Configuration

See `config_AE.py` for all parameters. Key settings:
- `embedding_dim`: 128 (as specified in requirements)
- `alpha`: 0.5 (default weighting parameter)
- `freeze_bert`: True (freeze DistilBERT for efficiency)

## Testing

Run tests to verify implementation:
```bash
python test_AE.py
``` 