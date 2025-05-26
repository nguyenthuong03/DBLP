#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/12/19
# @Author  : Dynamic CP-GNN Implementation
# @File    : evaluate_dynamic.py
# @Software: PyCharm
# @Describe: Evaluation script for Dynamic CP-GNN (D-CP-GNN)

from models import DynamicContextGNN
from utils import load_dynamic_data, evaluate, load_latest_model, save_attention_matrix, generate_attention_heat_map, save_config
import torch
import importlib
import os
import argparse


def evaluate_dynamic_task(config, checkpoint_path=None):
    """
    Evaluate the trained Dynamic CP-GNN model
    
    Args:
        config: Configuration object
        checkpoint_path: Path to saved model checkpoint
    """
    print("=" * 60)
    print("Dynamic CP-GNN (D-CP-GNN) Evaluation")
    print("=" * 60)
    
    # Load dynamic data
    print(f"Loading dynamic DBLP data with {config.dynamic_config['num_snapshots']} snapshots...")
    dataloader = load_dynamic_data(config.data_config, num_snapshots=config.dynamic_config['num_snapshots'])
    snapshots = dataloader.get_all_snapshots()
    
    print(f"Loaded {len(snapshots)} temporal snapshots:")
    for i, snapshot in enumerate(snapshots):
        print(f"  Snapshot {i+1} ({config.dynamic_config['snapshot_years'][i]}): "
              f"{snapshot.number_of_nodes('p')} papers, "
              f"{snapshot.number_of_nodes('a')} authors, "
              f"{snapshot.number_of_nodes('c')} conferences, "
              f"{snapshot.number_of_nodes('t')} terms")
    
    # Handle special datasets
    if config.data_config['dataset'] in ['AIFB', 'AM', 'BGS', 'MUTAG']:
        config.data_config['primary_type'] = dataloader.predict_category
        config.model_config['primary_type'] = dataloader.predict_category
    
    # Load model
    if not checkpoint_path:
        print("Loading model from latest checkpoint...")
        model = DynamicContextGNN(snapshots, config.model_config)
        model = load_latest_model(config.train_config['checkpoint_path'], model)
    else:
        print(f"Loading model from: {checkpoint_path}")
        config_path = os.path.join(checkpoint_path, 'config')
        config_path = os.path.relpath(config_path)
        config_file = config_path.replace(os.sep, '.')
        model_path = os.path.join(checkpoint_path, 'model.pth')
        
        # Load config from checkpoint
        try:
            config = importlib.import_module(config_file)
        except ImportError:
            print("Warning: Could not load config from checkpoint, using current config")
        
        model = DynamicContextGNN(snapshots, config.model_config)
        model.load_state_dict(torch.load(model_path, weights_only=True))
    
    # Set device and move model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Get final dynamic embeddings
    print("Generating dynamic embeddings...")
    with torch.no_grad():
        # Forward through temporal sequence to get final embeddings
        temporal_embeddings, final_embeddings = model.forward_temporal_sequence(
            config.data_config['K_length'], device
        )
        p_emb = final_embeddings.detach().cpu().numpy()
    
    print(f"Generated embeddings shape: {p_emb.shape}")
    
    # Load classification data and evaluate
    print("Loading classification data...")
    CF_data = dataloader.load_classification_data()
    
    print("=" * 60)
    print("Evaluation Results:")
    print("=" * 60)
    
    # Perform evaluation
    result_save_path = evaluate(
        p_emb, CF_data, None, 
        method=config.evaluate_config['method'],  
        metric=config.data_config['task'], 
        save_result=True,
        result_path=config.evaluate_config['result_path'],
        random_state=config.evaluate_config['random_state'],
        max_iter=config.evaluate_config['max_iter'], 
        n_jobs=config.evaluate_config['n_jobs']
    )
    
    # Save results and model artifacts
    if result_save_path:
        print(f"\nSaving results to: {result_save_path}")
        
        # Save configuration
        save_config(config, result_save_path)
        
        # Save model state
        model_save_path = os.path.join(result_save_path, "model.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to: {model_save_path}")
        
        # Save temporal embeddings
        temporal_emb_path = os.path.join(result_save_path, "temporal_embeddings.pth")
        torch.save(temporal_embeddings, temporal_emb_path)
        print(f"Temporal embeddings saved to: {temporal_emb_path}")
        
        # Save attention matrices
        attention_matrix_path = save_attention_matrix(model, result_save_path, config.data_config['K_length'])
        if attention_matrix_path and config.evaluate_config['save_heat_map']:
            # Use final snapshot for attention visualization
            final_snapshot = snapshots[-1]
            generate_attention_heat_map(final_snapshot.ntypes, attention_matrix_path)
            print(f"Attention heatmaps saved")
        
        # Save snapshot information
        snapshot_info_path = os.path.join(result_save_path, "snapshot_info.txt")
        with open(snapshot_info_path, 'w') as f:
            f.write("Dynamic CP-GNN Snapshot Information\n")
            f.write("=" * 40 + "\n")
            f.write(f"Number of snapshots: {len(snapshots)}\n")
            f.write(f"Snapshot years: {config.dynamic_config['snapshot_years']}\n\n")
            
            for i, snapshot in enumerate(snapshots):
                f.write(f"Snapshot {i+1} ({config.dynamic_config['snapshot_years'][i]}):\n")
                f.write(f"  Papers: {snapshot.number_of_nodes('p')}\n")
                f.write(f"  Authors: {snapshot.number_of_nodes('a')}\n")
                f.write(f"  Conferences: {snapshot.number_of_nodes('c')}\n")
                f.write(f"  Terms: {snapshot.number_of_nodes('t')}\n")
                f.write(f"  Total edges: {snapshot.number_of_edges()}\n\n")
        
        print(f"Snapshot information saved to: {snapshot_info_path}")
        
        print("=" * 60)
        print("Evaluation completed successfully!")
        print("=" * 60)
        
        return result_save_path
    
    return None


def analyze_temporal_evolution(config, checkpoint_path=None):
    """
    Analyze how embeddings evolve across temporal snapshots
    """
    print("=" * 60)
    print("Temporal Evolution Analysis")
    print("=" * 60)
    
    # Load data and model (similar to evaluate_dynamic_task)
    dataloader = load_dynamic_data(config.data_config, num_snapshots=config.dynamic_config['num_snapshots'])
    snapshots = dataloader.get_all_snapshots()
    
    if not checkpoint_path:
        model = DynamicContextGNN(snapshots, config.model_config)
        model = load_latest_model(config.train_config['checkpoint_path'], model)
    else:
        model = DynamicContextGNN(snapshots, config.model_config)
        model.load_state_dict(torch.load(os.path.join(checkpoint_path, 'model.pth'), weights_only=True))
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Get embeddings for each snapshot
    with torch.no_grad():
        temporal_embeddings, _ = model.forward_temporal_sequence(config.data_config['K_length'], device)
    
    # Analyze embedding evolution
    print("Analyzing embedding evolution across snapshots...")
    
    for i in range(1, len(temporal_embeddings)):
        prev_emb = temporal_embeddings[i-1].cpu().numpy()
        curr_emb = temporal_embeddings[i].cpu().numpy()
        
        # Calculate cosine similarity between consecutive snapshots
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = []
        
        min_nodes = min(prev_emb.shape[0], curr_emb.shape[0])
        for j in range(min_nodes):
            sim = cosine_similarity([prev_emb[j]], [curr_emb[j]])[0][0]
            similarities.append(sim)
        
        avg_similarity = sum(similarities) / len(similarities)
        print(f"Snapshot {i} -> {i+1}: Average cosine similarity = {avg_similarity:.4f}")
    
    print("Temporal evolution analysis completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dynamic CP-GNN Evaluation')
    parser.add_argument('-path', default=None, type=str, help='Checkpoint path')
    parser.add_argument('-s', '--snapshots', default=5, type=int, help='Number of temporal snapshots')
    parser.add_argument('--analyze', action='store_true', help='Perform temporal evolution analysis')
    args = parser.parse_args()
    
    # Import configuration
    import config_dynamic as config
    
    # Update number of snapshots from command line argument
    if args.snapshots != 5:
        config.data_config['num_snapshots'] = args.snapshots
        config.dynamic_config['num_snapshots'] = args.snapshots
        config.dynamic_config['snapshot_years'] = list(range(2010, 2010 + args.snapshots))
    
    if args.analyze:
        analyze_temporal_evolution(config, args.path)
    else:
        evaluate_dynamic_task(config, args.path) 