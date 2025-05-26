#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/12/19
# @Author  : Adaptive CP-GNN Implementation
# @File    : evaluate_adaptive.py
# @Software: PyCharm
# @Describe: Evaluation script for Adaptive CP-GNN (A-CP-GNN)

from models import AdaptiveContextGNN
from utils import load_data, evaluate, load_latest_model, save_attention_matrix, generate_attention_heat_map, save_config
import torch
import importlib
import os
import argparse
import numpy as np


def evaluate_adaptive_task(config, checkpoint_path=None, use_adaptive=True):
    """
    Evaluate Adaptive CP-GNN model.
    
    Args:
        config: Configuration object
        checkpoint_path: Path to model checkpoint
        use_adaptive: Whether to use adaptive aggregation for evaluation
    """
    dataloader = load_data(config.data_config)
    hg = dataloader.heter_graph
    
    if config.data_config['dataset'] in ['AIFB', 'AM', 'BGS', 'MUTAG']:
        config.data_config['primary_type'] = dataloader.predict_category
        config.model_config['primary_type'] = dataloader.predict_category
    
    # Load model
    if not checkpoint_path:
        model = AdaptiveContextGNN(hg, config.model_config)
        model = load_latest_model(config.train_config['checkpoint_path'], model)
    else:
        config_path = os.path.join(checkpoint_path, 'config')
        config_path = os.path.relpath(config_path)
        config_file = config_path.replace(os.sep, '.')
        model_path = os.path.join(checkpoint_path, 'model.pth')
        config = importlib.import_module(config_file)
        model = AdaptiveContextGNN(hg, config.model_config)
        model.load_state_dict(torch.load(model_path, weights_only=True))
    
    model.eval()
    
    # Get embeddings
    with torch.no_grad():
        if use_adaptive:
            print("Using adaptive aggregated embeddings for evaluation...")
            p_emb = model.forward().detach().cpu().numpy()  # Adaptive aggregation
        else:
            print("Using traditional primary embeddings for evaluation...")
            p_emb = model.primary_emb.weight.detach().cpu().numpy()  # Traditional
    
    # Load evaluation data
    CF_data = dataloader.load_classification_data()
    
    # Evaluate
    result_save_path = evaluate(p_emb, CF_data, None, 
                               method=config.evaluate_config['method'],  
                               metric=config.data_config['task'], 
                               save_result=True,
                               result_path=config.evaluate_config['result_path'],
                               random_state=config.evaluate_config['random_state'],
                               max_iter=config.evaluate_config['max_iter'], 
                               n_jobs=config.evaluate_config['n_jobs'])
    
    if result_save_path:
        # Save configuration
        save_config(config, result_save_path)
        
        # Save model
        model_save_path = os.path.join(result_save_path, "model.pth")
        torch.save(model.state_dict(), model_save_path)
        
        # Save adaptive weights analysis
        if use_adaptive:
            analyze_and_save_adaptive_weights(model, result_save_path)
        
        # Save attention matrix
        attention_matrix_path = save_attention_matrix(model, result_save_path, config.data_config['K_length'])
        if attention_matrix_path and config.evaluate_config['save_heat_map']:
            generate_attention_heat_map(hg.ntypes, attention_matrix_path)


def analyze_and_save_adaptive_weights(model, result_path):
    """
    Analyze adaptive weights and save detailed statistics.
    
    Args:
        model: Trained Adaptive CP-GNN model
        result_path: Path to save results
    """
    print("Analyzing adaptive weights...")
    
    with torch.no_grad():
        adaptive_weights = model.get_adaptive_weights()
        
        # Basic statistics
        weights_np = adaptive_weights.cpu().numpy()
        mean_weights = np.mean(weights_np, axis=0)
        std_weights = np.std(weights_np, axis=0)
        min_weights = np.min(weights_np, axis=0)
        max_weights = np.max(weights_np, axis=0)
        
        # Save weights
        weights_save_path = os.path.join(result_path, "adaptive_weights.npy")
        np.save(weights_save_path, weights_np)
        
        # Save statistics
        stats = {
            'mean_weights_per_hop': mean_weights.tolist(),
            'std_weights_per_hop': std_weights.tolist(),
            'min_weights_per_hop': min_weights.tolist(),
            'max_weights_per_hop': max_weights.tolist(),
            'total_nodes': weights_np.shape[0],
            'num_hops': weights_np.shape[1]
        }
        
        import json
        stats_save_path = os.path.join(result_path, "adaptive_weights_stats.json")
        with open(stats_save_path, 'w') as f:
            json.dump(stats, f, indent=4)
        
        # Print analysis
        print(f"\n=== Adaptive Weights Analysis ===")
        print(f"Total nodes: {weights_np.shape[0]}")
        print(f"Number of hops: {weights_np.shape[1]}")
        print(f"Mean weights per hop: {mean_weights}")
        print(f"Std weights per hop: {std_weights}")
        print(f"Min weights per hop: {min_weights}")
        print(f"Max weights per hop: {max_weights}")
        
        # Show distribution of dominant hops
        dominant_hops = np.argmax(weights_np, axis=1)
        unique_hops, counts = np.unique(dominant_hops, return_counts=True)
        print(f"\nDominant hop distribution:")
        for hop, count in zip(unique_hops, counts):
            percentage = (count / len(dominant_hops)) * 100
            print(f"  Hop {hop+1}: {count} nodes ({percentage:.1f}%)")
        
        # Show some example weights
        print(f"\nExample adaptive weights (first 10 nodes):")
        for i in range(min(10, weights_np.shape[0])):
            weights_str = ", ".join([f"{w:.3f}" for w in weights_np[i]])
            dominant_hop = dominant_hops[i] + 1
            print(f"  Node {i}: [{weights_str}] (dominant: hop {dominant_hop})")
        
        print(f"Adaptive weights saved to: {weights_save_path}")
        print(f"Statistics saved to: {stats_save_path}")


def compare_adaptive_vs_traditional(config, checkpoint_path=None):
    """
    Compare performance between adaptive and traditional embeddings.
    
    Args:
        config: Configuration object
        checkpoint_path: Path to model checkpoint
    """
    print("=== Comparing Adaptive vs Traditional Embeddings ===")
    
    # Evaluate with adaptive embeddings
    print("\n1. Evaluating with Adaptive Embeddings:")
    evaluate_adaptive_task(config, checkpoint_path, use_adaptive=True)
    
    # Evaluate with traditional embeddings
    print("\n2. Evaluating with Traditional Embeddings:")
    evaluate_adaptive_task(config, checkpoint_path, use_adaptive=False)


if __name__ == "__main__":
    import config_adaptive as config

    parser = argparse.ArgumentParser(description='Evaluate Adaptive CP-GNN')
    parser.add_argument('-path', default=None, type=str, help='checkpoint path')
    parser.add_argument('--adaptive', action='store_true', help='use adaptive embeddings')
    parser.add_argument('--compare', action='store_true', help='compare adaptive vs traditional')
    args = parser.parse_args()
    
    if args.compare:
        compare_adaptive_vs_traditional(config, args.path)
    else:
        evaluate_adaptive_task(config, args.path, use_adaptive=args.adaptive) 