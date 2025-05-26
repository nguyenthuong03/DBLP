#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Analysis script for comparing CP-GNN vs A-CP-GNN results

import json
import numpy as np
import os
import matplotlib.pyplot as plt

def load_results(result_path):
    """Load results from a result directory."""
    result_file = os.path.join(result_path, 'result.json')
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            return json.load(f)
    return None

def analyze_adaptive_weights(weights_path):
    """Analyze adaptive weights distribution."""
    if os.path.exists(weights_path):
        weights = np.load(weights_path)
        print(f"Adaptive Weights Analysis:")
        print(f"Shape: {weights.shape}")
        print(f"Mean per hop: {np.mean(weights, axis=0)}")
        print(f"Std per hop: {np.std(weights, axis=0)}")
        
        # Plot weight distribution
        plt.figure(figsize=(10, 6))
        for i in range(weights.shape[1]):
            plt.hist(weights[:, i], alpha=0.7, label=f'Hop {i+1}', bins=30)
        plt.xlabel('Weight Value')
        plt.ylabel('Frequency')
        plt.title('Distribution of Adaptive Weights per Hop')
        plt.legend()
        plt.savefig('adaptive_weights_distribution.png')
        plt.show()
        
        return weights
    return None

def compare_results():
    """Compare results between original and adaptive CP-GNN."""
    
    # Paths to result directories
    original_path = "result/DBLP"
    adaptive_path = "result/DBLP_adaptive"
    
    print("=== CP-GNN vs A-CP-GNN Comparison ===\n")
    
    # Load results
    original_results = None
    adaptive_results = None
    
    # Find latest result directories
    if os.path.exists(original_path):
        subdirs = [d for d in os.listdir(original_path) if os.path.isdir(os.path.join(original_path, d))]
        if subdirs:
            latest_original = max(subdirs)
            original_results = load_results(os.path.join(original_path, latest_original))
    
    if os.path.exists(adaptive_path):
        subdirs = [d for d in os.listdir(adaptive_path) if os.path.isdir(os.path.join(adaptive_path, d))]
        if subdirs:
            latest_adaptive = max(subdirs)
            adaptive_results = load_results(os.path.join(adaptive_path, latest_adaptive))
            
            # Analyze adaptive weights
            weights_path = os.path.join(adaptive_path, latest_adaptive, 'adaptive_weights.npy')
            analyze_adaptive_weights(weights_path)
    
    # Compare metrics
    if original_results and adaptive_results:
        print("Performance Comparison:")
        print("-" * 50)
        
        metrics = ['CF', 'CL']
        for metric in metrics:
            if metric in original_results and metric in adaptive_results:
                print(f"\n{metric} Task:")
                orig = original_results[metric]
                adapt = adaptive_results[metric]
                
                for key in orig:
                    if key in adapt:
                        orig_val = orig[key]
                        adapt_val = adapt[key]
                        improvement = ((adapt_val - orig_val) / orig_val) * 100
                        print(f"  {key}:")
                        print(f"    Original: {orig_val:.4f}")
                        print(f"    Adaptive: {adapt_val:.4f}")
                        print(f"    Improvement: {improvement:+.2f}%")
    
    else:
        print("Results not found. Make sure to run experiments first.")
        print("Expected paths:")
        print(f"  Original: {original_path}")
        print(f"  Adaptive: {adaptive_path}")

if __name__ == "__main__":
    compare_results() 