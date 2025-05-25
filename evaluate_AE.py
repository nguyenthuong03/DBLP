#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/12/19
# @Author  : AE-CP-GNN Extension
# @File    : evaluate_AE.py
# @Software: PyCharm
# @Describe: Evaluation script for AE-CP-GNN

import torch
import argparse
import os
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.cluster import KMeans

from models import AE_ContextGNN, create_ae_model_config
from utils import load_data, evaluate
import config_AE

parser = argparse.ArgumentParser(description='AE-CP-GNN Evaluation')
parser.add_argument('-path', default=None, type=str, help='Checkpoint path')
parser.add_argument('-alpha', default=0.5, type=float, help='Alpha value for evaluation')
parser.add_argument('--compare', action='store_true', help='Compare with baseline CP-GNN')
args = parser.parse_args()


def load_ae_model(checkpoint_path, config, device):
    """Load AE-CP-GNN model from checkpoint."""
    # Load data first
    dataloader = load_data(config.data_config)
    hg = dataloader.heter_graph
    
    # Create model
    model_config = create_ae_model_config(
        config.model_config,
        alpha=args.alpha,
        max_text_length=128,
        freeze_bert=True
    )
    
    model = AE_ContextGNN(hg, model_config, config.data_config)
    
    # Load checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        if os.path.isdir(checkpoint_path):
            model_path = os.path.join(checkpoint_path, 'model.pth')
        else:
            model_path = checkpoint_path
            
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            print(f"Loaded model from: {model_path}")
        else:
            print(f"Warning: Model file not found at {model_path}")
    
    return model.to(device), dataloader


def evaluate_embeddings(embeddings, CF_data, method='LR'):
    """Evaluate embeddings on classification task."""
    features, labels, num_classes, train_idx, test_idx = CF_data
    
    # Use embeddings as features
    X = embeddings
    y = labels
    
    # Ensure X and y have the same length
    min_len = min(len(X), len(y))
    X = X[:min_len]
    y = y[:min_len]
    
    print(f"Embeddings shape: {X.shape}, Labels shape: {y.shape}")
    
    # Filter valid labels
    valid_mask = (y >= 0) & (y < num_classes)
    X = X[valid_mask]
    y = y[valid_mask]
    
    # Create new train/test splits based on valid indices
    valid_indices = np.where(valid_mask)[0]
    train_mask = np.isin(valid_indices, train_idx)
    test_mask = np.isin(valid_indices, test_idx)
    
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Classes: {num_classes}, Label distribution: {np.bincount(y_train)}")
    
    # Classification evaluation
    if method == 'LR':
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=1000, random_state=42)
    else:
        from sklearn.svm import SVC
        clf = SVC(random_state=42)
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    
    print(f"Classification Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    
    # Clustering evaluation
    kmeans = KMeans(n_clusters=num_classes, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    # Calculate clustering metrics
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    
    # Filter for labeled data only
    labeled_mask = y >= 0
    ari = adjusted_rand_score(y[labeled_mask], cluster_labels[labeled_mask])
    nmi = normalized_mutual_info_score(y[labeled_mask], cluster_labels[labeled_mask])
    
    print(f"Clustering ARI: {ari:.4f}")
    print(f"Clustering NMI: {nmi:.4f}")
    
    return {
        'classification_accuracy': accuracy,
        'clustering_ari': ari,
        'clustering_nmi': nmi
    }


def compare_with_baseline(config, device):
    """Compare AE-CP-GNN with baseline CP-GNN."""
    print("=== Comparing AE-CP-GNN with Baseline CP-GNN ===")
    
    dataloader = load_data(config.data_config)
    hg = dataloader.heter_graph
    CF_data = dataloader.load_classification_data()
    
    results = {}
    
    # Evaluate baseline CP-GNN
    print("\n--- Baseline CP-GNN ---")
    from models import ContextGNN
    baseline_model = ContextGNN(hg, config.model_config).to(device)
    
    with torch.no_grad():
        baseline_emb = baseline_model(config.data_config['K_length']).cpu().numpy()
        print(f"Baseline embeddings shape: {baseline_emb.shape}")
        results['baseline'] = evaluate_embeddings(baseline_emb, CF_data)
    
    # Evaluate AE-CP-GNN with different alpha values
    alphas = [0.3, 0.5, 0.7]
    for alpha in alphas:
        print(f"\n--- AE-CP-GNN (α={alpha}) ---")
        
        ae_config = create_ae_model_config(
            config.model_config,
            alpha=alpha,
            max_text_length=128,
            freeze_bert=True
        )
        
        ae_model = AE_ContextGNN(hg, ae_config, config.data_config).to(device)
        
        with torch.no_grad():
            ae_emb = ae_model(config.data_config['K_length']).cpu().numpy()
            results[f'ae_alpha_{alpha}'] = evaluate_embeddings(ae_emb, CF_data)
    
    # Summary comparison
    print("\n=== Comparison Summary ===")
    print(f"{'Method':<20} {'Accuracy':<10} {'ARI':<8} {'NMI':<8}")
    print("-" * 50)
    
    for method, metrics in results.items():
        print(f"{method:<20} {metrics['classification_accuracy']:<10.4f} "
              f"{metrics['clustering_ari']:<8.4f} {metrics['clustering_nmi']:<8.4f}")
    
    return results


def main():
    """Main evaluation function."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.compare:
        # Compare with baseline
        compare_with_baseline(config_AE, device)
    else:
        # Evaluate specific checkpoint
        if not args.path:
            print("Please provide checkpoint path with -path argument")
            return
        
        print(f"Evaluating AE-CP-GNN checkpoint: {args.path}")
        print(f"Alpha value: {args.alpha}")
        
        # Load model
        model, dataloader = load_ae_model(args.path, config_AE, device)
        CF_data = dataloader.load_classification_data()
        
        # Get embeddings
        model.eval()
        with torch.no_grad():
            embeddings = model(config_AE.data_config['K_length']).cpu().numpy()
        
        print(f"Embeddings shape: {embeddings.shape}")
        
        # Evaluate
        results = evaluate_embeddings(embeddings, CF_data)
        
        print("\n=== Final Results ===")
        print(f"Classification Accuracy: {results['classification_accuracy']:.4f}")
        print(f"Clustering ARI: {results['clustering_ari']:.4f}")
        print(f"Clustering NMI: {results['clustering_nmi']:.4f}")


if __name__ == "__main__":
    main() 