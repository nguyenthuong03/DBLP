#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/12/19
# @Author  : Adaptive CP-GNN Implementation
# @File    : main_adaptive.py
# @Software: PyCharm
# @Describe: Training script for Adaptive CP-GNN (A-CP-GNN)

from statistics import mean
import torch
import dgl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import argparse

from models import AdaptiveContextGNN
from evaluate import evaluate_task
from utils import load_data, EarlyStopping, load_latest_model, evaluate
import config_adaptive as config

parser = argparse.ArgumentParser(description='Adaptive CP-GNN Training')
parser.add_argument('-n', default=0, type=int, help='GPU ID')
parser.add_argument('--mode', default='adaptive', choices=['adaptive', 'hybrid'], 
                   help='Training mode: adaptive (pure adaptive) or hybrid (mixed)')
args = parser.parse_args()


def main(config):
    print("Loading data...")
    dataloader = load_data(config.data_config)
    if config.data_config['dataset'] in ['AIFB', 'AM', 'BGS', 'MUTAG']:
        config.data_config['primary_type'] = dataloader.predict_category
        config.model_config['primary_type'] = dataloader.predict_category
    
    hg = dataloader.heter_graph
    edges_data_dict = dataloader.load_train_k_context_edges(hg, config.data_config['K_length'],
                                                            config.data_config['primary_type'],
                                                            config.train_config['pos_num_for_each_hop'],
                                                            config.train_config['neg_num_for_each_hop'])

    CF_data = dataloader.load_classification_data()
    device = torch.device('cuda:{}'.format(args.n) if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    dataloader_dict = {key: DataLoader(dataset, batch_size=config.train_config['batch_size'],
                                       num_workers=config.train_config['sample_workers'], 
                                       collate_fn=dataset.collate,
                                       shuffle=True, pin_memory=True)
                       for key, dataset in edges_data_dict.items() if len(dataset) > 0}

    print("Initializing Adaptive CP-GNN model...")
    model = AdaptiveContextGNN(hg, config.model_config)
    model = model.to(device)

    if config.train_config['continue']:
        model = load_latest_model(config.train_config['checkpoint_path'], model)

    stopper = EarlyStopping(checkpoint_path=config.train_config['checkpoint_path'], config=config,
                            patience=config.train_config['patience'])

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.train_config['lr'], 
                                 weight_decay=config.train_config['l2'])
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=config.train_config['factor'], 
                                 patience=config.train_config['patience'] // 3, verbose=True)

    print("Starting Adaptive CP-GNN training...")
    print(f"Training mode: {args.mode}")
    print(f"Use adaptive loss: {config.train_config['use_adaptive_loss']}")
    
    for epoch in range(config.train_config['total_epoch']):
        model.train()
        running_loss = []
        
        # Clear cache periodically to update structural features
        if epoch % config.train_config['structural_feature_update_freq'] == 0:
            model.clear_cache()
            print(f"Epoch {epoch}: Updated structural features cache")
        
        if args.mode == 'adaptive' and config.train_config['use_adaptive_loss']:
            # Pure adaptive training: use adaptive loss with all k-hops combined
            running_loss = train_adaptive_mode(model, dataloader_dict, optimizer, device)
        else:
            # Hybrid training: mix adaptive and traditional k-hop losses
            running_loss = train_hybrid_mode(model, dataloader_dict, optimizer, device, config)
        
        mean_loss = mean(running_loss)
        scheduler.step(mean_loss)
        
        print(f"Epoch:{epoch}/{config.train_config['total_epoch']} Loss: {mean_loss:.6f}")
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            if args.mode == 'adaptive':
                # Use adaptive aggregated embeddings for evaluation
                p_emb = model.forward().detach().cpu().numpy()
            else:
                # Use traditional primary embeddings
                p_emb = model.primary_emb.weight.detach().cpu().numpy()
            
            evaluate(p_emb, CF_data, None, config.evaluate_config['method'], 
                    metric=config.data_config['task'],
                    random_state=config.evaluate_config['random_state'],
                    max_iter=config.evaluate_config['max_iter'], 
                    n_jobs=config.evaluate_config['n_jobs'])
        
        # Early stopping
        early_stop = stopper.step(mean_loss, model)
        if early_stop:
            print("Early stopping triggered!")
            break
    
    print("Training completed. Evaluating final model...")
    checkpoint_path = stopper.filepath
    evaluate_task(config, checkpoint_path)


def train_adaptive_mode(model, dataloader_dict, optimizer, device):
    """Train using pure adaptive loss."""
    running_loss = []
    
    # Collect all positive and negative samples from all k-hops
    all_pos_src, all_pos_dst = [], []
    all_neg_src, all_neg_dst = [], []
    
    for k_hop, dataloader in dataloader_dict.items():
        for pos_src, pos_dst, neg_src, neg_dst in dataloader:
            pos_src, pos_dst = pos_src.to(device), pos_dst.to(device)
            neg_src, neg_dst = neg_src.to(device), neg_dst.to(device)
            
            all_pos_src.append(pos_src)
            all_pos_dst.append(pos_dst)
            all_neg_src.append(neg_src)
            all_neg_dst.append(neg_dst)
    
    # Combine all samples
    if all_pos_src:
        combined_pos_src = torch.cat(all_pos_src)
        combined_pos_dst = torch.cat(all_pos_dst)
        combined_neg_src = torch.cat(all_neg_src)
        combined_neg_dst = torch.cat(all_neg_dst)
        
        # Compute adaptive loss
        loss = model.get_adaptive_loss(combined_pos_src, combined_pos_dst, 
                                     combined_neg_src, combined_neg_dst)
        
        running_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return running_loss


def train_hybrid_mode(model, dataloader_dict, optimizer, device, config):
    """Train using hybrid approach: mix adaptive and traditional losses."""
    running_loss = []
    
    # Traditional k-hop specific training
    for k_hop, dataloader in dataloader_dict.items():
        for pos_src, pos_dst, neg_src, neg_dst in dataloader:
            pos_src, pos_dst = pos_src.to(device), pos_dst.to(device)
            neg_src, neg_dst = neg_src.to(device), neg_dst.to(device)
            
            # Traditional k-hop loss
            p_context_emb = model(k_hop)  # Get k-hop specific embedding
            p_emb = model.primary_emb.weight
            traditional_loss = model.get_loss(k_hop, pos_src, pos_dst, neg_src, neg_dst, 
                                            p_emb, p_context_emb)
            
            # Adaptive loss
            adaptive_loss = model.get_adaptive_loss(pos_src, pos_dst, neg_src, neg_dst)
            
            # Combined loss
            total_loss = (traditional_loss + 
                         config.train_config['adaptive_loss_weight'] * adaptive_loss)
            
            running_loss.append(total_loss.item())
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
    
    return running_loss


def analyze_adaptive_weights(model, save_path=None):
    """Analyze and optionally save the learned adaptive weights."""
    model.eval()
    with torch.no_grad():
        adaptive_weights = model.get_adaptive_weights()
        
        print("\n=== Adaptive Weights Analysis ===")
        print(f"Adaptive weights shape: {adaptive_weights.shape}")
        print(f"Mean weights per hop: {adaptive_weights.mean(dim=0)}")
        print(f"Std weights per hop: {adaptive_weights.std(dim=0)}")
        
        # Print some example weights
        print("\nExample adaptive weights for first 10 nodes:")
        for i in range(min(10, adaptive_weights.shape[0])):
            weights_str = ", ".join([f"{w:.3f}" for w in adaptive_weights[i]])
            print(f"Node {i}: [{weights_str}]")
        
        if save_path:
            torch.save(adaptive_weights, save_path)
            print(f"Adaptive weights saved to: {save_path}")


if __name__ == "__main__":
    main(config)
    
    # Analyze adaptive weights after training
    print("\nAnalyzing learned adaptive weights...")
    # Note: This would need to load the trained model for analysis
    # analyze_adaptive_weights(model, "adaptive_weights.pt") 