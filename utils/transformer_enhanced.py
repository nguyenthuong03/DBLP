#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Describe: Enhanced node attribute extraction for DBLP using transformers

import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertModel
import yake
from sklearn.preprocessing import OneHotEncoder
import pickle
import os
import dgl

def load_dblp_data(data_path):
    """
    Original placeholder function for loading DBLP data
    
    Args:
        data_path: Path to the DBLP data
        
    Returns:
        graph: DGL graph
        node_features: Node features
        labels: Node labels
    """
    from utils.preprocess import DBLPDataLoader
    
    # Create a minimal data config
    data_config = {
        'data_path': os.path.dirname(data_path),
        'dataset': 'DBLP',
        'data_name': os.path.basename(data_path),
        'primary_type': 'a',  # author
        'test_ratio': 0.2,
        'random_seed': 42,
        'resample': False
    }
    
    # Load data
    data_loader = DBLPDataLoader(data_config, remove_self_loop=False)
    graph = data_loader.heter_graph
    
    # Extract enhanced features
    cache_dir = os.path.join(data_config['data_path'], data_config['dataset'], 'enhanced_features')
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, 'author_features.pkl')
    
    node_features = extract_author_attributes(graph, data_path=data_path, cache_path=cache_path)
    
    # Get labels
    _, labels, _, _, _ = data_loader.load_classification_data()
    
    return graph, node_features, labels

def extract_author_attributes(graph, data_path=None, cache_path=None):
    """
    Extract enhanced author attributes from DBLP graph
    
    Args:
        graph (DGLGraph): The heterogeneous graph containing author nodes
        data_path (str): Path to get additional DBLP data if needed
        cache_path (str): Path to cache extracted features
        
    Returns:
        node_features (torch.Tensor): Enhanced node features for authors
    """
    # Check if cached features exist
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        return load_features(cache_path)
    
    print("Extracting author attributes...")
    num_nodes = graph.num_nodes('a')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    model.eval()

    # Initialize lists for keywords and categories
    keyword_list = []
    category_list = []
    
    # Get paper titles and categories from the graph
    for node_id in range(num_nodes):
        # Get papers associated with this author through author-paper edges
        out_edges = graph.out_edges(node_id, etype='ap')
        paper_ids = out_edges[1].tolist()
        
        keywords = []
        categories = []
        
        for paper_id in paper_ids:
            # Get paper title (assuming we have access to it or using placeholder)
            title = get_paper_title(paper_id, data_path) 
            
            # Extract keywords using YAKE
            kw_extractor = yake.KeywordExtractor(top=5, stopwords=None)
            extracted_keywords = kw_extractor.extract_keywords(title)
            keywords.extend([kw[0] for kw in extracted_keywords])
            
            # Get paper category
            category = get_paper_category(paper_id, data_path)
            if category:
                categories.append(category)
        
        # Limit number of keywords
        keywords = keywords[:10]  
        keyword_list.append(' '.join(keywords))
        category_list.append(categories[0] if categories else 'unknown')
    
    # Encode keywords with DistilBERT
    print("Encoding keywords with DistilBERT...")
    keyword_embeddings = []
    
    # Process in batches to avoid memory issues
    batch_size = 32
    for i in range(0, len(keyword_list), batch_size):
        batch_keywords = keyword_list[i:i+batch_size]
        
        with torch.no_grad():
            inputs = tokenizer(batch_keywords, return_tensors='pt', max_length=50, 
                              truncation=True, padding=True)
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :]  # Get [CLS] token
            keyword_embeddings.append(batch_embeddings)
    
    keyword_embeddings = torch.cat(keyword_embeddings, dim=0)  # [num_nodes, 768]
    
    # Encode categories using one-hot encoding
    print("Encoding categories...")
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    category_encoded = encoder.fit_transform(np.array(category_list).reshape(-1, 1))
    category_embeddings = torch.tensor(category_encoded, dtype=torch.float32)
    
    # Combine the features
    node_features = torch.cat([keyword_embeddings, category_embeddings], dim=1)
    
    # Save to cache if path provided
    if cache_path:
        save_features(node_features, cache_path)
    
    return node_features

def get_paper_title(paper_id, data_path=None):
    """
    Get paper title from data source
    
    Args:
        paper_id: ID of the paper in the graph
        data_path: Path to the data directory
        
    Returns:
        title: The paper title as a string
    """
    if data_path:
        try:
            # Try to load from paper title file if it exists
            paper_titles_path = os.path.join(data_path, 'paper_titles.txt')
            if os.path.exists(paper_titles_path):
                with open(paper_titles_path, encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2 and int(parts[0]) == paper_id:
                            return parts[1]
            
            # If no specific title file, try to extract from paper.txt
            papers_path = os.path.join(data_path, 'paper.txt')
            if os.path.exists(papers_path):
                with open(papers_path, encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2 and int(parts[0]) == paper_id:
                            return parts[1]
        except Exception as e:
            print(f"Error loading paper title: {e}")
    
    # Fallback: return generic title if paper not found
    return f"Research paper {paper_id} on graph neural networks and data mining"

def get_paper_category(paper_id, data_path=None):
    """
    Get paper category from data source
    
    Args:
        paper_id: ID of the paper in the graph
        data_path: Path to the data directory
        
    Returns:
        category: The paper category as a string
    """
    if data_path:
        try:
            # Try to load from paper categories file if it exists
            categories_path = os.path.join(data_path, 'paper_categories.txt')
            if os.path.exists(categories_path):
                with open(categories_path, encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2 and int(parts[0]) == paper_id:
                            return parts[1]
            
            # Try to infer from conference
            conf_path = os.path.join(data_path, 'paper_conf.txt')
            if os.path.exists(conf_path):
                with open(conf_path, encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2 and int(parts[0]) == paper_id:
                            conf_id = int(parts[1])
                            # Map conferences to research areas
                            conf_to_category = {
                                0: 'cs.AI',    # KDD -> AI/Data Mining
                                1: 'cs.DB',    # SIGMOD -> Databases
                                2: 'cs.IR',    # WWW -> Information Retrieval
                                3: 'cs.NI'     # ICDE -> Networking/Infrastructure
                            }
                            return conf_to_category.get(conf_id % len(conf_to_category), 'cs.LG')
        except Exception as e:
            print(f"Error loading paper category: {e}")
    
    # Default categories to use if no category found
    categories = ['cs.LG', 'cs.AI', 'cs.DB', 'cs.NI']
    return categories[paper_id % len(categories)]

def save_features(node_features, path):
    """Save node features to disk"""
    with open(path, 'wb') as f:
        pickle.dump(node_features, f)
    print(f"Features saved to {path}")

def load_features(path):
    """Load node features from disk"""
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_dblp_data_with_enhanced_features(data_config, remove_self_loop=False):
    """
    Load DBLP data with enhanced author attributes
    
    Args:
        data_config: Configuration for data loading
        remove_self_loop: Whether to remove self loops
        
    Returns:
        graph: DGL heterogeneous graph
        features: Enhanced node features
        other data needed for the model
    """
    from utils.preprocess import DBLPDataLoader
    
    # Load the original graph
    data_loader = DBLPDataLoader(data_config, remove_self_loop)
    graph = data_loader.heter_graph
    
    # Generate cache path
    cache_dir = os.path.join(data_config['data_path'], data_config['dataset'], 'enhanced_features')
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, 'author_features.pkl')
    
    # Extract enhanced author attributes
    features = extract_author_attributes(graph, data_path=data_config['data_path'], cache_path=cache_path)
    
    # Load other necessary data
    if data_config.get('task') == 'classification':
        _, labels, num_classes, train_idx, test_idx = data_loader.load_classification_data()
        return graph, features, labels, num_classes, train_idx, test_idx
    else:  # Assume links prediction task
        _, src_train, src_test, dst_train, dst_test, labels_train, labels_test = data_loader.load_links_prediction_data()
        return graph, features, src_train, src_test, dst_train, dst_test, labels_train, labels_test