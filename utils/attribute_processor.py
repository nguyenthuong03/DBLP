#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Describe: Node attribute processor for DBLP dataset

import os
import pickle
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import dgl

class NodeAttributeProcessor:
    """Process and prepare node attributes for DBLP dataset"""
    
    def __init__(self, data_config):
        self.data_config = data_config
        self.base_data_path = os.path.join(data_config['data_path'], data_config['dataset'])
        self.primary_type = data_config['primary_type']
        
        # Prepare output path for saving processed attributes
        self.output_path = data_config.get('node_attribute_path', 
                            os.path.join(self.base_data_path, 'node_attributes.pkl'))
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def process_attributes(self, hg=None):
        """Process attributes for different node types
        
        Args:
            hg: Heterogeneous graph object, optional
            
        Returns:
            Dict of node attributes by node type
        """
        # Check if processed attributes already exist
        if os.path.exists(self.output_path) and not self.data_config.get('reprocess_attributes', False):
            self.logger.info(f"Loading preprocessed node attributes from {self.output_path}")
            with open(self.output_path, 'rb') as f:
                node_attributes = pickle.load(f)
            return node_attributes
        
        self.logger.info("Processing node attributes for DBLP dataset")
        
        # Initialize attributes dictionary
        node_attributes = {}
        
        # Process author attributes ('a')
        if self.primary_type == 'a' or self.data_config.get('process_all_types', False):
            author_attrs = self._process_author_attributes(hg)
            if author_attrs is not None:
                node_attributes['a'] = author_attrs
        
        # Process paper attributes ('p')
        if self.primary_type == 'p' or self.data_config.get('process_all_types', False):
            paper_attrs = self._process_paper_attributes(hg)
            if paper_attrs is not None:
                node_attributes['p'] = paper_attrs
        
        # Process conference attributes ('c')
        if self.primary_type == 'c' or self.data_config.get('process_all_types', False):
            conf_attrs = self._process_conf_attributes(hg)
            if conf_attrs is not None:
                node_attributes['c'] = conf_attrs
            
        # Save processed attributes
        if node_attributes:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            with open(self.output_path, 'wb') as f:
                pickle.dump(node_attributes, f)
            self.logger.info(f"Saved processed node attributes to {self.output_path}")
        
        return node_attributes
    
    def _process_author_attributes(self, hg):
        """Process author node attributes
        
        For authors, we can use:
        1. Features already in matrix (a_feature), which might include research interests
        2. Create meta-path based features (e.g., author-paper-term patterns)
        """
        self.logger.info("Processing author attributes")
        
        try:
            # Try to load features from raw_matrix if available
            a_feature_path = os.path.join(self.base_data_path, 'a_feature.pkl')
            
            if os.path.exists(a_feature_path):
                with open(a_feature_path, 'rb') as f:
                    a_feature = pickle.load(f)
                return torch.FloatTensor(a_feature)
            
            # If not available as pickle, try loading from DBLP.mat
            # This assumes we can access the raw_matrix from somewhere else
            # (in a real implementation, you'd pass this in or load it here)
            
            # Alternatively, create meta-path based features
            if hg is not None:
                # Use meta-path APT (Author-Paper-Term) to create author features based on terms
                # Get all author nodes
                author_nodes = hg.nodes('a')
                num_authors = len(author_nodes)
                
                # Create meta-path-based embeddings using graph structure
                # For example: Use terms related to author's papers
                g_apt = dgl.metapath_reachable_graph(hg, ['ap', 'pt'])
                
                # Count terms for each author
                author_term_counts = np.zeros((num_authors, hg.number_of_nodes('t')))
                
                for i, author_id in enumerate(author_nodes):
                    # Find all terms connected to this author through papers
                    neighbors = g_apt.successors(author_id).numpy()
                    if len(neighbors) > 0:
                        for term_id in neighbors:
                            author_term_counts[i, term_id] += 1
                
                # Normalize the counts
                row_sums = author_term_counts.sum(axis=1)
                author_term_dist = author_term_counts / (row_sums[:, np.newaxis] + 1e-8)
                
                return torch.FloatTensor(author_term_dist)
            
            self.logger.warning("Could not process author attributes - no graph or feature file found")
            return None
            
        except Exception as e:
            self.logger.error(f"Error processing author attributes: {e}")
            return None
    
    def _process_paper_attributes(self, hg):
        """Process paper node attributes
        
        For papers, we can use:
        1. Term-based features (paper-term matrix)
        2. Title/abstract text embeddings if available
        """
        self.logger.info("Processing paper attributes")
        
        try:
            # Try to load paper-term matrix which represents paper content
            paper_term_path = os.path.join(self.base_data_path, 'p_vs_t.npz')
            
            if os.path.exists(paper_term_path):
                p_vs_t = sp.load_npz(paper_term_path)
                # Convert to dense if not too large, otherwise keep sparse
                if p_vs_t.shape[1] <= 1000:  # Threshold for dense conversion
                    return torch.FloatTensor(p_vs_t.toarray())
                else:
                    # Convert sparse matrix to PyTorch sparse tensor
                    coo = p_vs_t.tocoo()
                    values = torch.FloatTensor(coo.data)
                    indices = torch.LongTensor(np.vstack((coo.row, coo.col)))
                    shape = torch.Size(coo.shape)
                    return torch.sparse.FloatTensor(indices, values, shape)
            
            # If paper titles are available, create text embeddings
            paper_file = os.path.join(self.base_data_path, 'paper.txt')
            if os.path.exists(paper_file):
                # Read paper titles
                paper_titles = []
                paper_indices = []
                
                with open(paper_file, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        try:
                            parts = line.strip().split('\t')
                            if len(parts) >= 2:
                                idx, title = parts[0], parts[1]
                                paper_indices.append(int(idx))
                                paper_titles.append(title)
                        except Exception as e:
                            self.logger.warning(f"Error parsing line: {line.strip()} - {e}")
                
                if paper_titles:
                    # Preprocess and create TF-IDF features
                    processed_titles = [self._preprocess_text(title) for title in paper_titles]
                    
                    # Create TF-IDF vectors
                    vectorizer = TfidfVectorizer(max_features=1000)
                    tfidf_matrix = vectorizer.fit_transform(processed_titles)
                    
                    # Create a full matrix with correct indices
                    if hg is not None:
                        num_papers = hg.number_of_nodes('p')
                    else:
                        num_papers = max(paper_indices) + 1
                    
                    full_tfidf = np.zeros((num_papers, tfidf_matrix.shape[1]))
                    for i, paper_idx in enumerate(paper_indices):
                        if paper_idx < num_papers:
                            full_tfidf[paper_idx] = tfidf_matrix[i].toarray()
                    
                    return torch.FloatTensor(full_tfidf)
            
            self.logger.warning("Could not process paper attributes - no feature sources found")
            return None
            
        except Exception as e:
            self.logger.error(f"Error processing paper attributes: {e}")
            return None
    
    def _process_conf_attributes(self, hg):
        """Process conference node attributes
        
        For conferences, we can use:
        1. One-hot encoding
        2. Embedding based on papers published in the conference
        """
        self.logger.info("Processing conference attributes")
        
        try:
            conf_file = os.path.join(self.base_data_path, 'conf.txt')
            if os.path.exists(conf_file):
                # Read conference names
                conf_names = []
                conf_indices = []
                
                with open(conf_file, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        try:
                            parts = line.strip().split('\t')
                            if len(parts) >= 2:
                                idx, name = parts[0], parts[1]
                                conf_indices.append(int(idx))
                                conf_names.append(name)
                        except Exception as e:
                            self.logger.warning(f"Error parsing line: {line.strip()} - {e}")
                
                if conf_names:
                    # Simple one-hot encoding for conferences
                    if hg is not None:
                        num_confs = hg.number_of_nodes('c')
                    else:
                        num_confs = max(conf_indices) + 1
                    
                    # One-hot encoding
                    one_hot = np.zeros((num_confs, len(conf_names)))
                    for i, conf_idx in enumerate(conf_indices):
                        if conf_idx < num_confs:
                            one_hot[conf_idx, i] = 1
                    
                    return torch.FloatTensor(one_hot)
            
            # If we have the graph, we can create features based on paper-term distributions
            if hg is not None:
                # Use CTP (Conference-Paper-Term) meta-path
                g_cpt = dgl.metapath_reachable_graph(hg, ['cp', 'pt'])
                
                # Count terms for each conference
                num_confs = hg.number_of_nodes('c')
                conf_term_counts = np.zeros((num_confs, hg.number_of_nodes('t')))
                
                for conf_id in range(num_confs):
                    # Find all terms connected to this conference through papers
                    neighbors = g_cpt.successors(conf_id).numpy()
                    if len(neighbors) > 0:
                        for term_id in neighbors:
                            conf_term_counts[conf_id, term_id] += 1
                
                # Normalize the counts
                row_sums = conf_term_counts.sum(axis=1)
                conf_term_dist = conf_term_counts / (row_sums[:, np.newaxis] + 1e-8)
                
                return torch.FloatTensor(conf_term_dist)
            
            self.logger.warning("Could not process conference attributes - no graph or conference file found")
            return None
            
        except Exception as e:
            self.logger.error(f"Error processing conference attributes: {e}")
            return None
    
    def _preprocess_text(self, text):
        """Preprocess text for better feature extraction
        
        Args:
            text: Input text string
            
        Returns:
            Preprocessed text string
        """
        try:
            # Download required NLTK resources if not present
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords', quiet=True)
            
            # Initialize stemmer
            stemmer = PorterStemmer()
            
            # Convert to lowercase
            text = text.lower()
            
            # Remove special characters and digits
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            
            # Tokenize
            tokens = text.split()
            
            # Remove stopwords and stem
            stop_words = set(stopwords.words('english'))
            stemmed_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
            
            # Rejoin tokens
            return ' '.join(stemmed_tokens)
        
        except Exception as e:
            self.logger.warning(f"Error in text preprocessing: {e}, returning original text")
            return text


def load_node_attributes(data_config, hg=None):
    """Helper function to load node attributes
    
    Args:
        data_config: Data configuration dictionary
        hg: Heterogeneous graph object, optional
        
    Returns:
        Dictionary of node attributes or None if not available
    """
    processor = NodeAttributeProcessor(data_config)
    return processor.process_attributes(hg) 