#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/12/19
# @Author  : AE-CP-GNN Extension
# @File    : attribute_encoder.py
# @Describe: DistilBERT-based attribute encoder for node text attributes

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
import logging

# Suppress warnings from transformers
logging.getLogger("transformers").setLevel(logging.ERROR)


class AttributeEncoder(nn.Module):
    """
    DistilBERT-based encoder for node text attributes.
    Converts text attributes into semantic feature vectors.
    """
    
    def __init__(self, output_dim=128, max_length=128, freeze_bert=True):
        super(AttributeEncoder, self).__init__()
        
        self.output_dim = output_dim
        self.max_length = max_length
        
        # Initialize DistilBERT
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        # Freeze DistilBERT parameters if specified
        if freeze_bert:
            for param in self.distilbert.parameters():
                param.requires_grad = False
        
        # MLP layer to project DistilBERT output to desired dimension
        self.projection = nn.Sequential(
            nn.Linear(self.distilbert.config.hidden_size, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim)
        )
        
    def encode_texts(self, texts):
        """
        Encode a list of texts into feature vectors.
        
        Args:
            texts: List of strings to encode
            
        Returns:
            torch.Tensor: Encoded features of shape [len(texts), output_dim]
        """
        if not texts:
            return torch.zeros(0, self.output_dim)
        
        # Handle None or empty texts
        processed_texts = []
        for text in texts:
            if text is None or text.strip() == "":
                processed_texts.append("[UNK]")  # Use unknown token for empty text
            else:
                processed_texts.append(str(text).strip())
        
        # Tokenize texts
        encoded = self.tokenizer(
            processed_texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move to same device as model
        device = next(self.parameters()).device
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # Get DistilBERT outputs
        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.distilbert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Use [CLS] token representation (first token)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        
        # Project to desired dimension
        features = self.projection(cls_embeddings)
        
        return features
    
    def forward(self, texts):
        """Forward pass for training."""
        return self.encode_texts(texts)


class NodeAttributeManager:
    """
    Manages node attributes for different node types.
    Handles loading and preprocessing of text attributes.
    """
    
    def __init__(self, data_path, dataset_name):
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.node_attributes = {}
        self._load_attributes()
    
    def _load_attributes(self):
        """Load text attributes for different node types."""
        import os
        
        base_path = os.path.join(self.data_path, self.dataset_name)
        
        # Load paper titles
        paper_file = os.path.join(base_path, 'paper.txt')
        if os.path.exists(paper_file):
            self.node_attributes['p'] = self._load_text_file(paper_file)
        
        # Load author names
        author_file = os.path.join(base_path, 'author.txt')
        if os.path.exists(author_file):
            self.node_attributes['a'] = self._load_text_file(author_file)
        
        # Load conference names
        conf_file = os.path.join(base_path, 'conf.txt')
        if os.path.exists(conf_file):
            self.node_attributes['c'] = self._load_text_file(conf_file)
        
        # Load term names
        term_file = os.path.join(base_path, 'term.txt')
        if os.path.exists(term_file):
            self.node_attributes['t'] = self._load_text_file(term_file)
    
    def _load_text_file(self, file_path):
        """
        Load text file with format: id \t text
        
        Returns:
            dict: Mapping from node_id to text
        """
        attributes = {}
        try:
            # Try different encodings
            encodings = ['utf-8', 'gbk', 'latin-1']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                parts = line.split('\t', 1)
                                if len(parts) == 2:
                                    node_id = int(parts[0])
                                    text = parts[1].strip()
                                    attributes[node_id] = text
                    break
                except UnicodeDecodeError:
                    continue
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
        
        return attributes
    
    def get_node_attributes(self, node_type, node_ids):
        """
        Get text attributes for given node IDs of a specific type.
        
        Args:
            node_type: Type of nodes ('p', 'a', 'c', 't')
            node_ids: List of node IDs
            
        Returns:
            list: List of text attributes (None for missing attributes)
        """
        if node_type not in self.node_attributes:
            return [None] * len(node_ids)
        
        attributes = self.node_attributes[node_type]
        return [attributes.get(node_id, None) for node_id in node_ids]
    
    def has_attributes(self, node_type):
        """Check if a node type has text attributes."""
        return node_type in self.node_attributes and len(self.node_attributes[node_type]) > 0


def preprocess_text(text):
    """
    Preprocess text for DistilBERT.
    
    Args:
        text: Input text string
        
    Returns:
        str: Preprocessed text
    """
    if text is None:
        return ""
    
    # Basic cleaning
    text = str(text).strip()
    
    # Remove excessive whitespace
    text = " ".join(text.split())
    
    # Convert to lowercase (DistilBERT is case-insensitive)
    text = text.lower()
    
    return text 