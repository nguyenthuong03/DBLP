#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/12/19
# @Author  : Dynamic CP-GNN Implementation
# @File    : dynamic_preprocess.py
# @Software: PyCharm
# @Describe: Dynamic graph preprocessing for temporal snapshots

import os
import pickle
import numpy as np
import dgl
import torch
from scipy import io as sio
from .preprocess import GraphDataLoader, EdgesDataset
from sklearn.model_selection import train_test_split


class DynamicDBLPDataLoader(GraphDataLoader):
    """
    Dynamic DBLP Data Loader for temporal graph snapshots
    """
    def __init__(self, data_config, remove_self_loop=False, num_snapshots=5):
        super(DynamicDBLPDataLoader, self).__init__(data_config, remove_self_loop)
        self.num_snapshots = num_snapshots
        self.snapshots = []
        self.snapshot_years = []
        self.heter_graph, self.raw_matrix = self.load_raw_matrix()
        self.create_temporal_snapshots()

    def load_raw_matrix(self):
        """Load the complete DBLP dataset"""
        data = sio.loadmat(self.data_path)
        
        p_vs_a = data['p_vs_a']  # paper-author
        p_vs_c = data['p_vs_c']  # paper-conference
        p_vs_t = data['p_vs_t']  # paper-term

        # Extract indices for heterogeneous graph
        src_p_a, dst_a = p_vs_a.nonzero()
        src_a_p, dst_p_a = p_vs_a.transpose().nonzero()

        src_p_c, dst_c = p_vs_c.nonzero()
        src_c_p, dst_p_c = p_vs_c.transpose().nonzero()

        src_p_t, dst_t = p_vs_t.nonzero()
        src_t_p, dst_p_t = p_vs_t.transpose().nonzero()

        # Create heterogeneous graph
        graph_data = {
            ('p', 'pa', 'a'): (src_p_a, dst_a),
            ('a', 'ap', 'p'): (src_a_p, dst_p_a),
            ('p', 'pc', 'c'): (src_p_c, dst_c),
            ('c', 'cp', 'p'): (src_c_p, dst_p_c),
            ('p', 'pt', 't'): (src_p_t, dst_t),
            ('t', 'tp', 'p'): (src_t_p, dst_p_t),
        }

        hg = dgl.heterograph(graph_data)
        return hg, data

    def create_temporal_snapshots(self):
        """
        Create temporal snapshots based on simulated publication years
        Since DBLP.mat doesn't contain temporal information, we simulate it
        """
        print(f"Creating {self.num_snapshots} temporal snapshots...")
        
        # Get total number of papers
        num_papers = self.heter_graph.number_of_nodes('p')
        
        # Simulate temporal information by dividing papers into time periods
        # Each snapshot will be cumulative (includes all previous papers)
        papers_per_snapshot = num_papers // self.num_snapshots
        
        for t in range(self.num_snapshots):
            # Cumulative approach: snapshot t includes papers 0 to (t+1)*papers_per_snapshot
            if t == self.num_snapshots - 1:
                # Last snapshot includes all remaining papers
                end_paper_idx = num_papers
            else:
                end_paper_idx = (t + 1) * papers_per_snapshot
            
            # Create subgraph for this snapshot
            snapshot_graph = self.create_snapshot_subgraph(end_paper_idx)
            self.snapshots.append(snapshot_graph)
            self.snapshot_years.append(2010 + t)  # Simulate years 2010-2014
            
            print(f"Snapshot {t+1}: Papers 0-{end_paper_idx-1}, Year {2010+t}")

    def create_snapshot_subgraph(self, max_paper_idx):
        """
        Create a subgraph containing papers up to max_paper_idx and their connections
        """
        # Get paper indices for this snapshot
        paper_nodes = torch.arange(max_paper_idx)
        
        # Find connected authors, conferences, and terms
        p_vs_a = self.raw_matrix['p_vs_a'][:max_paper_idx, :]
        p_vs_c = self.raw_matrix['p_vs_c'][:max_paper_idx, :]
        p_vs_t = self.raw_matrix['p_vs_t'][:max_paper_idx, :]
        
        # Get active authors, conferences, and terms
        active_authors = np.unique(p_vs_a.nonzero()[1])
        active_confs = np.unique(p_vs_c.nonzero()[1])
        active_terms = np.unique(p_vs_t.nonzero()[1])
        
        # Create mappings for the subgraph
        author_mapping = {old_id: new_id for new_id, old_id in enumerate(active_authors)}
        conf_mapping = {old_id: new_id for new_id, old_id in enumerate(active_confs)}
        term_mapping = {old_id: new_id for new_id, old_id in enumerate(active_terms)}
        
        # Extract edges for subgraph
        src_p_a, dst_a = p_vs_a.nonzero()
        src_p_c, dst_c = p_vs_c.nonzero()
        src_p_t, dst_t = p_vs_t.nonzero()
        
        # Remap destination indices
        dst_a_remapped = [author_mapping[old_id] for old_id in dst_a]
        dst_c_remapped = [conf_mapping[old_id] for old_id in dst_c]
        dst_t_remapped = [term_mapping[old_id] for old_id in dst_t]
        
        # Create reverse edges
        src_a_p = dst_a_remapped
        dst_p_a = src_p_a
        src_c_p = dst_c_remapped
        dst_p_c = src_p_c
        src_t_p = dst_t_remapped
        dst_p_t = src_p_t
        
        # Create snapshot graph data
        snapshot_graph_data = {
            ('p', 'pa', 'a'): (src_p_a, dst_a_remapped),
            ('a', 'ap', 'p'): (src_a_p, dst_p_a),
            ('p', 'pc', 'c'): (src_p_c, dst_c_remapped),
            ('c', 'cp', 'p'): (src_c_p, dst_p_c),
            ('p', 'pt', 't'): (src_p_t, dst_t_remapped),
            ('t', 'tp', 'p'): (src_t_p, dst_p_t),
        }
        
        snapshot_hg = dgl.heterograph(snapshot_graph_data)
        
        # Store mapping information for later use
        snapshot_hg.paper_mapping = torch.arange(max_paper_idx)
        snapshot_hg.author_mapping = torch.tensor(active_authors)
        snapshot_hg.conf_mapping = torch.tensor(active_confs)
        snapshot_hg.term_mapping = torch.tensor(active_terms)
        
        return snapshot_hg

    def get_snapshot(self, t):
        """Get the t-th snapshot"""
        if t >= len(self.snapshots):
            raise IndexError(f"Snapshot {t} does not exist. Only {len(self.snapshots)} snapshots available.")
        return self.snapshots[t]

    def get_all_snapshots(self):
        """Get all snapshots"""
        return self.snapshots

    def load_classification_data(self):
        """Load classification data for the final snapshot (most recent)"""
        # Use the final snapshot for evaluation
        final_snapshot = self.snapshots[-1]
        
        task_path = os.path.join(self.data_config['data_path'], self.data_config['dataset'], 'CF')
        if not os.path.exists(task_path):
            os.mkdir(task_path)
        
        node_type = 'author'
        if self.data_config['primary_type'] == 'a':
            node_type = "author"
        elif self.data_config['primary_type'] == 'p':
            node_type = "paper"
        elif self.data_config['primary_type'] == 'c':
            node_type = "conf"
        
        task_data_path = os.path.join(task_path, f'dynamic_data_test_{node_type}_{self.data_config["test_ratio"]}.pkl')
        
        if self.data_config['resample'] or not os.path.exists(task_data_path):
            data_path = os.path.dirname(self.data_path)
            author_idx_map_path = os.path.join(data_path, f"{node_type}.txt")
            author_idx_map = {}
            
            if node_type == "paper":
                f = open(author_idx_map_path, encoding="gbk")
            else:
                f = open(author_idx_map_path)
            
            for i, l in enumerate(f.readlines()):
                l = l.replace("\n", "")
                idx, item = l.split("\t")
                if item not in author_idx_map:
                    author_idx_map[int(idx)] = i
            f.close()
            
            author_label_path = os.path.join(data_path, f"{node_type}_label.txt")
            num_classes = 4
            data = []
            label_dict = {}
            
            with open(author_label_path) as f:
                for l in f.readlines():
                    l = l.replace("\n", "").strip("\t")
                    author, label, name = l.split("\t")
                    # Map to final snapshot's author indices
                    original_author_id = int(author)
                    if original_author_id in final_snapshot.author_mapping:
                        # Find the new index in the final snapshot
                        new_author_id = torch.where(final_snapshot.author_mapping == original_author_id)[0]
                        if len(new_author_id) > 0:
                            new_author_id = new_author_id[0].item()
                            data.append(new_author_id)
                            label_dict[new_author_id] = int(label)
            
            data = np.array(data, dtype=np.int32)
            labels = np.full(final_snapshot.number_of_nodes('a'), -1)
            for idx, label in label_dict.items():
                labels[idx] = label
            
            labels = np.array(labels)
            train_idx, test_idx = train_test_split(data, test_size=self.data_config['test_ratio'],
                                                   random_state=self.data_config['random_seed'])
            
            # Use author features from the original dataset
            features = self.raw_matrix['a_feature'].toarray()
            # Map features to final snapshot authors
            final_features = features[final_snapshot.author_mapping.numpy()]
            
            with open(task_data_path, 'wb') as f:
                pickle.dump([final_features, labels, num_classes, train_idx, test_idx], f)
        else:
            with open(task_data_path, 'rb') as f:
                features, labels, num_classes, train_idx, test_idx = pickle.load(f)
        
        return features, labels, num_classes, train_idx, test_idx

    def load_train_k_context_edges_for_snapshot(self, snapshot_idx, K, primary_type, pos_num_for_each_hop, neg_num_for_each_hop):
        """Load training edges for a specific snapshot"""
        snapshot = self.snapshots[snapshot_idx]
        edges_data_dict = {}
        
        for k in range(1, K + 2):
            k_hop_primary_graph = self._load_k_hop_graph_for_snapshot(snapshot, k, primary_type, snapshot_idx)
            k_hop_edge = EdgesDataset(k_hop_primary_graph, pos_num_for_each_hop[k], neg_num_for_each_hop[k])
            edges_data_dict[k] = k_hop_edge
        
        return edges_data_dict

    def _load_k_hop_graph_for_snapshot(self, snapshot, k, primary_type, snapshot_idx):
        """Create k-hop graph for a specific snapshot"""
        print(f'Process: {k} hop graph for snapshot {snapshot_idx}')
        
        k_hop_graph_path = os.path.join(self.k_hop_graph_path,
                                        f'{primary_type}_{k}_hop_graph_snapshot_{snapshot_idx}.pkl')
        
        if not os.path.exists(k_hop_graph_path):
            ntype = snapshot.ntypes
            primary_type_id = ntype.index(primary_type)
            homo_g = dgl.to_homogeneous(snapshot)
            p_nodes_id = homo_g.filter_nodes(
                lambda nodes: (nodes.data['_TYPE'] == primary_type_id))
            
            min_p = torch.min(p_nodes_id).item()
            max_p = torch.max(p_nodes_id).item()
            raw_adj = homo_g.adjacency_matrix()
            raw_adj = raw_adj.to_dense().float()
            adj_k = torch.matrix_power(raw_adj, k)
            p_adj = adj_k[min_p:max_p+1, min_p:max_p+1].cpu()
            row, col = torch.nonzero(p_adj, as_tuple=True)
            p_g = dgl.graph((row, col))
            
            with open(k_hop_graph_path, 'wb') as f:
                pickle.dump(p_g, f, protocol=4)
        else:
            with open(k_hop_graph_path, 'rb') as f:
                p_g = pickle.load(f)
        
        return p_g


def load_dynamic_data(data_config, remove_self_loop=False, num_snapshots=5):
    """
    Load dynamic data with temporal snapshots
    """
    dataset = data_config['dataset']
    if dataset == 'DBLP':
        return DynamicDBLPDataLoader(data_config, remove_self_loop, num_snapshots)
    else:
        raise NotImplementedError(f'Dynamic version not implemented for dataset {dataset}') 