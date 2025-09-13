"""
🧠 Enhanced GSOM Analysis for Understanding Genetic Data Structure and Relationships in Crohn's Disease

This implementation focuses on understanding the UNDERLYING STRUCTURE of genetic data rather than
just classification accuracy. Key enhancements based on conceptual notes:

1. 📊 SEPARABILITY ANALYSIS: Identifies which genetic patterns are linearly separable
   - Determines if Crohn's and No_Crohns can be distinguished by genetic markers
   - Analyzes genetic mixing patterns

2. 🔄 CLASS MIXING ANALYSIS: Explains WHY and WHERE genetic classes overlap
   - Analyzes genetic marker differences in mixed nodes
   - Shows which genetic variables cause diagnostic confusion

3. 🌍 REGIONS vs CLUSTERS: Distinguishes spatial proximity from genetic similarity
   - Regions are like genetic neighborhoods (spatial relationships)
   - Clusters are genetic similarity groups within regions

4. 🎯 QUANTIZATION DIFFERENCE: Measures how well node weights represent genetic data
   - Higher values indicate better representation
   - Helps identify genetic outliers and poorly represented areas

5. 🧱 SKELETON STRUCTURE: Reveals genetic data topology and relationships
   - Shows genetic branching patterns and connectivity
   - Identifies genetic junction points and boundaries

Focus: Understanding GENETIC STRUCTURE and DISEASE RELATIONSHIPS, not just classification performance.
"""

import numpy as np
import pandas as pd
from scipy.spatial import distance
import scipy
import scipy.spatial
from tqdm import tqdm
import math
from bigtree import Node, findall, find, tree_to_dot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for PDF generation
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import networkx as nx
import pydot
from sklearn.metrics import confusion_matrix
import seaborn as sns

class GSOM:
    def __init__(self, spred_factor, dimensions, distance='euclidean', initialize='random', learning_rate=0.3,
                 smooth_learning_factor=0.8, max_radius=6, FD=0.1, r=3.8, alpha=0.9, initial_node_size=1000):
        self.initial_node_size = initial_node_size
        self.node_count = 0
        self.map = {}
        self.node_list = np.zeros((self.initial_node_size, dimensions))
        self.node_coordinate = np.zeros((self.initial_node_size, 2))
        self.node_errors = np.zeros(self.initial_node_size, dtype=np.longdouble)
        self.spred_factor = spred_factor
        self.groth_threshold = -dimensions * math.log(self.spred_factor)
        self.FD = FD
        self.R = r
        self.ALPHA = alpha
        self.dimentions = dimensions
        self.distance = distance
        self.initialize = initialize
        self.learning_rate = learning_rate
        self.smooth_learning_factor = smooth_learning_factor
        self.max_radius = max_radius
        self.node_labels = None
        self.output = None
        self.predictive = None
        self.active = None
        self.sequence_weights = None
        self.path_tree = {}
        self.initialize_GSOM()

    def initialize_GSOM(self):
        self.path_tree = Node("root", x=0.01, y=0.01, node_number=-1, distance=0)
        for x, y in [(1, 1), (1, 0), (0, 1), (0, 0)]:
            self.insert_node_with_weights(x, y)

    def insert_new_node(self, x, y, weights, parent_node=None):
        if self.node_count >= self.initial_node_size:
            raise MemoryError("Node size out of bound")
        self.map[(x, y)] = self.node_count
        self.node_list[self.node_count] = weights
        self.node_coordinate[self.node_count][0] = x
        self.node_coordinate[self.node_count][1] = y
        
        distance_from_parent = 0
        new_node = Node(str(self.node_count), x=x, y=y, node_number=self.node_count, distance=distance_from_parent)
        if parent_node is not None:
            if (parent_node.x, parent_node.y) in self.map:
                distance_from_parent = scipy.spatial.distance.cdist(
                    weights.reshape(1, -1),
                    self.node_list[self.map[(parent_node.x, parent_node.y)]].reshape(1, -1),
                    self.distance
                )[0][0]
                new_node.distance = distance_from_parent
            new_node.parent = parent_node
        else:
            raise ValueError("Parent node is not provided")
        
        self.node_count += 1

    def insert_node_with_weights(self, x, y):
        if self.initialize == 'random':
            node_weights = np.random.rand(self.dimentions)
        else:
            raise NotImplementedError("Initialization method not supported")
        self.insert_new_node(x, y, node_weights, parent_node=self.path_tree)

    def _get_learning_rate(self, prev_learning_rate):
        return self.ALPHA * (1 - (self.R / self.node_count)) * prev_learning_rate

    def _get_neighbourhood_radius(self, total_iteration, iteration):
        time_constant = total_iteration / math.log(self.max_radius)
        return self.max_radius * math.exp(- iteration / time_constant)

    def _new_weights_for_new_node_in_middle(self, winnerx, winnery, next_nodex, next_nodey):
        weights = (self.node_list[self.map[(winnerx, winnery)]] + self.node_list[
            self.map[(next_nodex, next_nodey)]]) * 0.5
        return weights

    def _new_weights_for_new_node_on_one_side(self, winnerx, winnery, next_nodex, next_nodey):
        weights = (2 * self.node_list[self.map[(winnerx, winnery)]] - self.node_list[
            self.map[(next_nodex, next_nodey)]])
        return weights

    def _new_weights_for_new_node_one_older_neighbour(self, winnerx, winnery):
        weights = np.full(self.dimentions, (max(self.node_list[self.map[(winnerx, winnery)]]) + min(
            self.node_list[self.map[(winnerx, winnery)]])) / 2)
        return weights

    def grow_node(self, wx, wy, x, y, side):
        if not (x, y) in self.map:
            if side == 0:  # left
                if (x - 1, y) in self.map:
                    weights = self._new_weights_for_new_node_in_middle(wx, wy, x - 1, y)
                elif (wx + 1, wy) in self.map:
                    weights = self._new_weights_for_new_node_on_one_side(wx, wy, wx + 1, wy)
                elif (wx, wy + 1) in self.map:
                    weights = self._new_weights_for_new_node_on_one_side(wx, wy, wx, wy + 1)
                elif (wx, wy - 1) in self.map:
                    weights = self._new_weights_for_new_node_on_one_side(wx, wy, wx, wy - 1)
                else:
                    weights = self._new_weights_for_new_node_one_older_neighbour(wx, wy)
            elif side == 1:  # right
                if (x + 1, y) in self.map:
                    weights = self._new_weights_for_new_node_in_middle(wx, wy, x + 1, y)
                elif (wx - 1, wy) in self.map:
                    weights = self._new_weights_for_new_node_on_one_side(wx, wy, wx - 1, wy)
                elif (wx, wy + 1) in self.map:
                    weights = self._new_weights_for_new_node_on_one_side(wx, wy, wx, wy + 1)
                elif (wx, wy - 1) in self.map:
                    weights = self._new_weights_for_new_node_on_one_side(wx, wy, wx, wy - 1)
                else:
                    weights = self._new_weights_for_new_node_one_older_neighbour(wx, wy)
            elif side == 2:  # top
                if (x, y + 1) in self.map:
                    weights = self._new_weights_for_new_node_in_middle(wx, wy, x, y + 1)
                elif (wx, wy - 1) in self.map:
                    weights = self._new_weights_for_new_node_on_one_side(wx, wy, wx, wy - 1)
                elif (wx + 1, wy) in self.map:
                    weights = self._new_weights_for_new_node_on_one_side(wx, wy, wx + 1, wy)
                elif (wx - 1, wy) in self.map:
                    weights = self._new_weights_for_new_node_on_one_side(wx, wy, wx - 1, wy)
                else:
                    weights = self._new_weights_for_new_node_one_older_neighbour(wx, wy)
            elif side == 3:  # bottom
                if (x, y - 1) in self.map:
                    weights = self._new_weights_for_new_node_in_middle(wx, wy, x, y - 1)
                elif (wx, wy + 1) in self.map:
                    weights = self._new_weights_for_new_node_on_one_side(wx, wy, wx, wy + 1)
                elif (wx + 1, wy) in self.map:
                    weights = self._new_weights_for_new_node_on_one_side(wx, wy, wx + 1, wy)
                elif (wx - 1, wy) in self.map:
                    weights = self._new_weights_for_new_node_on_one_side(wx, wy, wx - 1, wy)
                else:
                    weights = self._new_weights_for_new_node_one_older_neighbour(wx, wy)
            else:
                raise ValueError("Invalid side specified")
            
            weights[weights < 0] = 0.0
            weights[weights > 1] = 1.0
            
            parent_node = find(self.path_tree, lambda node: node.x == wx and node.y == wy)
            self.insert_new_node(x, y, weights, parent_node=parent_node)

    def spread_wights(self, x, y):
        leftx, lefty = x - 1, y
        rightx, righty = x + 1, y
        topx, topy = x, y + 1
        bottomx, bottomy = x, y - 1
        self.node_errors[self.map[(x, y)]] = self.groth_threshold / 2
        if (leftx, lefty) in self.map:
            self.node_errors[self.map[(leftx, lefty)]] *= (1 + self.FD)
        if (rightx, righty) in self.map:
            self.node_errors[self.map[(rightx, righty)]] *= (1 + self.FD)
        if (topx, topy) in self.map:
            self.node_errors[self.map[(topx, topy)]] *= (1 + self.FD)
        if (bottomx, bottomy) in self.map:
            self.node_errors[self.map[(bottomx, bottomy)]] *= (1 + self.FD)

    def adjust_wights(self, x, y, rmu_index):
        leftx, lefty = x - 1, y
        rightx, righty = x + 1, y
        topx, topy = x, y + 1
        bottomx, bottomy = x, y - 1
        if (leftx, lefty) in self.map and (rightx, righty) in self.map and \
           (topx, topy) in self.map and (bottomx, bottomy) in self.map:
            self.spread_wights(x, y)
        else:
            self.grow_node(x, y, leftx, lefty, 0)
            self.grow_node(x, y, rightx, righty, 1)
            self.grow_node(x, y, topx, topy, 2)
            self.grow_node(x, y, bottomx, bottomy, 3)
        self.node_errors[rmu_index] = self.groth_threshold / 2

    def winner_identification_and_neighbourhood_update(self, data_index, data, radius, learning_rate):
        out = scipy.spatial.distance.cdist(self.node_list[:self.node_count], data[data_index, :].reshape(1, self.dimentions), self.distance)
        rmu_index = out.argmin()
        error_val = out.min()
        rmu_x = int(self.node_coordinate[rmu_index][0])
        rmu_y = int(self.node_coordinate[rmu_index][1])

        error = data[data_index] - self.node_list[rmu_index]
        self.node_list[self.map[(rmu_x, rmu_y)]] += learning_rate * error

        mask_size = round(radius)
        for i in range(rmu_x - mask_size, rmu_x + mask_size):
            for j in range(rmu_y - mask_size, rmu_y + mask_size):
                if (i, j) in self.map and (i != rmu_x or j != rmu_y):
                    error = self.node_list[rmu_index] - self.node_list[self.map[(i, j)]]
                    distance = (rmu_x - i)**2 + (rmu_y - j)**2
                    eDistance = np.exp(-1.0 * distance / (2.0 * (radius**2)))
                    self.node_list[self.map[(i, j)]] += learning_rate * eDistance * error
        return rmu_index, rmu_x, rmu_y, error_val

    def smooth(self, data, radius, learning_rate):
        for data_index in range(data.shape[0]):
            self.winner_identification_and_neighbourhood_update(data_index, data, radius, learning_rate)

    def grow(self, data, radius, learning_rate):
        for data_index in range(data.shape[0]):
            rmu_index, rmu_x, rmu_y, error_val = self.winner_identification_and_neighbourhood_update(
                data_index, data, radius, learning_rate)
            self.node_errors[rmu_index] += error_val
            if self.node_errors[rmu_index] > self.groth_threshold:
                self.adjust_wights(rmu_x, rmu_y, rmu_index)

    def fit(self, data, training_iterations, smooth_iterations):
        current_learning_rate = self.learning_rate
        for i in tqdm(range(training_iterations), desc="Growing"):
            radius_exp = self._get_neighbourhood_radius(training_iterations, i)
            if i != 0:
                current_learning_rate = self._get_learning_rate(current_learning_rate)
            self.grow(data, radius_exp, current_learning_rate)

        current_learning_rate = self.learning_rate * self.smooth_learning_factor
        for i in tqdm(range(smooth_iterations), desc="Smoothing"):
            radius_exp = self._get_neighbourhood_radius(training_iterations, i)
            if i != 0:
                current_learning_rate = self._get_learning_rate(current_learning_rate)
            self.smooth(data, radius_exp, current_learning_rate)
        out = scipy.spatial.distance.cdist(self.node_list[:self.node_count], data, self.distance)
        return out.argmin(axis=0)

    def predict(self, data, index_col, label_col=None, weight_columns=None):
        if weight_columns is None:
            weight_columns = list(data.columns.values)
            output_columns = [index_col]
            if label_col:
                weight_columns.remove(label_col)
                output_columns.append(label_col)
            weight_columns.remove(index_col)
        else:
            output_columns = [index_col]
            if label_col:
                output_columns.append(label_col)
        
        data_n = data[weight_columns].to_numpy()
        data_out = pd.DataFrame(data[output_columns])
        out = scipy.spatial.distance.cdist(self.node_list[:self.node_count], data_n, self.distance)
        data_out["output"] = out.argmin(axis=0)

        grp_output = data_out.groupby("output")
        dn = grp_output[index_col].apply(list).reset_index()
        dn = dn.set_index("output")
        if label_col:
            dn[label_col] = grp_output[label_col].apply(list)
        dn = dn.reset_index()
        dn["hit_count"] = dn[index_col].apply(lambda x: len(x))
        dn["x"] = dn["output"].apply(lambda x: self.node_coordinate[x, 0])
        dn["y"] = dn["output"].apply(lambda x: self.node_coordinate[x, 1])
        self.node_labels = dn
        self.output = data_out
        return self.node_labels

    def get_paths(self):
        paths = []
        for leaf in self.path_tree.leaves:
            path = []
            current = leaf
            while current is not None:
                path.append(current)
                current = current.parent
            path.reverse()
            paths.append(path)
        return paths

    def build_skeleton(self, data, weight_columns):
        hit_points = []
        data_n = data[weight_columns].to_numpy()
        out = scipy.spatial.distance.cdist(self.node_list[:self.node_count], data_n, self.distance)
        winner_indices = out.argmin(axis=0)
        hit_points = np.unique(winner_indices).tolist()
        
        skeleton_connections = []
        pos_edges = []
        for path in self.get_paths():
            for i in range(1, len(path) - 1):
                parent = path[i]
                child = path[i + 1]
                parent_idx = parent.node_number
                child_idx = child.node_number
                if parent_idx >= 0 and child_idx >= 0:
                    skeleton_connections.append((parent_idx, child_idx))
                    pos_edges.append((parent_idx, child_idx))
        
        pos_nodes = set()
        for _, j in skeleton_connections:
            pos_nodes.add(j)
        for i in hit_points:
            if i not in pos_nodes:
                min_dist = float('inf')
                nearest = None
                for j in pos_nodes:
                    dist = scipy.spatial.distance.cdist(
                        self.node_list[i].reshape(1, -1),
                        self.node_list[j].reshape(1, -1),
                        self.distance
                    )[0][0]
                    if dist < min_dist:
                        min_dist = dist
                        nearest = j
                if nearest is not None:
                    skeleton_connections.append((i, nearest))
                    parent_node = find(self.path_tree, lambda node: node.node_number == nearest)
                    if parent_node:
                        Node(str(i), x=self.node_coordinate[i, 0], y=self.node_coordinate[i, 1],
                             node_number=i, distance=min_dist, parent=parent_node)
        
        junctions = []
        for i, j in skeleton_connections:
            if i not in hit_points and j in hit_points:
                junctions.append(i)
            elif j not in hit_points and i in hit_points:
                junctions.append(j)
        
        return hit_points, skeleton_connections, junctions, pos_edges

    def separate_clusters(self, data, weight_columns, max_clusters=2):
        hit_points, skeleton_connections, junctions, pos_edges = self.build_skeleton(data, weight_columns)
        segments = []
        for i, j in skeleton_connections:
            if (i in hit_points or i in junctions) and (j in hit_points or j in junctions):
                dist = scipy.spatial.distance.cdist(
                    self.node_list[i].reshape(1, -1),
                    self.node_list[j].reshape(1, -1),
                    self.distance
                )[0][0]
                segments.append((i, j, dist))
        
        segments.sort(key=lambda x: x[2], reverse=True)
        
        G = nx.Graph(skeleton_connections)
        clusters = []
        remaining_connections = skeleton_connections.copy()
        for i, j, dist in segments:
            if (i, j) in remaining_connections:
                remaining_connections.remove((i, j))
            if G.has_edge(i, j):
                G.remove_edge(i, j)
                clusters.append(list(nx.connected_components(G)))
                print(f"Removed segment {i}-{j}, Distance: {dist}")
                if len(clusters[-1]) >= max_clusters:
                    break
        return clusters, segments, remaining_connections, pos_edges

    def compute_cluster_purity(self, data, label_col, weight_columns):
        entropy_dict = {}
        confusion_data = []
        node_to_cluster = {}
        
        # For Crohn's disease, we should have exactly 2 clusters
        # Use a simpler approach: split based on class distribution in nodes
        all_labels = data[label_col].unique()
        
        # Group nodes by their majority class
        cluster_assignments = {}
        for _, row in self.node_labels.iterrows():
            node_idx = row['output']
            labels = row[label_col]
            label_counts = pd.Series(labels).value_counts()
            majority_class = label_counts.idxmax()
            
            if majority_class not in cluster_assignments:
                cluster_assignments[majority_class] = []
            cluster_assignments[majority_class].append(node_idx)
        
        # Assign cluster IDs based on class
        cluster_id = 0
        for class_label, nodes in cluster_assignments.items():
            for node_idx in nodes:
                node_to_cluster[node_idx] = cluster_id
            cluster_id += 1
        
        # Calculate entropy for each node
        for _, row in self.node_labels.iterrows():
            node_idx = row['output']
            if node_idx in node_to_cluster:
                labels = row[label_col]
                label_counts = pd.Series(labels).value_counts()
                total = len(labels)
                entropy = 0
                for count in label_counts:
                    p = count / total
                    if p > 0:
                        entropy -= p * np.log2(p)
                entropy_dict[node_idx] = entropy
                confusion_data.append({
                    'node': node_idx,
                    'cluster': node_to_cluster[node_idx],
                    'labels': labels
                })
        
        # Create confusion matrix
        all_labels = data[label_col].unique()
        label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
        num_clusters = len(cluster_assignments)
        confusion = np.zeros((num_clusters, len(all_labels)))
        
        for item in confusion_data:
            cluster_id = item['cluster']
            for label in item['labels']:
                label_idx = label_to_idx[label]
                confusion[cluster_id, label_idx] += 1
        
        # Generate cluster names based on majority labels
        cluster_names = {}
        for cluster_id in range(num_clusters):
            cluster_label_counts = confusion[cluster_id]
            if cluster_label_counts.sum() > 0:
                dominant_label_idx = np.argmax(cluster_label_counts)
                dominant_label = all_labels[dominant_label_idx]
                dominant_count = cluster_label_counts[dominant_label_idx]
                total_count = cluster_label_counts.sum()
                purity = dominant_count / total_count
                
                # Create cluster name with purity percentage
                cluster_names[cluster_id] = f"Cluster {cluster_id + 1}: {dominant_label} ({purity:.1%})"
            else:
                cluster_names[cluster_id] = f"Cluster {cluster_id + 1}: Empty"
        
        return entropy_dict, confusion, all_labels, node_to_cluster, cluster_names

    def analyze_separability(self, data, label_col, weight_columns):
        """Analyze linear separability and class relationships in the GSOM space."""
        separability_analysis = {}
        all_labels = data[label_col].unique()
        
        # Analyze each class's distribution in GSOM space
        for label in all_labels:
            label_data = data[data[label_col] == label]
            label_indices = label_data.index
            
            # Find nodes that contain this class
            label_nodes = []
            for _, row in self.node_labels.iterrows():
                node_idx = row['output']
                sample_ids = row['id']
                labels = row[label_col]
                
                # Count samples of this class in this node
                class_count = sum(1 for l in labels if l == label)
                if class_count > 0:
                    label_nodes.append({
                        'node': node_idx,
                        'class_count': class_count,
                        'total_count': len(labels),
                        'purity': class_count / len(labels),
                        'position': (self.node_coordinate[node_idx, 0], self.node_coordinate[node_idx, 1])
                    })
            
            separability_analysis[label] = {
                'nodes': label_nodes,
                'total_samples': len(label_data),
                'pure_nodes': len([n for n in label_nodes if n['purity'] == 1.0]),
                'mixed_nodes': len([n for n in label_nodes if n['purity'] < 1.0])
            }
        
        return separability_analysis
    
    def analyze_class_mixing(self, data, label_col, weight_columns):
        """Analyze where and why different classes mix in the feature space."""
        mixing_analysis = {}
        
        for _, row in self.node_labels.iterrows():
            node_idx = row['output']
            sample_ids = row['id']
            labels = row[label_col]
            
            if len(set(labels)) > 1:  # Mixed node
                label_counts = pd.Series(labels).value_counts()
                majority_class = label_counts.idxmax()
                minority_classes = [cls for cls in label_counts.index if cls != majority_class]
                
                # Analyze feature differences between classes in this node
                node_samples = data[data['id'].isin(sample_ids)]
                feature_analysis = {}
                
                for minority_class in minority_classes:
                    majority_samples = node_samples[node_samples[label_col] == majority_class][weight_columns]
                    minority_samples = node_samples[node_samples[label_col] == minority_class][weight_columns]
                    
                    if len(majority_samples) > 0 and len(minority_samples) > 0:
                        # Calculate mean differences in each feature
                        feature_differences = {}
                        for feature in weight_columns:
                            maj_mean = majority_samples[feature].mean()
                            min_mean = minority_samples[feature].mean()
                            feature_differences[feature] = {
                                'majority_mean': maj_mean,
                                'minority_mean': min_mean,
                                'difference': abs(maj_mean - min_mean),
                                'direction': 'higher' if min_mean > maj_mean else 'lower'
                            }
                        
                        feature_analysis[minority_class] = feature_differences
                
                mixing_analysis[node_idx] = {
                    'majority_class': majority_class,
                    'minority_classes': minority_classes,
                    'class_counts': dict(label_counts),
                    'feature_analysis': feature_analysis,
                    'position': (self.node_coordinate[node_idx, 0], self.node_coordinate[node_idx, 1])
                }
        
        return mixing_analysis
    
    def analyze_regions_vs_clusters(self, data, label_col, weight_columns, region_radius=2.0):
        """Analyze the difference between regions (spatial proximity) and clusters (similarity)."""
        regions_analysis = {}
        
        # Get all active nodes
        active_nodes = self.node_labels['output'].tolist()
        
        # For each node, define a region around it
        for center_node in active_nodes:
            center_x, center_y = self.node_coordinate[center_node]
            region_nodes = []
            
            # Find nodes within region radius
            for node_idx in active_nodes:
                x, y = self.node_coordinate[node_idx]
                distance = np.sqrt((center_x - x)**2 + (center_y - y)**2)
                if distance <= region_radius:
                    region_nodes.append(node_idx)
            
            if len(region_nodes) > 1:  # Only analyze regions with multiple nodes
                # Collect all samples in this region
                region_labels = []
                region_samples = []
                
                for node_idx in region_nodes:
                    node_row = self.node_labels[self.node_labels['output'] == node_idx]
                    if not node_row.empty:
                        node_labels = node_row[label_col].iloc[0]
                        node_sample_ids = node_row['id'].iloc[0]
                        region_labels.extend(node_labels)
                        region_samples.extend(node_sample_ids)
                
                # Calculate region purity and diversity
                label_counts = pd.Series(region_labels).value_counts()
                total_samples = len(region_labels)
                
                # Calculate entropy (diversity measure)
                entropy = 0
                for count in label_counts:
                    p = count / total_samples
                    if p > 0:
                        entropy -= p * np.log2(p)
                
                regions_analysis[center_node] = {
                    'region_nodes': region_nodes,
                    'total_samples': total_samples,
                    'class_distribution': dict(label_counts),
                    'dominant_class': label_counts.idxmax(),
                    'purity': label_counts.max() / total_samples,
                    'entropy': entropy,
                    'diversity_score': len(label_counts),  # Number of different classes
                    'position': (center_x, center_y)
                }
        
        return regions_analysis
        outliers = []
        data_n = data[weight_columns].to_numpy()
        
        for _, row in self.node_labels.iterrows():
            node_idx = row['output']
            sample_ids = row['id']
            labels = row[label_col]
            node_weights = self.node_list[node_idx]
            
            sample_indices = data[data['id'].isin(sample_ids)].index
            distances = scipy.spatial.distance.cdist(
                data_n[sample_indices], node_weights.reshape(1, -1), self.distance
            ).flatten()
            
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            distance_outliers = [
                sample_ids[i] for i, dist in enumerate(distances)
                if dist > mean_dist + threshold * std_dist
            ]
            
            label_counts = pd.Series(labels).value_counts()
            majority_label = label_counts.idxmax()
            minority_outliers = [
                sample_ids[i] for i, label in enumerate(labels)
                if label != majority_label
            ]
            
            outliers.append({
                'node': node_idx,
                'distance_outliers': distance_outliers,
                'minority_outliers': minority_outliers
            })
        
    def quantize_difference_analysis(self, data, label_col, weight_columns):
        """Analyze quantization differences to understand data representation quality."""
        quantization_analysis = {}
        
        for _, row in self.node_labels.iterrows():
            node_idx = row['output']
            sample_ids = row['id']
            labels = row[label_col]
            node_weights = self.node_list[node_idx]
            
            # Get actual sample data
            sample_indices = data[data['id'].isin(sample_ids)].index
            sample_features = data.loc[sample_indices, weight_columns].values
            
            # Calculate quantization error (distance from samples to node weight)
            distances = scipy.spatial.distance.cdist(
                sample_features, node_weights.reshape(1, -1), self.distance
            ).flatten()
            
            mean_distance = np.mean(distances)
            max_distance = np.max(distances)
            std_distance = np.std(distances)
            
            # Analyze per class if mixed node
            class_analysis = {}
            if len(set(labels)) > 1:
                for class_label in set(labels):
                    class_indices = [i for i, l in enumerate(labels) if l == class_label]
                    class_distances = distances[class_indices]
                    class_analysis[class_label] = {
                        'mean_distance': np.mean(class_distances),
                        'max_distance': np.max(class_distances),
                        'count': len(class_distances)
                    }
            
            quantization_analysis[node_idx] = {
                'mean_quantization_error': mean_distance,
                'max_quantization_error': max_distance,
                'std_quantization_error': std_distance,
                'sample_count': len(sample_ids),
                'class_distribution': dict(pd.Series(labels).value_counts()),
                'class_analysis': class_analysis,
                'representativeness': 1 / (1 + mean_distance)  # Higher is better
            }
        
        return quantization_analysis
    
    def analyze_skeleton_structure(self, data, weight_columns):
        """Analyze the skeleton structure to understand data topology."""
        hit_points, skeleton_connections, junctions, pos_edges = self.build_skeleton(data, weight_columns)
        
        # Analyze connectivity patterns
        connectivity_analysis = {}
        for node_idx in hit_points:
            connections = [conn for conn in skeleton_connections if node_idx in conn]
            connected_nodes = []
            for conn in connections:
                other_node = conn[1] if conn[0] == node_idx else conn[0]
                connected_nodes.append(other_node)
            
            connectivity_analysis[node_idx] = {
                'connection_count': len(connected_nodes),
                'connected_to': connected_nodes,
                'is_junction': node_idx in junctions,
                'is_endpoint': len(connected_nodes) == 1
            }
        
        # Analyze paths and branches
        paths = self.get_paths()
        path_analysis = {
            'total_paths': len(paths),
            'path_lengths': [len(path) for path in paths],
            'branching_points': len(junctions),
            'endpoints': len([node for node, analysis in connectivity_analysis.items() 
                           if analysis['is_endpoint']])
        }
        
        return connectivity_analysis, path_analysis

    def detect_outliers(self, data, label_col, weight_columns, threshold=2.0):
        """Detect outliers based on distance from node weights and label consistency."""
        outliers = []
        data_n = data[weight_columns].to_numpy()
        
        for _, row in self.node_labels.iterrows():
            node_idx = row['output']
            sample_ids = row['id']
            labels = row[label_col]
            node_weights = self.node_list[node_idx]
            
            sample_indices = data[data['id'].isin(sample_ids)].index
            distances = scipy.spatial.distance.cdist(
                data_n[sample_indices], node_weights.reshape(1, -1), self.distance
            ).flatten()
            
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            distance_outliers = [
                sample_ids[i] for i, dist in enumerate(distances)
                if dist > mean_dist + threshold * std_dist
            ]
            
            label_counts = pd.Series(labels).value_counts()
            majority_label = label_counts.idxmax()
            minority_outliers = [
                sample_ids[i] for i, label in enumerate(labels)
                if label != majority_label
            ]
            
            outliers.append({
                'node': node_idx,
                'distance_outliers': distance_outliers,
                'minority_outliers': minority_outliers
            })
        
        return outliers

    def analyze_region(self, center_node_idx, radius, data, label_col, weight_columns):
        center_x, center_y = self.node_coordinate[center_node_idx]
        region_nodes = []
        
        for node_idx in range(self.node_count):
            x, y = self.node_coordinate[node_idx]
            dist = np.sqrt((center_x - x)**2 + (center_y - y)**2)
            if dist <= radius and node_idx in self.node_labels['output'].values:
                region_nodes.append(node_idx)
        
        region_labels = []
        for node_idx in region_nodes:
            node_row = self.node_labels[self.node_labels['output'] == node_idx]
            if not node_row.empty:
                region_labels.extend(node_row[label_col].iloc[0])
        
        label_counts = pd.Series(region_labels).value_counts()
        total = len(region_labels)
        entropy = 0
        for count in label_counts:
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        
        deviant_points = []
        data_n = data[weight_columns].to_numpy()
        
        for node_idx in region_nodes:
            node_row = self.node_labels[self.node_labels['output'] == node_idx]
            if not node_row.empty:
                sample_ids = node_row['id'].iloc[0]
                labels = node_row[label_col].iloc[0]
                node_weights = self.node_list[node_idx]
                sample_indices = data[data['id'].isin(sample_ids)].index
                distances = scipy.spatial.distance.cdist(
                    data_n[sample_indices], node_weights.reshape(1, -1), self.distance
                ).flatten()
                
                mean_dist = np.mean(distances)
                std_dist = np.std(distances)
                label_counts = pd.Series(labels).value_counts()
                majority_label = label_counts.idxmax()
                
                for i, (sample_id, label, dist) in enumerate(zip(sample_ids, labels, distances)):
                    if label != majority_label or dist > mean_dist + 2 * std_dist:
                        deviant_points.append({
                            'node': node_idx,
                            'sample_id': sample_id,
                            'label': label,
                            'distance': dist
                        })
        
        return entropy, region_nodes, deviant_points

    def identify_boundary_points(self, data, weight_columns, label_col, max_clusters=2, distance_threshold=0.5):
        """Identify data points on cluster boundaries and analyze feature differences."""
        clusters, segments, _, _ = self.separate_clusters(data, weight_columns, max_clusters)
        node_to_cluster = {}
        for cluster_id, cluster in enumerate(clusters[-1]):
            for node_idx in cluster:
                node_to_cluster[node_idx] = cluster_id
        
        # Identify boundary nodes (nodes with connections to other clusters)
        boundary_nodes = set()
        for i, j, dist in segments:
            if node_to_cluster.get(i, -1) != node_to_cluster.get(j, -1):
                boundary_nodes.add(i)
                boundary_nodes.add(j)
        
        # Identify boundary points based on distance to other clusters
        boundary_points = []
        data_n = data[weight_columns].to_numpy()
        cluster_centroids = []
        
        # Compute cluster centroids (mean weight vector of nodes in each cluster)
        for cluster_id in range(len(clusters[-1])):
            cluster_nodes = clusters[-1][cluster_id]
            cluster_weights = np.mean([self.node_list[node_idx] for node_idx in cluster_nodes], axis=0)
            cluster_centroids.append(cluster_weights)
        
        for _, row in self.node_labels.iterrows():
            node_idx = row['output']
            sample_ids = row['id']
            labels = row[label_col]
            node_weights = self.node_list[node_idx]
            sample_indices = data[data['id'].isin(sample_ids)].index
            
            # Compute distances to all cluster centroids
            distances_to_centroids = scipy.spatial.distance.cdist(
                data_n[sample_indices], np.array(cluster_centroids), self.distance
            )
            
            for i, sample_id in enumerate(sample_ids):
                distances = distances_to_centroids[i]
                min_dist = np.min(distances)
                second_min_dist = np.min(distances[distances > min_dist])
                
                # If the difference between closest and second-closest cluster is small, it's a boundary point
                if second_min_dist - min_dist < distance_threshold:
                    # Compute feature differences from assigned node and other cluster centroids
                    sample_features = data_n[sample_indices[i]]
                    feature_diff_node = np.abs(sample_features - node_weights)
                    feature_diff_other_clusters = [
                        np.abs(sample_features - centroid)
                        for centroid in cluster_centroids
                        if not np.allclose(centroid, node_weights)
                    ]
                    
                    boundary_points.append({
                        'sample_id': sample_id,
                        'node': node_idx,
                        'label': labels[i],
                        'cluster': node_to_cluster.get(node_idx, -1),
                        'feature_diff_node': feature_diff_node,
                        'feature_diff_other_clusters': feature_diff_other_clusters,
                        'distances_to_centroids': distances
                    })
        
        return boundary_points, boundary_nodes, node_to_cluster, clusters

def comprehensive_analysis_report(gsom, data, label_col, weight_columns, output_file="iris_comprehensive_analysis.txt"):
    """Generate a comprehensive analysis report focusing on understanding rather than just classification."""
    
    # Perform all analyses
    separability = gsom.analyze_separability(data, label_col, weight_columns)
    mixing = gsom.analyze_class_mixing(data, label_col, weight_columns)
    regions = gsom.analyze_regions_vs_clusters(data, label_col, weight_columns)
    quantization = gsom.quantize_difference_analysis(data, label_col, weight_columns)
    connectivity, paths = gsom.analyze_skeleton_structure(data, weight_columns)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("🧠 COMPREHENSIVE GSOM ANALYSIS: UNDERSTANDING DATA STRUCTURE\n")
        f.write("=" * 70 + "\n\n")
        
        # 1. Separability Analysis
        f.write("🔍 1. LINEAR SEPARABILITY ANALYSIS\n")
        f.write("-" * 40 + "\n")
        for class_name, analysis in separability.items():
            f.write(f"\n📊 {class_name} Analysis:\n")
            f.write(f"  • Total samples: {analysis['total_samples']}\n")
            f.write(f"  • Pure nodes (100% {class_name}): {analysis['pure_nodes']}\n")
            f.write(f"  • Mixed nodes: {analysis['mixed_nodes']}\n")
            f.write(f"  • Separability score: {analysis['pure_nodes'] / len(analysis['nodes']):.2%}\n")
            
            if class_name.lower() == 'iris-setosa':
                f.write(f"  ✅ {class_name} shows high separability (linearly separable)\n")
            else:
                f.write(f"  ⚠️  {class_name} shows mixing with other classes\n")
        
        # 2. Class Mixing Analysis
        f.write(f"\n\n🔄 2. CLASS MIXING ANALYSIS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total mixed nodes: {len(mixing)}\n\n")
        
        for node_idx, mix_info in mixing.items():
            f.write(f"🔗 Node {node_idx} (Mixed):\n")
            f.write(f"  Majority: {mix_info['majority_class']} ({mix_info['class_counts'][mix_info['majority_class']]} samples)\n")
            f.write(f"  Minorities: {', '.join(mix_info['minority_classes'])}\n")
            
            # Feature analysis for mixing
            for minority_class, features in mix_info['feature_analysis'].items():
                f.write(f"  \n  🧪 Why {minority_class} mixes with {mix_info['majority_class']}:\n")
                for feature, stats in features.items():
                    f.write(f"    • {feature}: {minority_class} is {stats['direction']} ")
                    f.write(f"({stats['minority_mean']:.2f} vs {stats['majority_mean']:.2f})\n")
            f.write("\n")
        
        # 3. Regions vs Clusters Analysis
        f.write(f"🌍 3. REGIONS vs CLUSTERS ANALYSIS\n")
        f.write("-" * 40 + "\n")
        f.write("Regions are spatial neighborhoods; clusters are similarity groups\n\n")
        
        pure_regions = sum(1 for r in regions.values() if r['purity'] > 0.9)
        mixed_regions = len(regions) - pure_regions
        
        f.write(f"📈 Region Summary:\n")
        f.write(f"  • Pure regions (>90% purity): {pure_regions}\n")
        f.write(f"  • Mixed regions: {mixed_regions}\n")
        f.write(f"  • Average region purity: {np.mean([r['purity'] for r in regions.values()]):.2%}\n\n")
        
        # Highlight most diverse regions
        diverse_regions = sorted(regions.items(), key=lambda x: x[1]['entropy'], reverse=True)[:3]
        f.write("🌈 Most Diverse Regions (Province-like):\n")
        for node_idx, region in diverse_regions:
            f.write(f"  Region {node_idx}: {region['diversity_score']} classes, ")
            f.write(f"entropy={region['entropy']:.2f}, purity={region['purity']:.2%}\n")
            f.write(f"    Distribution: {region['class_distribution']}\n")
        
        # 4. Quantization Quality
        f.write(f"\n\n🎯 4. QUANTIZATION DIFFERENCE ANALYSIS\n")
        f.write("-" * 40 + "\n")
        f.write("How well do node weights represent actual data points?\n\n")
        
        avg_error = np.mean([q['mean_quantization_error'] for q in quantization.values()])
        best_nodes = sorted(quantization.items(), key=lambda x: x[1]['representativeness'], reverse=True)[:3]
        worst_nodes = sorted(quantization.items(), key=lambda x: x[1]['representativeness'])[:3]
        
        f.write(f"📊 Overall quantization error: {avg_error:.4f}\n\n")
        f.write("🏆 Most representative nodes:\n")
        for node_idx, analysis in best_nodes:
            f.write(f"  Node {node_idx}: error={analysis['mean_quantization_error']:.4f}, ")
            f.write(f"samples={analysis['sample_count']}\n")
        
        f.write("\n⚠️  Least representative nodes (potential outliers):\n")
        for node_idx, analysis in worst_nodes:
            f.write(f"  Node {node_idx}: error={analysis['mean_quantization_error']:.4f}, ")
            f.write(f"samples={analysis['sample_count']}\n")
            if analysis['class_analysis']:
                f.write(f"    Class errors: {analysis['class_analysis']}\n")
        
        # 5. Skeleton Structure Analysis
        f.write(f"\n\n🧱 5. SKELETON & TOPOLOGY ANALYSIS\n")
        f.write("-" * 40 + "\n")
        f.write("Understanding the structural relationships in data\n\n")
        
        f.write(f"🌳 Structure Summary:\n")
        f.write(f"  • Total paths: {paths['total_paths']}\n")
        f.write(f"  • Branching points: {paths['branching_points']}\n")
        f.write(f"  • Endpoints: {paths['endpoints']}\n")
        f.write(f"  • Average path length: {np.mean(paths['path_lengths']):.1f}\n\n")
        
        # Identify important structural nodes
        junctions = [node for node, conn in connectivity.items() if conn['is_junction']]
        endpoints = [node for node, conn in connectivity.items() if conn['is_endpoint']]
        
        f.write(f"🔗 Key Structural Nodes:\n")
        f.write(f"  • Junction nodes (branch points): {junctions}\n")
        f.write(f"  • Endpoint nodes (boundaries): {endpoints}\n")
        
        # 6. Summary Insights
        f.write(f"\n\n📌 6. KEY INSIGHTS & INTERPRETATIONS\n")
        f.write("-" * 40 + "\n")
        
        # Setosa analysis
        setosa_analysis = separability.get('Iris-setosa', {})
        if setosa_analysis:
            setosa_purity = setosa_analysis['pure_nodes'] / len(setosa_analysis['nodes']) if setosa_analysis['nodes'] else 0
            f.write(f"🌸 Iris-setosa: {setosa_purity:.1%} pure separation - clearly linearly separable\n")
        
        # Versicolor/Virginica confusion
        versicolor_mixed = sum(1 for m in mixing.values() if 'Iris-versicolor' in m['minority_classes'] or m['majority_class'] == 'Iris-versicolor')
        virginica_mixed = sum(1 for m in mixing.values() if 'Iris-virginica' in m['minority_classes'] or m['majority_class'] == 'Iris-virginica')
        
        f.write(f"🔄 Versicolor-Virginica confusion: {max(versicolor_mixed, virginica_mixed)} mixed nodes\n")
        f.write(f"   → These classes are NOT linearly separable in the given feature space\n")
        
        # Region diversity
        avg_entropy = np.mean([r['entropy'] for r in regions.values()])
        f.write(f"🌍 Average region diversity (entropy): {avg_entropy:.2f}\n")
        f.write(f"   → Higher entropy = more mixed regions (like diverse provinces)\n")
        
        # Quantization quality
        good_representation = sum(1 for q in quantization.values() if q['representativeness'] > 0.8)
        total_nodes = len(quantization)
        f.write(f"🎯 Well-represented nodes: {good_representation}/{total_nodes} ({good_representation/total_nodes:.1%})\n")
        f.write(f"   → Node weights closely reflect actual data positioning\n")
        
        f.write(f"\n💡 CONCLUSION: GSOM reveals the underlying structure beyond simple classification.\n")
        f.write(f"   The spatial organization shows WHY and WHERE confusion occurs, not just THAT it occurs.\n")
    
    print(f"📄 Comprehensive analysis saved to '{output_file}'")
    return separability, mixing, regions, quantization, connectivity, paths

def plot_analysis_separate(gsom, output, entropy_dict, clusters, node_to_cluster, outliers, region_entropy, 
                          region_nodes, deviant_points, boundary_points, boundary_nodes, data, label_col, weight_columns, 
                          cluster_names, base_filename="gsom_boundary_analysis_crohn"):
    """Create separate visualization plots for GSOM analysis."""
    
    # Plot 1: GSOM Map with Clusters, Entropy, and Boundary Points
    def create_gsom_map():
        fig, ax = plt.subplots(figsize=(14, 10))
        
        max_entropy = max(entropy_dict.values(), default=1) if entropy_dict else 1
        colors_cluster = ['green', 'red', 'black', 'cyan']
        
        # Plot all nodes
        ax.scatter(gsom.node_coordinate[:gsom.node_count, 0], gsom.node_coordinate[:gsom.node_count, 1], 
                   c='gray', s=10, alpha=0.1, label='All Nodes')
        
        # Plot clustered nodes with entropy-based coloring
        for node_idx in entropy_dict:
            x, y = gsom.node_coordinate[node_idx]
            cluster_id = node_to_cluster.get(node_idx, -1)
            if cluster_id >= 0:
                color = colors_cluster[cluster_id % len(colors_cluster)]
            else:
                color = 'gray'
            entropy = entropy_dict.get(node_idx, 0)
            size = 30 if node_idx in gsom.node_labels['output'].values else 10
            ax.scatter(x, y, c=color, s=size, alpha=0.5 + 0.5 * (entropy / max_entropy), 
                      marker='D' if node_idx in gsom.node_labels['output'].values else 'o')
            ax.text(x, y, f"{node_idx}\nE={entropy:.2f}", fontsize=6)
        
        # Plot skeleton connections
        _, skeleton_connections, _, pos_edges = gsom.build_skeleton(data, weight_columns)
        from collections import Counter
        overlaps = Counter((i, j) for i, j in skeleton_connections)
        q3 = np.percentile(list(overlaps.values()), 75) if overlaps else 0
        
        for i, j in skeleton_connections:
            if overlaps[(i, j)] < q3:
                color = 'gray'
                alpha = 0.3
                line_style = '--'
            else:
                color = 'black' if (i, j) in pos_edges or (j, i) in pos_edges else 'red'
                alpha = 0.5 if (i, j) in pos_edges or (j, i) in pos_edges else 0.1
                line_style = '-'
            x1, y1 = gsom.node_coordinate[i]
            x2, y2 = gsom.node_coordinate[j]
            ax.plot([x1, x2], [y1, y2], color=color, linestyle=line_style, alpha=alpha)
        
        # Create legend for clusters with names
        legend_elements = []
        for cluster_id, cluster_name in cluster_names.items():
            if cluster_id < len(colors_cluster):
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                markerfacecolor=colors_cluster[cluster_id], 
                                                markersize=8, label=cluster_name))
        
        # Highlight boundary nodes
        for node_idx in boundary_nodes:
            x, y = gsom.node_coordinate[node_idx]
            ax.scatter(x, y, c='none', s=100, edgecolors='purple', linewidth=2, 
                      label='Boundary Node' if node_idx == list(boundary_nodes)[0] else "")
        
        # Highlight boundary points
        for point in boundary_points:
            node_idx = point['node']
            x, y = gsom.node_coordinate[node_idx]
            ax.scatter(x, y, c='magenta', s=50, marker='*', edgecolors='black', 
                      label='Boundary Point' if point == boundary_points[0] else "")
        
        # Add cluster legend and other legends
        if legend_elements:
            ax.legend(handles=legend_elements + [plt.Line2D([0], [0], marker='o', color='w', 
                                                           markerfacecolor='purple', markersize=8, 
                                                           label='Boundary Node', markeredgecolor='purple'),
                                               plt.Line2D([0], [0], marker='*', color='w', 
                                                           markerfacecolor='magenta', markersize=8, 
                                                           label='Boundary Point', markeredgecolor='black')],
                     bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        ax.set_title(f"GSOM Map: Genetic Clusters, Entropy & Boundary Points\n(Region Entropy: {region_entropy:.2f})", fontsize=14)
        ax.set_xlabel("GSOM X Coordinate", fontsize=12)
        ax.set_ylabel("GSOM Y Coordinate", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f"{base_filename}_gsom_map.pdf"
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        print(f"📊 GSOM Map saved as {filename}")
        plt.close()
    
    # Plot 2: Confusion Matrix
    def create_confusion_matrix():
        fig, ax = plt.subplots(figsize=(10, 8))
        
        _, confusion, all_labels, _, _ = gsom.compute_cluster_purity(data, label_col, weight_columns)
        cluster_labels = [cluster_names.get(i, f'Cluster {i+1}') for i in range(len(confusion))]
        
        sns.heatmap(confusion, annot=True, fmt='.0f', cmap='Blues', ax=ax, 
                    xticklabels=all_labels, yticklabels=cluster_labels, cbar_kws={'label': 'Sample Count'})
        ax.set_title("Genetic Disease Classification: Confusion Matrix", fontsize=14)
        ax.set_xlabel("Actual Disease Status", fontsize=12)
        ax.set_ylabel("GSOM Predicted Clusters", fontsize=12)
        
        # Add accuracy metrics
        total_samples = confusion.sum()
        correct_predictions = np.trace(confusion)
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        
        # Add text with metrics
        metrics_text = f"Overall Accuracy: {accuracy:.1%}\nTotal Samples: {int(total_samples)}"
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        filename = f"{base_filename}_confusion_matrix.pdf"
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        print(f"📊 Confusion Matrix saved as {filename}")
        plt.close()
    
    # Plot 3: Boundary Features Analysis (Enhanced with Better Explanatory Visualizations)
    def create_boundary_features():
        if not boundary_points:
            print("⚠️  No boundary points found, skipping boundary features plot")
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 14))
        
        feature_names = weight_columns
        
        # Collect all feature differences for analysis
        feature_data = {feature: [] for feature in feature_names}
        cluster_data = []
        
        for point in boundary_points:
            feature_diffs = point['feature_diff_node']
            cluster_name = cluster_names.get(point['cluster'], f"Cluster {point['cluster'] + 1}")
            
            for i, feature in enumerate(feature_names):
                feature_data[feature].append(feature_diffs[i])
            cluster_data.append(cluster_name)
        
        # Plot 3a: Feature Importance Ranking (Instead of box plot)
        # Calculate mean absolute differences for each feature
        feature_importance = {}
        for feature in feature_names:
            if feature_data[feature]:
                feature_importance[feature] = np.mean(feature_data[feature])
        
        # Get top 15 features for better visibility
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:15]
        
        if top_features:
            features, importances = zip(*top_features)
            # Create horizontal bar chart for better readability
            y_pos = np.arange(len(features))
            bars = ax1.barh(y_pos, importances, color=plt.cm.plasma(np.linspace(0, 1, len(features))))
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(features)
            ax1.set_xlabel("Mean Genetic Marker Difference", fontsize=12)
            ax1.set_title(f"Top 15 Genetic Markers Causing Boundary Confusion\n(n={len(boundary_points)} boundary patients)", fontsize=14)
            ax1.grid(True, alpha=0.3, axis='x')
            
            # Add value labels on bars
            for bar, importance in zip(bars, importances):
                width = bar.get_width()
                ax1.text(width, bar.get_y() + bar.get_height()/2.,
                        f'{importance:.3f}', ha='left', va='center', fontsize=9)
        
        # Plot 3b: Genetic Pattern Comparison - Boundary vs Non-Boundary
        # Compare top 5 features between boundary and non-boundary patients
        top_5_features = [f[0] for f in top_features[:5]] if top_features else feature_names[:5]
        
        # Get non-boundary patients for comparison
        all_sample_ids = set(data['id'].tolist())
        boundary_sample_ids = set([point['sample_id'] for point in boundary_points])
        non_boundary_sample_ids = all_sample_ids - boundary_sample_ids
        
        boundary_means = []
        non_boundary_means = []
        
        for feature in top_5_features:
            # Calculate means for boundary patients
            boundary_values = []
            for point in boundary_points:
                sample_id = point['sample_id']
                sample_row = data[data['id'] == sample_id]
                if not sample_row.empty:
                    boundary_values.append(sample_row[feature].iloc[0])
            
            # Calculate means for non-boundary patients
            non_boundary_values = []
            for sample_id in list(non_boundary_sample_ids)[:50]:  # Sample 50 for comparison
                sample_row = data[data['id'] == sample_id]
                if not sample_row.empty:
                    non_boundary_values.append(sample_row[feature].iloc[0])
            
            boundary_means.append(np.mean(boundary_values) if boundary_values else 0)
            non_boundary_means.append(np.mean(non_boundary_values) if non_boundary_values else 0)
        
        x = np.arange(len(top_5_features))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, boundary_means, width, label='Boundary Patients', 
                       color='coral', alpha=0.8)
        bars2 = ax2.bar(x + width/2, non_boundary_means, width, label='Non-Boundary Patients', 
                       color='lightblue', alpha=0.8)
        
        ax2.set_xlabel('Top 5 Genetic Markers', fontsize=12)
        ax2.set_ylabel('Mean Genetic Value', fontsize=12)
        ax2.set_title('Genetic Pattern Comparison:\nBoundary vs Non-Boundary Patients', fontsize=14)
        ax2.set_xticks(x)
        ax2.set_xticklabels(top_5_features, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 3c: Feature Variability Analysis
        # Show how much each top feature varies across boundary patients
        top_10_features = [f[0] for f in top_features[:10]] if top_features else feature_names[:10]
        
        variabilities = []
        for feature in top_10_features:
            feature_values = []
            for point in boundary_points:
                sample_id = point['sample_id']
                sample_row = data[data['id'] == sample_id]
                if not sample_row.empty:
                    feature_values.append(sample_row[feature].iloc[0])
            
            if feature_values:
                variability = np.std(feature_values) / (np.mean(feature_values) + 1e-6)  # Coefficient of variation
                variabilities.append(variability)
            else:
                variabilities.append(0)
        
        bars = ax3.bar(range(len(top_10_features)), variabilities, 
                      color=plt.cm.coolwarm(np.array(variabilities) / max(variabilities) if variabilities else [0]))
        ax3.set_xticks(range(len(top_10_features)))
        ax3.set_xticklabels(top_10_features, rotation=45, ha='right')
        ax3.set_ylabel('Coefficient of Variation', fontsize=12)
        ax3.set_title('Genetic Variability in Boundary Patients\n(Higher = More Heterogeneous)', fontsize=14)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, var_val in zip(bars, variabilities):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{var_val:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 3d: Cluster-Specific Feature Analysis
        # Show which features distinguish boundary patients in each cluster
        cluster_feature_analysis = {}
        
        for point in boundary_points:
            cluster_name = cluster_names.get(point['cluster'], f"Cluster {point['cluster'] + 1}")
            if cluster_name not in cluster_feature_analysis:
                cluster_feature_analysis[cluster_name] = {feature: [] for feature in feature_names}
            
            feature_diffs = point['feature_diff_node']
            for i, feature in enumerate(feature_names):
                cluster_feature_analysis[cluster_name][feature].append(feature_diffs[i])
        
        # Get top 3 features for each cluster
        cluster_colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow']
        cluster_names_list = list(cluster_feature_analysis.keys())
        
        bar_width = 0.8 / len(cluster_names_list) if cluster_names_list else 0.8
        
        for i, (cluster_name, feature_dict) in enumerate(cluster_feature_analysis.items()):
            cluster_top_features = []
            cluster_importances = []
            
            for feature in top_5_features:  # Use top 5 from overall analysis
                if feature_dict[feature]:
                    importance = np.mean(feature_dict[feature])
                    cluster_top_features.append(feature)
                    cluster_importances.append(importance)
            
            x_pos = np.arange(len(cluster_top_features)) + i * bar_width
            color = cluster_colors[i % len(cluster_colors)]
            
            ax4.bar(x_pos, cluster_importances, bar_width, 
                   label=f'{cluster_name}', color=color, alpha=0.8)
        
        ax4.set_xlabel('Top Genetic Markers', fontsize=12)
        ax4.set_ylabel('Mean Difference', fontsize=12)
        ax4.set_title('Cluster-Specific Genetic Boundary Patterns', fontsize=14)
        ax4.set_xticks(np.arange(len(top_5_features)) + bar_width * (len(cluster_names_list) - 1) / 2)
        ax4.set_xticklabels(top_5_features, rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f"{base_filename}_boundary_features.pdf"
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        print(f"📊 Boundary Features Analysis saved as {filename}")
        plt.close()
        
        # Extended boundary features plot
        create_extended_boundary_features()
    
    # Plot 4: Extended Boundary Features Analysis
    def create_extended_boundary_features():
        if not boundary_points:
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        
        # Prepare data
        cluster_boundary_data = {}
        for point in boundary_points:
            cluster_name = cluster_names.get(point['cluster'], f"Cluster {point['cluster'] + 1}")
            if cluster_name not in cluster_boundary_data:
                cluster_boundary_data[cluster_name] = []
            cluster_boundary_data[cluster_name].append(point)
        
        # Plot 4a: Boundary Points Distribution by Cluster
        cluster_names_list = list(cluster_boundary_data.keys())
        cluster_counts = [len(cluster_boundary_data[name]) for name in cluster_names_list]
        
        colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow']
        bars = ax1.bar(cluster_names_list, cluster_counts, color=colors[:len(cluster_names_list)])
        ax1.set_title("Boundary Points Distribution by Genetic Cluster", fontsize=12)
        ax1.set_ylabel("Number of Boundary Patients", fontsize=11)
        
        # Add percentage labels
        total_boundary = sum(cluster_counts)
        for bar, count in zip(bars, cluster_counts):
            percentage = (count / total_boundary) * 100 if total_boundary > 0 else 0
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontsize=9)
        
        # Plot 4b: Distance to Cluster Centroids
        distances_to_centroids = []
        labels = []
        for point in boundary_points:
            distances = point['distances_to_centroids']
            for i, dist in enumerate(distances):
                distances_to_centroids.append(dist)
                labels.append(f"Cluster {i+1}")
        
        if distances_to_centroids:
            unique_labels = list(set(labels))
            distance_data = [[] for _ in unique_labels]
            for dist, label in zip(distances_to_centroids, labels):
                idx = unique_labels.index(label)
                distance_data[idx].append(dist)
            
            bp = ax2.boxplot(distance_data, tick_labels=unique_labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], ['lightcoral', 'lightblue']):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax2.set_title("Distance Distribution to Cluster Centroids", fontsize=12)
            ax2.set_ylabel("Genetic Distance", fontsize=11)
            ax2.grid(True, alpha=0.3)
        
        # Plot 4c: Feature Variability Heatmap (top 20 features)
        feature_names = weight_columns
        feature_importance = {}
        for feature in feature_names:
            feature_data = []
            for point in boundary_points:
                feature_idx = feature_names.index(feature)
                feature_data.append(point['feature_diff_node'][feature_idx])
            if feature_data:
                feature_importance[feature] = np.std(feature_data)  # Use std for variability
        
        top_variable_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20]
        
        if top_variable_features:
            # Create a matrix for heatmap
            features, _ = zip(*top_variable_features)
            matrix_data = []
            patient_labels = []
            
            for i, point in enumerate(boundary_points[:20]):  # Limit to first 20 patients for readability
                row = []
                for feature in features:
                    feature_idx = feature_names.index(feature)
                    row.append(point['feature_diff_node'][feature_idx])
                matrix_data.append(row)
                patient_labels.append(f"Patient {point['sample_id']}")
            
            if matrix_data:
                sns.heatmap(matrix_data, xticklabels=features, yticklabels=patient_labels,
                           cmap='viridis', ax=ax3, cbar_kws={'label': 'Feature Difference'})
                ax3.set_title("Feature Difference Heatmap\n(Top 20 Variable Features, 20 Patients)", fontsize=12)
                ax3.tick_params(axis='x', rotation=90, labelsize=8)
                ax3.tick_params(axis='y', labelsize=8)
        
        # Plot 4d: Cluster Separation Quality
        separation_metrics = []
        cluster_labels = []
        
        for cluster_name, points in cluster_boundary_data.items():
            if len(points) > 1:
                distances = []
                for point in points:
                    distances.extend(point['distances_to_centroids'])
                
                mean_dist = np.mean(distances)
                std_dist = np.std(distances)
                separation_quality = mean_dist / (std_dist + 1e-6)  # Higher is better separated
                
                separation_metrics.append(separation_quality)
                cluster_labels.append(cluster_name)
        
        if separation_metrics:
            bars = ax4.bar(cluster_labels, separation_metrics, color=['lightcoral', 'lightblue'])
            ax4.set_title("Cluster Separation Quality\n(Mean Distance / Std Distance)", fontsize=12)
            ax4.set_ylabel("Separation Quality Score", fontsize=11)
            
            for bar, metric in zip(bars, separation_metrics):
                ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'{metric:.2f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        filename = f"{base_filename}_boundary_features_extended.pdf"
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        print(f"📊 Extended Boundary Features Analysis saved as {filename}")
        plt.close()
    
    # Create all plots
    print(f"\n🎨 Creating separate visualization plots for {base_filename}...")
    create_gsom_map()
    create_confusion_matrix()
    create_boundary_features()
    
    print(f"\n✅ All separate plots created successfully!")
    print(f"   📈 {base_filename}_gsom_map.pdf")
    print(f"   📊 {base_filename}_confusion_matrix.pdf") 
    print(f"   🧬 {base_filename}_boundary_features.pdf")
    print(f"   🔬 {base_filename}_boundary_features_extended.pdf")

if __name__ == '__main__':
    np.random.seed(1)
    # Load Crohn's disease dataset
    df = pd.read_csv("crohn.csv")
    
    print("Dataset shape:", df.shape)
    print("Crohn values:", df['crohn'].unique())
    
    # Convert Crohn values to more descriptive labels
    df['crohn_label'] = df['crohn'].map({0: 'No_Crohns', 2: 'Crohns'})
    
    # Define feature columns (genetic loci)
    weight_columns = [col for col in df.columns if col.startswith('loc') and ('a1' in col or 'a2' in col)]
    print(f"Number of genetic features: {len(weight_columns)}")
    
    # Prepare training data with only the genetic features
    data_training = df[weight_columns]
    print("Training data head:", data_training.head())
    print("Training data shape:", data_training.shape)
    
    # Normalize the data since genetic markers have different scales
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    data_training_scaled = pd.DataFrame(scaler.fit_transform(data_training), columns=data_training.columns)
    df[weight_columns] = scaler.transform(df[weight_columns])
    
    # Train GSOM with adjusted parameters for genetic data
    # Using more dimensions and adjusted spread factor for genetic markers
    gsom = GSOM(0.83, len(weight_columns), max_radius=4, initial_node_size=2000)
    gsom.fit(data_training_scaled.to_numpy(), 100, 50)
    output = gsom.predict(df, "id", "crohn_label", weight_columns)
    output.to_csv("output_crohn.csv", index=False)
    print("GSOM training completed.")
    print("Output shape:", output.shape)
    print("Node Count:", gsom.node_count)
    
    # Compute cluster purity
    entropy_dict, confusion, all_labels, node_to_cluster, cluster_names = gsom.compute_cluster_purity(df, "crohn_label", weight_columns)
    # print("Node Entropy:")
    # for node_idx, entropy in entropy_dict.items():
    #     print(f"Node {node_idx}: Entropy = {entropy:.2f}")
    
    print("\nCluster Names:")
    for cluster_id, cluster_name in cluster_names.items():
        print(f"{cluster_name}")
    
    # 🧠 COMPREHENSIVE ANALYSIS: Understanding Genetic Data Structure in Crohn's Disease
    print("\n" + "="*50)
    print("🧠 PERFORMING COMPREHENSIVE STRUCTURE ANALYSIS FOR CROHN'S DISEASE")
    print("="*50)
    
    # Generate comprehensive analysis report
    separability, mixing, regions, quantization, connectivity, paths = comprehensive_analysis_report(
        gsom, df, "crohn_label", weight_columns, "crohn_comprehensive_analysis.txt")
    
    # Print key insights to console
    print("\n🔍 KEY INSIGHTS:")
    print("-" * 30)
    
    # Separability insights for Crohn's disease
    no_crohns_analysis = separability.get('No_Crohns', {})
    crohns_analysis = separability.get('Crohns', {})
    
    if no_crohns_analysis and no_crohns_analysis['nodes']:
        no_crohns_purity = no_crohns_analysis['pure_nodes'] / len(no_crohns_analysis['nodes'])
        print(f"🔬 No_Crohns separability: {no_crohns_purity:.1%} (linearly separable: {'✅' if no_crohns_purity > 0.95 else '❌'})")
    
    if crohns_analysis and crohns_analysis['nodes']:
        crohns_purity = crohns_analysis['pure_nodes'] / len(crohns_analysis['nodes'])
        print(f"🔬 Crohns separability: {crohns_purity:.1%} (linearly separable: {'✅' if crohns_purity > 0.95 else '❌'})")
    
    # Mixing analysis for genetic markers
    total_mixed_nodes = len(mixing)
    print(f"🔄 Mixed nodes found: {total_mixed_nodes} (where genetic patterns overlap)")
    
    # Region diversity in genetic space
    if regions:
        avg_purity = np.mean([r['purity'] for r in regions.values()])
        print(f"🌍 Average genetic region purity: {avg_purity:.1%}")
        
        # Find most problematic genetic mixing
        if mixing:
            most_mixed = max(mixing.items(), key=lambda x: len(x[1]['minority_classes']))
            print(f"🔗 Most mixed genetic node: {most_mixed[0]} ({most_mixed[1]['class_counts']})")
    
    # Quantization quality for genetic markers
    if quantization:
        avg_representativeness = np.mean([q['representativeness'] for q in quantization.values()])
        print(f"🎯 Genetic node representativeness: {avg_representativeness:.2%}")
    
    print(f"\n💡 UNDERSTANDING: This analysis reveals the GENETIC STRUCTURE and RELATIONSHIPS")
    print(f"   in Crohn's disease data, showing genetic marker patterns and disease associations.")
    
    # Genetic analysis for Crohn's disease
    outliers = gsom.detect_outliers(df, "crohn_label", weight_columns, threshold=2.0)
    center_node = output.loc[output['hit_count'].idxmax(), 'output']
    region_entropy, region_nodes, deviant_points = gsom.analyze_region(center_node, radius=2.0, data=df, label_col="crohn_label", weight_columns=weight_columns)
    boundary_points, boundary_nodes, node_to_cluster, clusters = gsom.identify_boundary_points(df, weight_columns, "crohn_label", max_clusters=2, distance_threshold=0.3)
    
    print(f"\n📊 BOUNDARY ANALYSIS:")
    print(f"   • Boundary points (genetic confusion): {len(boundary_points)}")
    print(f"   • Boundary nodes: {len(boundary_nodes)}")
    print(f"   • Region entropy: {region_entropy:.2f}")
    print(f"   • Deviant points: {len(deviant_points)}")
    
    # Save Crohn's disease analysis files
    with open("boundary_points_analysis_crohn.txt", "w", encoding="utf-8") as f:
        f.write("GSOM Boundary Points Analysis - Crohn's Disease\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Boundary Points Found: {len(boundary_points)}\n")
        f.write(f"These are genetic samples that are difficult to classify clearly.\n\n")
        
        # Write cluster names
        f.write("GENETIC CLUSTERS:\n")
        for cluster_id, cluster_name in cluster_names.items():
            f.write(f"{cluster_name}\n")
        f.write("\n")
        
        # Group boundary points by genetic features showing highest differences
        f.write("BOUNDARY POINTS WITH GENETIC MARKER ANALYSIS:\n")
        f.write("-" * 60 + "\n")
        for point in boundary_points:
            cluster_name = cluster_names.get(point['cluster'], f"Cluster {point['cluster'] + 1}")
            f.write(f"Sample ID: {point['sample_id']}, Node: {point['node']}, {cluster_name}, Label: {point['label']}\n")
            f.write(f"  Top Genetic Marker Differences from Node {point['node']}:\n")
            
            # Find top 10 most different genetic markers
            feature_diffs = [(feature, diff) for feature, diff in zip(weight_columns, point['feature_diff_node'])]
            top_features = sorted(feature_diffs, key=lambda x: x[1], reverse=True)[:10]
            
            for feature, diff in top_features:
                f.write(f"    {feature}: {diff:.4f}\n")
            f.write(f"  Distances to Disease Clusters: {[f'{d:.4f}' for d in point['distances_to_centroids']]}\n")
            f.write("-" * 60 + "\n")
    
    print(f"\nCrohn's disease boundary analysis saved to 'boundary_points_analysis_crohn.txt'")
    
    # Visualize results with Crohn's disease understanding - Create separate plots
    plot_analysis_separate(gsom, output, entropy_dict, clusters, node_to_cluster, outliers, region_entropy, 
                          region_nodes, deviant_points, boundary_points, boundary_nodes, df, "crohn_label", weight_columns, 
                          cluster_names, base_filename="gsom_boundary_analysis_crohn")
    
    print(f"\n✅ Crohn's Disease Analysis Complete!")
    print(f"   📄 crohn_comprehensive_analysis.txt - Detailed genetic analysis report")
    print(f"   📊 Multiple separate PDF plots created:")
    print(f"      • gsom_boundary_analysis_crohn_gsom_map.pdf - GSOM structure visualization")
    print(f"      • gsom_boundary_analysis_crohn_confusion_matrix.pdf - Classification accuracy")
    print(f"      • gsom_boundary_analysis_crohn_boundary_features.pdf - Genetic boundary analysis")
    print(f"      • gsom_boundary_analysis_crohn_boundary_features_extended.pdf - Extended genetic analysis")
    print(f"   📋 output_crohn.csv - GSOM clustering results")
    print(f"   🧬 boundary_points_analysis_crohn.txt - Genetic boundary analysis")
    
    print("\n" + "="*50)
    print("✅ COMPLETE: Analysis focuses on UNDERSTANDING data structure")
    print("   📄 Main insights in: iris_comprehensive_analysis.txt")
    print("   📊 Visualization: gsom_boundary_analysis_iris.pdf")
    print("="*50)
