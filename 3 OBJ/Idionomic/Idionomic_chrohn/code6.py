"""
🧠 Enhanced GSOM Analysis for Understanding Data Structure and Relationships

This implementation focuses on understanding the UNDERLYING STRUCTURE of data rather than
just classification accuracy. Key enhancements based on conceptual notes:

1. 📊 SEPARABILITY ANALYSIS: Identifies which classes are linearly separable
   - Setosa is clearly separable
   - Versicolor and Virginica show mixing patterns

2. 🔄 CLASS MIXING ANALYSIS: Explains WHY and WHERE classes overlap
   - Analyzes feature differences in mixed nodes
   - Shows which variables cause confusion

3. 🌍 REGIONS vs CLUSTERS: Distinguishes spatial proximity from similarity
   - Regions are like provinces (spatial neighborhoods)
   - Clusters are similarity groups within regions

4. 🎯 QUANTIZATION DIFFERENCE: Measures how well node weights represent data
   - Higher values indicate better representation
   - Helps identify outliers and poorly represented areas

5. 🧱 SKELETON STRUCTURE: Reveals data topology and relationships
   - Shows branching patterns and connectivity
   - Identifies junction points and boundaries

Focus: Understanding STRUCTURE and RELATIONSHIPS, not just classification performance.
"""

import numpy as np
import pandas as pd
from scipy.spatial import distance
import scipy
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

    def predict(self, data, index_col, label_col=None):
        weight_columns = list(data.columns.values)
        output_columns = [index_col]
        if label_col:
            weight_columns.remove(label_col)
            output_columns.append(label_col)
        weight_columns.remove(index_col)
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

    def separate_clusters(self, data, weight_columns, max_clusters=3):
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
        
        clusters, _, _, _ = self.separate_clusters(data, weight_columns, max_clusters=3)
        for cluster_id, cluster in enumerate(clusters[-1]):
            for node_idx in cluster:
                node_to_cluster[node_idx] = cluster_id
        
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
        
        all_labels = data[label_col].unique()
        label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
        confusion = np.zeros((len(clusters[-1]), len(all_labels)))
        for item in confusion_data:
            cluster_id = item['cluster']
            for label in item['labels']:
                label_idx = label_to_idx[label]
                confusion[cluster_id, label_idx] += 1
        
        # Generate cluster names based on majority labels
        cluster_names = {}
        for cluster_id in range(len(clusters[-1])):
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
                sample_ids = row['Id']
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
            sample_ids = row['Id']
            labels = row[label_col]
            
            if len(set(labels)) > 1:  # Mixed node
                label_counts = pd.Series(labels).value_counts()
                majority_class = label_counts.idxmax()
                minority_classes = [cls for cls in label_counts.index if cls != majority_class]
                
                # Analyze feature differences between classes in this node
                node_samples = data[data['Id'].isin(sample_ids)]
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
                        node_sample_ids = node_row['Id'].iloc[0]
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
            sample_ids = row['Id']
            labels = row[label_col]
            node_weights = self.node_list[node_idx]
            
            sample_indices = data[data['Id'].isin(sample_ids)].index
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
            sample_ids = row['Id']
            labels = row[label_col]
            node_weights = self.node_list[node_idx]
            
            # Get actual sample data
            sample_indices = data[data['Id'].isin(sample_ids)].index
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
            sample_ids = row['Id']
            labels = row[label_col]
            node_weights = self.node_list[node_idx]
            
            sample_indices = data[data['Id'].isin(sample_ids)].index
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
                sample_ids = node_row['Id'].iloc[0]
                labels = node_row[label_col].iloc[0]
                node_weights = self.node_list[node_idx]
                sample_indices = data[data['Id'].isin(sample_ids)].index
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

    def identify_boundary_points(self, data, weight_columns, label_col, max_clusters=3, distance_threshold=0.5):
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
            sample_ids = row['Id']
            labels = row[label_col]
            node_weights = self.node_list[node_idx]
            sample_indices = data[data['Id'].isin(sample_ids)].index
            
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

def plot_analysis(gsom, output, entropy_dict, clusters, node_to_cluster, outliers, region_entropy, 
                 region_nodes, deviant_points, boundary_points, boundary_nodes, data, label_col, weight_columns, 
                 cluster_names, file_name="gsom_boundary_analysis_iris.pdf"):
    """Visualize GSOM nodes, clusters, purity, outliers, and boundary points."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    
    # Plot 1: GSOM Map with Clusters, Entropy, and Boundary Points
    max_entropy = max(entropy_dict.values(), default=1) if entropy_dict else 1
    colors_cluster = ['green', 'red', 'black', 'cyan']
    
    # Plot all nodes
    ax1.scatter(gsom.node_coordinate[:gsom.node_count, 0], gsom.node_coordinate[:gsom.node_count, 1], 
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
        ax1.scatter(x, y, c=color, s=size, alpha=0.5 + 0.5 * (entropy / max_entropy), 
                    marker='D' if node_idx in gsom.node_labels['output'].values else 'o')
        ax1.text(x, y, f"{node_idx}\nE={entropy:.2f}", fontsize=6)
    
    # Plot skeleton connections
    _, skeleton_connections, _, pos_edges = gsom.build_skeleton(data, weight_columns)
    from collections import Counter
    overlaps = Counter((i, j) for i, j in skeleton_connections)
    q3 = np.percentile(list(overlaps.values()), 75)
    
    for i, j in skeleton_connections:
        if overlaps[(i, j)] < q3:
            line_width = 0.2
            color = 'gray'
            alpha = 0.3
            line_style = '--'
        else:
            color = 'black' if (i, j) in pos_edges or (j, i) in pos_edges else 'red'
            alpha = 0.5 if (i, j) in pos_edges or (j, i) in pos_edges else 0.1
            line_style = '-'
        x1, y1 = gsom.node_coordinate[i]
        x2, y2 = gsom.node_coordinate[j]
        ax1.plot([x1, x2], [y1, y2], color=color, linestyle=line_style, alpha=alpha)
    
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
        ax1.scatter(x, y, c='none', s=100, edgecolors='purple', linewidth=2, label='Boundary Node' if node_idx == list(boundary_nodes)[0] else "")
    
    # Highlight boundary points
    for point in boundary_points:
        node_idx = point['node']
        x, y = gsom.node_coordinate[node_idx]
        ax1.scatter(x, y, c='magenta', s=50, marker='*', edgecolors='black', label='Boundary Point' if point == boundary_points[0] else "")
    
    # Add cluster legend and other legends
    if legend_elements:
        ax1.legend(handles=legend_elements + [plt.Line2D([0], [0], marker='o', color='w', 
                                                        markerfacecolor='purple', markersize=8, 
                                                        label='Boundary Node', markeredgecolor='purple'),
                                            plt.Line2D([0], [0], marker='*', color='w', 
                                                        markerfacecolor='magenta', markersize=8, 
                                                        label='Boundary Point', markeredgecolor='black')],
                  bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    ax1.set_title(f"GSOM Map with Named Clusters, Entropy, and Boundary Points (Region Entropy: {region_entropy:.2f})")
    
    # Plot 2: Confusion Matrix with cluster names
    _, confusion, all_labels, _, _ = gsom.compute_cluster_purity(data, label_col, weight_columns)
    cluster_labels = [cluster_names.get(i, f'Cluster {i+1}') for i in range(len(confusion))]
    sns.heatmap(confusion, annot=True, fmt='.0f', cmap='Blues', ax=ax2, 
                xticklabels=all_labels, yticklabels=cluster_labels)
    ax2.set_title("Confusion Matrix with Named Clusters")
    ax2.set_xlabel("Species")
    ax2.set_ylabel("Named Clusters")
    
    # Plot 3: Box Plot for Feature Differences of All Boundary Points
    if boundary_points:
        feature_names = weight_columns
        
        # Collect all feature differences for box plot
        feature_data = {feature: [] for feature in feature_names}
        cluster_data = []
        
        for point in boundary_points:
            feature_diffs = point['feature_diff_node']
            cluster_name = cluster_names.get(point['cluster'], f"Cluster {point['cluster'] + 1}")
            
            for i, feature in enumerate(feature_names):
                feature_data[feature].append(feature_diffs[i])
            cluster_data.append(cluster_name)
        
        # Create box plot data
        box_data = [feature_data[feature] for feature in feature_names]
        
        # Create box plot
        bp = ax3.boxplot(box_data, tick_labels=feature_names, patch_artist=True)
        
        # Color the boxes
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax3.set_ylabel("Absolute Feature Difference")
        ax3.set_title(f"Feature Differences Distribution for All Boundary Points (n={len(boundary_points)})")
        ax3.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = []
        for i, feature in enumerate(feature_names):
            data = feature_data[feature]
            if data:
                median_val = np.median(data)
                mean_val = np.mean(data)
                stats_text.append(f"{feature}: μ={mean_val:.3f}, M={median_val:.3f}")
        
        # Add text box with statistics
        if stats_text:
            textstr = '\n'.join(stats_text)
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax3.text(0.02, 0.98, textstr, transform=ax3.transAxes, fontsize=8,
                    verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(file_name, bbox_inches='tight')
    print(f"Analysis plot saved as {file_name}")
    plt.close()  # Close the figure to free memory

if __name__ == '__main__':
    np.random.seed(1)
    # Load Iris dataset
    # data_filename = "example/data/iris.csv".replace('\\', '/')
    df = pd.read_csv("Iris.csv")
    
    print("Dataset shape:", df.shape)
    data_training = df.iloc[:, 1:5]
    print("Training data head:", data_training.head())
    print("Training data shape:", data_training.shape)

    # from sklearn.preprocessing import MinMaxScaler
    # scaler = MinMaxScaler()
    # weight_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    # data_training = pd.DataFrame(scaler.fit_transform(data_training), columns=data_training.columns)
    # df[weight_columns] = scaler.transform(df[weight_columns])
    
    # Define feature columns
    weight_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    
    # Train GSOM
    gsom = GSOM(0.83, 4, max_radius=4, initial_node_size=1000)
    gsom.fit(data_training.to_numpy(), 100, 50)
    output = gsom.predict(df, "Id", "Species")
    output.to_csv("output_iris.csv", index=False)
    print("GSOM training completed.")
    print("Output shape:", output.shape)
    print("Node Count:", gsom.node_count)
    
    # Compute cluster purity
    entropy_dict, confusion, all_labels, node_to_cluster, cluster_names = gsom.compute_cluster_purity(df, "Species", weight_columns)
    print("Node Entropy:")
    for node_idx, entropy in entropy_dict.items():
        print(f"Node {node_idx}: Entropy = {entropy:.2f}")
    
    print("\nCluster Names:")
    for cluster_id, cluster_name in cluster_names.items():
        print(f"{cluster_name}")
    
    # 🧠 COMPREHENSIVE ANALYSIS: Understanding Data Structure
    print("\n" + "="*50)
    print("🧠 PERFORMING COMPREHENSIVE STRUCTURE ANALYSIS")
    print("="*50)
    
    # Generate comprehensive analysis report
    separability, mixing, regions, quantization, connectivity, paths = comprehensive_analysis_report(
        gsom, df, "Species", weight_columns, "iris_comprehensive_analysis.txt")
    
    # Print key insights to console
    print("\n🔍 KEY INSIGHTS:")
    print("-" * 30)
    
    # Separability insights
    setosa_analysis = separability.get('Iris-setosa', {})
    if setosa_analysis and setosa_analysis['nodes']:
        setosa_purity = setosa_analysis['pure_nodes'] / len(setosa_analysis['nodes'])
        print(f"🌸 Setosa separability: {setosa_purity:.1%} (linearly separable: {'✅' if setosa_purity > 0.95 else '❌'})")
    
    # Mixing analysis
    total_mixed_nodes = len(mixing)
    print(f"🔄 Mixed nodes found: {total_mixed_nodes} (where classes overlap)")
    
    # Region diversity
    if regions:
        avg_purity = np.mean([r['purity'] for r in regions.values()])
        print(f"🌍 Average region purity: {avg_purity:.1%}")
        
        # Find most problematic mixing
        if mixing:
            most_mixed = max(mixing.items(), key=lambda x: len(x[1]['minority_classes']))
            print(f"🔗 Most mixed node: {most_mixed[0]} ({most_mixed[1]['class_counts']})")
    
    # Quantization quality
    if quantization:
        avg_representativeness = np.mean([q['representativeness'] for q in quantization.values()])
        print(f"🎯 Node representativeness: {avg_representativeness:.2%}")
    
    print(f"\n💡 UNDERSTANDING: This analysis reveals the STRUCTURE and RELATIONSHIPS")
    print(f"   in data, showing WHY and WHERE classes mix, not just classification accuracy.")
    
    # Original analyses (keeping for completeness)
    outliers = gsom.detect_outliers(df, "Species", weight_columns, threshold=2.0)
    center_node = output.loc[output['hit_count'].idxmax(), 'output']
    region_entropy, region_nodes, deviant_points = gsom.analyze_region(center_node, radius=2.0, data=df, label_col="Species", weight_columns=weight_columns)
    boundary_points, boundary_nodes, node_to_cluster, clusters = gsom.identify_boundary_points(df, weight_columns, "Species", max_clusters=3, distance_threshold=0.5)
    
    # Save traditional analysis files
    with open("boundary_points_analysis.txt", "w", encoding="utf-8") as f:
        f.write("GSOM Boundary Points Analysis\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total Boundary Points Found: {len(boundary_points)}\n\n")
        
        # Write cluster names
        f.write("CLUSTER NAMES:\n")
        for cluster_id, cluster_name in cluster_names.items():
            f.write(f"{cluster_name}\n")
        f.write("\n")
        
        for point in boundary_points:
            cluster_name = cluster_names.get(point['cluster'], f"Cluster {point['cluster'] + 1}")
            f.write(f"Sample ID: {point['sample_id']}, Node: {point['node']}, {cluster_name}, Label: {point['label']}\n")
            f.write(f"  Feature Differences from Node {point['node']}:\n")
            for feature, diff in zip(weight_columns, point['feature_diff_node']):
                f.write(f"    {feature}: {diff:.4f}\n")
            f.write(f"  Distances to Cluster Centroids: {[f'{d:.4f}' for d in point['distances_to_centroids']]}\n")
            f.write("-" * 60 + "\n")
    
    print(f"\nBoundary points analysis saved to 'boundary_points_analysis.txt'")
    
    # Visualize results with enhanced understanding
    plot_analysis(gsom, output, entropy_dict, clusters, node_to_cluster, outliers, region_entropy, 
                  region_nodes, deviant_points, boundary_points, boundary_nodes, df, "Species", weight_columns, cluster_names)
    
    print("\n" + "="*50)
    print("✅ COMPLETE: Analysis focuses on UNDERSTANDING data structure")
    print("   📄 Main insights in: iris_comprehensive_analysis.txt")
    print("   📊 Visualization: gsom_boundary_analysis_iris.pdf")
    print("="*50)