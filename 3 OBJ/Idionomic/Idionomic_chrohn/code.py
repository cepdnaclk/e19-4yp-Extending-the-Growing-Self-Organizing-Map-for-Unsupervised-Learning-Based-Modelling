"""
🧠 Enhanced GSOM Analysis for Understanding Data Structure and Relationships - Crohn's Disease Dataset

This implementation focuses on understanding the UNDERLYING STRUCTURE of genetic data rather than
just classification accuracy. Key enhancements based on conceptual notes:

1. 📊 SEPARABILITY ANALYSIS: Identifies which classes are linearly separable
   - Crohn's disease cases vs controls
   - Genetic loci patterns and mixing

2. 🔄 CLASS MIXING ANALYSIS: Explains WHY and WHERE classes overlap
   - Analyzes genetic differences in mixed nodes
   - Shows which genetic loci cause confusion

3. 🌍 REGIONS vs CLUSTERS: Distinguishes spatial proximity from similarity
   - Regions are like provinces (spatial neighborhoods)
   - Clusters are similarity groups within regions

4. 🎯 QUANTIZATION DIFFERENCE: Measures how well node weights represent data
   - Higher values indicate better representation
   - Helps identify outliers and poorly represented areas

5. 🧱 SKELETON STRUCTURE: Reveals data topology and relationships
   - Shows branching patterns and connectivity
   - Identifies junction points and boundaries

Focus: Understanding GENETIC STRUCTURE and RELATIONSHIPS in Crohn's disease, not just classification performance.
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

    def separate_clusters(self, data, weight_columns, max_clusters=None):
        """
        Smart clustering that dynamically determines optimal number of clusters
        based on distance thresholds and data structure. Only creates non-empty clusters.
        """
        hit_points, skeleton_connections, junctions, pos_edges = self.build_skeleton(data, weight_columns)
        
        if not skeleton_connections:
            # No connections, each hit point is its own cluster
            return [[{node} for node in hit_points]], [], skeleton_connections, pos_edges
        
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
        
        # Determine optimal max_clusters if not provided
        if max_clusters is None:
            # Use number of unique classes as upper bound, but allow dynamic detection
            num_classes = len(data[data.columns[-1]].unique()) if hasattr(data, 'columns') else 3
            max_clusters = min(num_classes, len(hit_points))  # Can't have more clusters than nodes
        
        G = nx.Graph(skeleton_connections)
        clusters_history = []
        removed_segments = []
        remaining_connections = skeleton_connections.copy()
        
        # Calculate distance statistics for smart thresholding
        distances = [dist for _, _, dist in segments]
        if distances:
            distance_threshold = np.mean(distances) + np.std(distances)
        else:
            distance_threshold = float('inf')
        
        # Start with one cluster (all connected)
        initial_clusters = list(nx.connected_components(G))
        clusters_history.append(initial_clusters)
        
        # Remove segments iteratively until we reach desired clusters or threshold
        for i, j, dist in segments:
            if (i, j) in remaining_connections:
                remaining_connections.remove((i, j))
                
            if G.has_edge(i, j):
                G.remove_edge(i, j)
                current_clusters = list(nx.connected_components(G))
                
                # Only add to history if we actually split into more clusters
                if len(current_clusters) > len(clusters_history[-1]):
                    clusters_history.append(current_clusters)
                    removed_segments.append((i, j, dist))
                    print(f"Removed segment {i}-{j}, Distance: {dist:.4f}")
                    
                    # Stop conditions:
                    # 1. Reached max_clusters
                    # 2. Distance becomes too small (clusters too granular)
                    # 3. All meaningful connections removed
                    if (len(current_clusters) >= max_clusters or 
                        dist < distance_threshold / 2 or
                        len(current_clusters) >= len(hit_points)):
                        break
        
        # If no meaningful splits occurred, keep the initial clustering
        if not clusters_history:
            clusters_history = [initial_clusters]
        
        return clusters_history, removed_segments, remaining_connections, pos_edges

    def compute_cluster_purity(self, data, label_col, weight_columns):
        entropy_dict = {}
        confusion_data = []
        node_to_cluster = {}
        
        # Use smart clustering instead of fixed number
        clusters_history, removed_segments, _, _ = self.separate_clusters(data, weight_columns, max_clusters=None)
        
        if not clusters_history:
            print("No clusters generated - using single cluster.")
            return {}, np.array([]), [], {}, {"0": "Single Cluster: All data"}
        
        # Use the final clustering result
        final_clusters = clusters_history[-1]
        
        # Map nodes to clusters
        for cluster_id, cluster in enumerate(final_clusters):
            for node_idx in cluster:
                node_to_cluster[node_idx] = cluster_id
        
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
        
        # Build confusion matrix
        all_labels = data[label_col].unique()
        label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
        confusion = np.zeros((len(final_clusters), len(all_labels)))
        for item in confusion_data:
            cluster_id = item['cluster']
            for label in item['labels']:
                label_idx = label_to_idx[label]
                confusion[cluster_id, label_idx] += 1
        
        # Count clusters with actual data
        non_empty_clusters = sum(1 for cluster_id in range(len(final_clusters)) 
                                if confusion[cluster_id].sum() > 0)
        print(f"Generated {non_empty_clusters} meaningful clusters with data (removed {len(removed_segments)} segments)")
        
        # Generate meaningful cluster names based on content and characteristics
        # Only create names for clusters that actually have data
        cluster_names = {}
        for cluster_id in range(len(final_clusters)):
            cluster_label_counts = confusion[cluster_id]
            if cluster_label_counts.sum() > 0:  # Only process clusters with actual data
                dominant_label_idx = np.argmax(cluster_label_counts)
                dominant_label = all_labels[dominant_label_idx]
                dominant_count = cluster_label_counts[dominant_label_idx]
                total_count = cluster_label_counts.sum()
                purity = dominant_count / total_count
                
                # Create descriptive names based on purity and content
                if purity >= 0.9:
                    cluster_type = "Pure"
                elif purity >= 0.7:
                    cluster_type = "Dominant"
                else:
                    cluster_type = "Mixed"
                
                # Add size information
                if total_count >= 50:
                    size_desc = "Large"
                elif total_count >= 20:
                    size_desc = "Medium"
                else:
                    size_desc = "Small"
                
                cluster_names[cluster_id] = f"{size_desc} {cluster_type}: Class {dominant_label} ({purity:.1%})"
            # Skip empty clusters - don't create names for them
        
        # Print only meaningful clusters
        print("Cluster Names:")
        for cluster_id, cluster_name in cluster_names.items():
            print(cluster_name)
        
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

    def auto_determine_boundary_threshold(self, data, weight_columns, label_col, max_clusters=None):
        """
        Automatically determine optimal boundary threshold based on dataset characteristics.
        Uses statistical analysis of distance distributions and class separability.
        """
        if max_clusters is None:
            max_clusters = len(data[label_col].unique())
        
        # Get cluster information
        clusters, _, _, _ = self.separate_clusters(data, weight_columns, max_clusters)
        node_to_cluster = {}
        for cluster_id, cluster in enumerate(clusters[-1]):
            for node_idx in cluster:
                node_to_cluster[node_idx] = cluster_id
        
        # Compute cluster centroids
        cluster_centroids = []
        for cluster_id in range(len(clusters[-1])):
            cluster_nodes = clusters[-1][cluster_id]
            if len(cluster_nodes) > 0:
                cluster_weights = np.mean([self.node_list[node_idx] for node_idx in cluster_nodes], axis=0)
                cluster_centroids.append(cluster_weights)
        
        if len(cluster_centroids) < 2:
            return 0.1  # Default fallback
        
        # Calculate all inter-cluster distances
        inter_cluster_distances = []
        for i in range(len(cluster_centroids)):
            for j in range(i + 1, len(cluster_centroids)):
                dist = scipy.spatial.distance.cdist(
                    cluster_centroids[i].reshape(1, -1),
                    cluster_centroids[j].reshape(1, -1),
                    self.distance
                )[0][0]
                inter_cluster_distances.append(dist)
        
        # Calculate intra-cluster distances (samples to their cluster centroid)
        intra_cluster_distances = []
        data_n = data[weight_columns].to_numpy()
        
        for _, row in self.node_labels.iterrows():
            node_idx = row['output']
            sample_ids = row['id']
            cluster_id = node_to_cluster.get(node_idx, -1)
            
            if cluster_id >= 0 and cluster_id < len(cluster_centroids):
                sample_indices = data[data['id'].isin(sample_ids)].index
                centroid = cluster_centroids[cluster_id]
                
                distances = scipy.spatial.distance.cdist(
                    data_n[sample_indices], centroid.reshape(1, -1), self.distance
                ).flatten()
                intra_cluster_distances.extend(distances)
        
        # Statistical analysis
        if len(inter_cluster_distances) == 0 or len(intra_cluster_distances) == 0:
            return 0.1  # Default fallback
        
        inter_mean = np.mean(inter_cluster_distances)
        inter_std = np.std(inter_cluster_distances)
        intra_mean = np.mean(intra_cluster_distances)
        intra_std = np.std(intra_cluster_distances)
        
        # Dataset characteristics
        n_features = len(weight_columns)
        n_samples = len(data)
        
        # Adaptive threshold calculation based on multiple factors
        
        # Factor 1: Cluster separation ratio (how well separated clusters are)
        separation_ratio = inter_mean / (intra_mean + 1e-8)  # Avoid division by zero
        
        # Factor 2: Feature dimensionality impact
        # Higher dimensions need smaller thresholds
        dimensionality_factor = 1.0 / np.log(n_features + 1)
        
        # Factor 3: Overlap detection sensitivity
        # Based on the overlap between intra and inter cluster distance distributions
        overlap_threshold = max(intra_mean + 2 * intra_std, inter_mean - 2 * inter_std)
        normalized_overlap = overlap_threshold / inter_mean if inter_mean > 0 else 1.0
        
        # Factor 4: Sample density factor
        # More samples allow for more precise boundary detection
        density_factor = min(1.0, np.log(n_samples) / 10.0)
        
        # Combine factors to calculate optimal threshold
        base_threshold = 0.15  # Conservative base
        
        # Adjust based on separation quality
        if separation_ratio > 3.0:  # Well separated clusters
            threshold = base_threshold * 1.5
        elif separation_ratio > 2.0:  # Moderately separated
            threshold = base_threshold
        else:  # Poorly separated clusters - need more sensitivity
            threshold = base_threshold * 0.5
        
        # Apply dimensionality correction
        threshold *= dimensionality_factor
        
        # Apply overlap sensitivity
        threshold *= (1.0 - normalized_overlap * 0.5)
        
        # Apply density correction
        threshold *= density_factor
        
        # Ensure reasonable bounds
        min_threshold = 0.02  # Minimum for any dataset
        max_threshold = 0.5   # Maximum to avoid too many boundary points
        
        optimal_threshold = np.clip(threshold, min_threshold, max_threshold)
        
        # Validation: Check boundary point percentage
        test_boundary_points, _, _, _ = self.identify_boundary_points(
            data, weight_columns, label_col, max_clusters, optimal_threshold
        )
        boundary_percentage = len(test_boundary_points) / n_samples
        
        # If boundary percentage is too high (>40%), reduce threshold
        if boundary_percentage > 0.4:
            optimal_threshold *= 0.5
        # If too low (<5%), increase threshold
        elif boundary_percentage < 0.05:
            optimal_threshold *= 1.5
        
        # Final bounds check
        optimal_threshold = np.clip(optimal_threshold, min_threshold, max_threshold)
        
        return optimal_threshold, {
            'separation_ratio': separation_ratio,
            'dimensionality_factor': dimensionality_factor,
            'overlap_sensitivity': normalized_overlap,
            'density_factor': density_factor,
            'inter_cluster_mean': inter_mean,
            'intra_cluster_mean': intra_mean,
            'boundary_percentage': len(test_boundary_points) / n_samples,
            'n_features': n_features,
            'n_samples': n_samples
        }

    def identify_boundary_points(self, data, weight_columns, label_col, max_clusters=None, distance_threshold=0.5):
        """Identify data points on cluster boundaries and analyze feature differences."""
        
        # Use smart clustering
        clusters_history, removed_segments, _, _ = self.separate_clusters(data, weight_columns, max_clusters)
        
        if not clusters_history:
            print("No clusters for boundary analysis.")
            return [], set(), {}, []
        
        final_clusters = clusters_history[-1]
        node_to_cluster = {}
        for cluster_id, cluster in enumerate(final_clusters):
            for node_idx in cluster:
                node_to_cluster[node_idx] = cluster_id
        
        # Identify boundary nodes (nodes with connections to other clusters)
        boundary_nodes = set()
        for i, j, dist in removed_segments:
            if node_to_cluster.get(i, -1) != node_to_cluster.get(j, -1):
                boundary_nodes.add(i)
                boundary_nodes.add(j)
        
        # Identify boundary points based on distance to other clusters
        boundary_points = []
        data_n = data[weight_columns].to_numpy()
        
        # Compute cluster centroids (mean weight vector of nodes in each cluster)
        cluster_centroids = []
        for cluster_id in range(len(final_clusters)):
            cluster_nodes = final_clusters[cluster_id]
            if len(cluster_nodes) > 0:
                cluster_weights = np.mean([self.node_list[node_idx] for node_idx in cluster_nodes], axis=0)
                cluster_centroids.append(cluster_weights)
            else:
                cluster_centroids.append(np.zeros(len(weight_columns)))
        
        if len(cluster_centroids) < 2:
            print("Not enough clusters for boundary analysis.")
            return [], boundary_nodes, node_to_cluster, final_clusters
        
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
                sorted_distances = np.sort(distances)
                
                # Check if there's a second cluster close enough
                if len(sorted_distances) > 1:
                    second_min_dist = sorted_distances[1]
                    
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
        
        return boundary_points, boundary_nodes, node_to_cluster, final_clusters

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
            
            # Determine class separability based on purity score
            purity_score = analysis['pure_nodes'] / len(analysis['nodes'])
            if purity_score > 0.8:
                f.write(f"  ✅ Class {class_name} shows high separability (well-separated)\n")
            elif purity_score > 0.5:
                f.write(f"  🔶 Class {class_name} shows moderate separability\n")
            else:
                f.write(f"  ⚠️  Class {class_name} shows high mixing with other classes\n")
        
        # 2. Class Mixing Analysis
        f.write(f"\n\n🔄 2. CLASS MIXING ANALYSIS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total mixed nodes: {len(mixing)}\n\n")
        
        for node_idx, mix_info in mixing.items():
            f.write(f"🔗 Node {node_idx} (Mixed):\n")
            f.write(f"  Majority: {mix_info['majority_class']} ({mix_info['class_counts'][mix_info['majority_class']]} samples)\n")
            f.write(f"  Minorities: {', '.join(str(cls) for cls in mix_info['minority_classes'])}\n")
            
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
        
        # Class-specific analysis
        class_names = list(separability.keys())
        if len(class_names) >= 2:
            # Analyze most separable class
            best_class = max(separability.keys(), key=lambda x: separability[x]['pure_nodes'] / len(separability[x]['nodes']) if separability[x]['nodes'] else 0)
            best_purity = separability[best_class]['pure_nodes'] / len(separability[best_class]['nodes']) if separability[best_class]['nodes'] else 0
            f.write(f"✅ Most separable class: {best_class} ({best_purity:.1%} pure separation)\n")
            
            # Analyze most confusing classes
            most_mixed_class = max(separability.keys(), key=lambda x: len(separability[x]['nodes']) - separability[x]['pure_nodes'] if separability[x]['nodes'] else 0)
            mixed_count = len(separability[most_mixed_class]['nodes']) - separability[most_mixed_class]['pure_nodes'] if separability[most_mixed_class]['nodes'] else 0
            f.write(f"⚠️  Most mixed class: {most_mixed_class} ({mixed_count} mixed nodes)\n")
        
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
    """Visualize GSOM nodes, clusters, purity, outliers, and boundary points as separate PDF files."""
    
    base_name = file_name.replace('.pdf', '')
    
    # ========== PLOT 1: GSOM Map with Clusters, Entropy, and Boundary Points ==========
    plt.figure(figsize=(12, 10))
    
    max_entropy = max(entropy_dict.values(), default=1) if entropy_dict else 1
    colors_cluster = ['green', 'red', 'black', 'cyan']
    
    # Plot all nodes
    plt.scatter(gsom.node_coordinate[:gsom.node_count, 0], gsom.node_coordinate[:gsom.node_count, 1], 
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
        plt.scatter(x, y, c=color, s=size, alpha=0.5 + 0.5 * (entropy / max_entropy), 
                    marker='D' if node_idx in gsom.node_labels['output'].values else 'o')
        plt.text(x, y, f"{node_idx}\nE={entropy:.2f}", fontsize=6)
    
    # Plot skeleton connections
    _, skeleton_connections, _, pos_edges = gsom.build_skeleton(data, weight_columns)
    from collections import Counter
    overlaps = Counter((i, j) for i, j in skeleton_connections)
    q3 = np.percentile(list(overlaps.values()), 75) if overlaps.values() else 0
    
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
        plt.plot([x1, x2], [y1, y2], color=color, linestyle=line_style, alpha=alpha)
    
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
        plt.scatter(x, y, c='none', s=100, edgecolors='purple', linewidth=2, 
                   label='Boundary Node' if node_idx == list(boundary_nodes)[0] else "")
    
    # Highlight boundary points
    for point in boundary_points:
        node_idx = point['node']
        x, y = gsom.node_coordinate[node_idx]
        plt.scatter(x, y, c='magenta', s=50, marker='*', edgecolors='black', 
                   label='Boundary Point' if point == boundary_points[0] else "")
    
    # Add cluster legend and other legends
    if legend_elements:
        plt.legend(handles=legend_elements + [plt.Line2D([0], [0], marker='o', color='w', 
                                                        markerfacecolor='purple', markersize=8, 
                                                        label='Boundary Node', markeredgecolor='purple'),
                                            plt.Line2D([0], [0], marker='*', color='w', 
                                                        markerfacecolor='magenta', markersize=8, 
                                                        label='Boundary Point', markeredgecolor='black')],
                  bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.title(f"GSOM Map with Named Clusters, Entropy, and Boundary Points (Region Entropy: {region_entropy:.2f})")
    plt.tight_layout()
    gsom_map_file = f"{base_name}_gsom_map.pdf"
    plt.savefig(gsom_map_file, bbox_inches='tight')
    print(f"GSOM Map plot saved as {gsom_map_file}")
    plt.close()
    
    # ========== PLOT 2: Confusion Matrix with cluster names ==========
    plt.figure(figsize=(8, 6))
    
    _, confusion, all_labels, _, cluster_names = gsom.compute_cluster_purity(data, label_col, weight_columns)
    
    # Filter out empty clusters - only show clusters with actual data
    non_empty_clusters = []
    non_empty_confusion = []
    non_empty_labels = []
    
    for cluster_id in range(len(confusion)):
        if confusion[cluster_id].sum() > 0:  # Only clusters with data
            non_empty_clusters.append(cluster_id)
            non_empty_confusion.append(confusion[cluster_id])
            # Use meaningful cluster names from cluster_names dict
            cluster_label = cluster_names.get(cluster_id, f'Cluster {cluster_id+1}')
            non_empty_labels.append(cluster_label)
    
    if non_empty_confusion:
        # Convert to numpy array for plotting
        filtered_confusion = np.array(non_empty_confusion)
        
        sns.heatmap(filtered_confusion, annot=True, fmt='.0f', cmap='Blues', 
                    xticklabels=all_labels, yticklabels=non_empty_labels)
        plt.title("Confusion Matrix with Named Clusters")
        plt.xlabel("Disease Status")
        plt.ylabel("Named Clusters")
        plt.tight_layout()
        confusion_matrix_file = f"{base_name}_confusion_matrix.pdf"
        plt.savefig(confusion_matrix_file, bbox_inches='tight')
        print(f"Confusion Matrix plot saved as {confusion_matrix_file}")
    else:
        print("No clusters with data found for confusion matrix")
    plt.close()
    
    # ========== PLOT 3: Box Plot for Feature Differences of All Boundary Points ==========
    if boundary_points:
        plt.figure(figsize=(16, 8))
        
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
        
        # Create box plot data - limit to first 20 features for readability
        max_features = min(20, len(feature_names))
        selected_features = feature_names[:max_features]
        box_data = [feature_data[feature] for feature in selected_features]
        
        # Create box plot
        bp = plt.boxplot(box_data, tick_labels=selected_features, patch_artist=True)
        
        # Color the boxes
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
        for patch, color in zip(bp['boxes'], colors * (len(bp['boxes']) // len(colors) + 1)):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        plt.ylabel("Absolute Feature Difference")
        plt.title(f"Feature Differences Distribution for All Boundary Points (n={len(boundary_points)}) - First {max_features} Features")
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = []
        for i, feature in enumerate(selected_features):
            data = feature_data[feature]
            if data:
                median_val = np.median(data)
                mean_val = np.mean(data)
                stats_text.append(f"{feature}: μ={mean_val:.3f}, M={median_val:.3f}")
        
        # Add text box with statistics
        if stats_text:
            textstr = '\n'.join(stats_text[:10])  # Limit to first 10 for readability
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=8,
                    verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        boundary_features_file = f"{base_name}_boundary_features.pdf"
        plt.savefig(boundary_features_file, bbox_inches='tight')
        print(f"Boundary Features plot saved as {boundary_features_file}")
        plt.close()
        
        # ========== PLOT 4: Additional Feature Analysis for remaining features ==========
        if len(feature_names) > 20:
            plt.figure(figsize=(16, 8))
            
            # Show next 20 features
            start_idx = 20
            end_idx = min(40, len(feature_names))
            remaining_features = feature_names[start_idx:end_idx]
            box_data_remaining = [feature_data[feature] for feature in remaining_features]
            
            bp2 = plt.boxplot(box_data_remaining, tick_labels=remaining_features, patch_artist=True)
            
            # Color the boxes
            for patch, color in zip(bp2['boxes'], colors * (len(bp2['boxes']) // len(colors) + 1)):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            plt.ylabel("Absolute Feature Difference")
            plt.title(f"Feature Differences Distribution - Features {start_idx+1} to {end_idx} (n={len(boundary_points)})")
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            boundary_features_2_file = f"{base_name}_boundary_features_extended.pdf"
            plt.savefig(boundary_features_2_file, bbox_inches='tight')
            print(f"Extended Boundary Features plot saved as {boundary_features_2_file}")
            plt.close()
    
    print(f"\n📊 All plots generated successfully as separate PDF files!")
    print(f"   Base name: {base_name}")
    return [f"{base_name}_gsom_map.pdf", f"{base_name}_confusion_matrix.pdf", 
            f"{base_name}_boundary_features.pdf"]

if __name__ == '__main__':
    np.random.seed(1)
    # Load Crohn's disease dataset
    df = pd.read_csv("crohn.csv")
    
    print("Dataset shape:", df.shape)
    print("Dataset head:", df.head())
    
    # Get feature columns (all genetic loci)
    weight_columns = [col for col in df.columns if col.startswith('loc')]
    print(f"Number of genetic features: {len(weight_columns)}")
    print("Sample feature columns:", weight_columns[:10])
    
    # Prepare training data (genetic features only)
    data_training = df[weight_columns]
    print("Training data shape:", data_training.shape)
    
    # Handle missing values if any
    data_training = data_training.fillna(0)
    
    # Normalize data to 0-1 range for GSOM
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    data_training_scaled = pd.DataFrame(scaler.fit_transform(data_training), columns=weight_columns)
    
    # Create a copy of the original dataframe with scaled features for prediction
    df_scaled = df.copy()
    df_scaled[weight_columns] = scaler.transform(df[weight_columns])
    
    # Train GSOM with parameters adjusted for genetic data
    # Using larger spread factor and initial node size for high-dimensional genetic data
    gsom = GSOM(0.95, len(weight_columns), max_radius=8, initial_node_size=10000)
    gsom.fit(data_training_scaled.to_numpy(), 100, 50)  # Reasonable iterations for complex genetic data
    
    # Custom predict method to ensure correct feature columns
    data_n = df_scaled[weight_columns].to_numpy()
    output_data = df_scaled[["id", "crohn"]].copy()
    out = scipy.spatial.distance.cdist(gsom.node_list[:gsom.node_count], data_n, gsom.distance)
    output_data["output"] = out.argmin(axis=0)
    
    grp_output = output_data.groupby("output")
    output = grp_output["id"].apply(list).reset_index()
    output = output.set_index("output")
    output["crohn"] = grp_output["crohn"].apply(list)
    output = output.reset_index()
    output["hit_count"] = output["id"].apply(lambda x: len(x))
    output["x"] = output["output"].apply(lambda x: gsom.node_coordinate[x, 0])
    output["y"] = output["output"].apply(lambda x: gsom.node_coordinate[x, 1])
    
    # Set the node_labels attribute for the gsom object
    gsom.node_labels = output
    output.to_csv("output_crohn.csv", index=False)
    print("GSOM training completed.")
    print("Output shape:", output.shape)
    print("Node Count:", gsom.node_count)
    
    # Compute cluster purity using original numeric labels
    entropy_dict, confusion, all_labels, node_to_cluster, cluster_names = gsom.compute_cluster_purity(df_scaled, "crohn", weight_columns)
    # print("Node Entropy:")
    # for node_idx, entropy in entropy_dict.items():
    #     print(f"Node {node_idx}: Entropy = {entropy:.2f}")
    
    print("\nCluster Names:")
    for cluster_id, cluster_name in cluster_names.items():
        print(f"{cluster_name}")
    
    # 🧠 COMPREHENSIVE ANALYSIS: Understanding Genetic Data Structure
    print("\n" + "="*60)
    print("🧠 PERFORMING COMPREHENSIVE GENETIC STRUCTURE ANALYSIS")
    print("="*60)
    
    # Generate comprehensive analysis report
    separability, mixing, regions, quantization, connectivity, paths = comprehensive_analysis_report(
        gsom, df_scaled, "crohn", weight_columns, "crohn_comprehensive_analysis.txt")
    
    # Print key insights to console
    print("\n🔍 KEY GENETIC INSIGHTS:")
    print("-" * 35)
    
    # Separability insights for Crohn's disease
    control_analysis = separability.get(0, {})  # Control group
    crohn_analysis = separability.get(2, {})    # Crohn's group
    
    if control_analysis and control_analysis['nodes']:
        control_purity = control_analysis['pure_nodes'] / len(control_analysis['nodes'])
        print(f"🧬 Control group separability: {control_purity:.1%}")
    
    if crohn_analysis and crohn_analysis['nodes']:
        crohn_purity = crohn_analysis['pure_nodes'] / len(crohn_analysis['nodes'])
        print(f"🩺 Crohn's group separability: {crohn_purity:.1%}")
    
    # Mixing analysis
    total_mixed_nodes = len(mixing)
    print(f"🔄 Mixed nodes found: {total_mixed_nodes} (genetic overlap between groups)")
    
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
    
    print(f"\n💡 UNDERSTANDING: This analysis reveals the GENETIC STRUCTURE")
    print(f"   showing WHERE and WHY genetic patterns lead to disease classification.")
    
    # Original analyses adapted for Crohn's dataset
    outliers = gsom.detect_outliers(df_scaled, "crohn", weight_columns, threshold=2.0)
    center_node = output.loc[output['hit_count'].idxmax(), 'output']
    region_entropy, region_nodes, deviant_points = gsom.analyze_region(center_node, radius=2.0, data=df_scaled, label_col="crohn", weight_columns=weight_columns)
    
    # Analyze boundary points with adaptive threshold based on data characteristics
    print("\n🔍 Auto-determining optimal boundary threshold...")
    
    # Use adaptive threshold calculation
    optimal_threshold, threshold_info = gsom.auto_determine_boundary_threshold(df_scaled, weight_columns, "crohn")
    
    print(f"📊 Dataset Analysis:")
    print(f"  • Features: {threshold_info['n_features']}")
    print(f"  • Samples: {threshold_info['n_samples']}")
    print(f"  • Cluster separation ratio: {threshold_info['separation_ratio']:.2f}")
    print(f"  • Dimensionality factor: {threshold_info['dimensionality_factor']:.3f}")
    print(f"  • Inter-cluster distance: {threshold_info['inter_cluster_mean']:.3f}")
    print(f"  • Intra-cluster distance: {threshold_info['intra_cluster_mean']:.3f}")
    
    # Test different thresholds for comparison (optional)
    print(f"\n📋 Threshold Comparison:")
    comparison_thresholds = [0.05, 0.1, optimal_threshold, 0.2, 0.3]
    for thresh in comparison_thresholds:
        test_boundary_points, _, _, _ = gsom.identify_boundary_points(df_scaled, weight_columns, "crohn", distance_threshold=thresh)
        percentage = len(test_boundary_points) / len(df_scaled) * 100
        marker = "✅ AUTO" if abs(thresh - optimal_threshold) < 0.001 else "  "
        print(f"  {marker} Threshold {thresh:.3f}: {len(test_boundary_points)} points ({percentage:.1f}%)")
    
    # Use the automatically determined optimal threshold
    boundary_points, boundary_nodes, node_to_cluster, clusters = gsom.identify_boundary_points(df_scaled, weight_columns, "crohn", distance_threshold=optimal_threshold)
    
    print(f"\n✅ Using auto-determined threshold {optimal_threshold:.3f}: {len(boundary_points)} boundary points ({threshold_info['boundary_percentage']:.1%} of data)")
    
    # Save boundary analysis files
    with open("boundary_points_analysis_crohn.txt", "w", encoding="utf-8") as f:
        f.write("GSOM Boundary Points Analysis - Crohn's Disease\n")
        f.write("=" * 50 + "\n\n")
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
            # Show only first 10 genetic features to avoid overwhelming output
            for i, (feature, diff) in enumerate(zip(weight_columns[:10], point['feature_diff_node'][:10])):
                f.write(f"    {feature}: {diff:.4f}\n")
            if len(weight_columns) > 10:
                f.write(f"    ... and {len(weight_columns) - 10} more genetic features\n")
            f.write(f"  Distances to Cluster Centroids: {[f'{d:.4f}' for d in point['distances_to_centroids']]}\n")
            f.write("-" * 60 + "\n")
    
    print(f"\nBoundary points analysis saved to 'boundary_points_analysis_crohn.txt'")
    
    # Visualize results with enhanced understanding for Crohn's data
    plot_analysis(gsom, output, entropy_dict, clusters, node_to_cluster, outliers, region_entropy, 
                  region_nodes, deviant_points, boundary_points, boundary_nodes, df_scaled, "crohn", weight_columns, 
                  cluster_names)
    
    print("\n" + "="*60)
    print("✅ COMPLETE: Genetic analysis focuses on UNDERSTANDING Crohn's data structure")
    print("   📄 Main insights in: crohn_comprehensive_analysis.txt")
    print("   📊 Visualization: gsom_boundary_analysis_crohn.pdf")
    print("   📋 Boundary analysis: boundary_points_analysis_crohn.txt")
    print("="*60)
