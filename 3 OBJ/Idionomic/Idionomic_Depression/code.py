"""
Enhanced GSOM Analysis: Regional Understanding Beyond Classification

This implementation focuses on understanding data structure through spatial organization:
- REGIONS vs CLUSTERS: Spatial groupings that may contain multiple classes
- QUANTIZE DIFFERENCE: Measures how well nodes represent their data (pure vs mixed)
- BOUNDARY ANALYSIS: Identifies where and why classes naturally overlap
- OUTLIER EXPLANATIONS: Understanding why outliers occur, not just detecting them

Key Concepts:
- No dataset is 100% separable (no perfect purity)
- Node weights are meaningful for comparison with original features
- Regions like "provinces" - some pure, others mixed
- Skeleton modeling provides explainability beyond black-box classification
- Spatial structure reveals relationships between data points

Example: Iris dataset
- Setosa: Linearly separable (pure region expected)
- Versicolor/Virginica: Known overlap (mixed regions expected)
- Boundary points show natural class confusion areas
"""

import numpy as np
import pandas as pd
from scipy.spatial import distance
import scipy
from tqdm import tqdm
import math
from bigtree import Node, findall, find, tree_to_dot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import seaborn as sns
import networkx as nx
import pydot
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

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
        # Use provided weight_columns or derive from data
        if weight_columns is None:
            weight_columns = list(data.columns.values)
            output_columns = [index_col]
            if label_col:
                weight_columns.remove(label_col)
                output_columns.append(label_col)
            weight_columns.remove(index_col)
        else:
            # Use the provided weight_columns directly
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
        """
        Compute cluster purity and region analysis.
        Regions are spatial groupings in GSOM that may contain multiple clusters.
        Pure regions have homogeneous class labels, mixed regions show overlapping classes.
        """
        entropy_dict = {}
        confusion_data = []
        node_to_cluster = {}
        
        clusters, _, _, _ = self.separate_clusters(data, weight_columns, max_clusters=self.max_clusters)
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
        
        cluster_names = {}
        region_analysis = {}
        for cluster_id in range(len(clusters[-1])):
            cluster_label_counts = confusion[cluster_id]
            if cluster_label_counts.sum() > 0:
                dominant_label_idx = np.argmax(cluster_label_counts)
                dominant_label = all_labels[dominant_label_idx]
                dominant_count = cluster_label_counts[dominant_label_idx]
                total_count = cluster_label_counts.sum()
                purity = dominant_count / total_count
                
                # Determine region type based on purity
                if purity >= 0.9:
                    region_type = "Pure"
                elif purity >= 0.7:
                    region_type = "Mostly Pure"
                elif purity >= 0.5:
                    region_type = "Mixed"
                else:
                    region_type = "Highly Mixed"
                
                cluster_names[cluster_id] = f"Region {cluster_id + 1}: {dominant_label} ({purity:.1%}, {region_type})"
                region_analysis[cluster_id] = {
                    'dominant_label': dominant_label,
                    'purity': purity,
                    'type': region_type,
                    'total_samples': int(total_count),
                    'label_distribution': {all_labels[i]: int(cluster_label_counts[i]) for i in range(len(all_labels)) if cluster_label_counts[i] > 0}
                }
            else:
                cluster_names[cluster_id] = f"Region {cluster_id + 1}: Empty"
                region_analysis[cluster_id] = {'type': 'Empty', 'purity': 0.0}
        
        # Calculate classification accuracy
        correct_predictions = np.sum(np.diag(confusion))
        total_predictions = np.sum(confusion)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return entropy_dict, confusion, all_labels, node_to_cluster, cluster_names, accuracy, region_analysis

    def detect_outliers(self, data, label_col, weight_columns, index_col='Id', threshold=2.0):
        """
        Enhanced outlier detection with explanations.
        Identifies why outliers occur: distance-based (spatial) or label-based (classification).
        
        Distance outliers: Points far from their assigned node (spatial anomalies)
        Label outliers: Points with minority labels in their node (classification anomalies)
        Mixed outliers: Points that are both spatially and label-wise different
        """
        outliers = []
        data_n = data[weight_columns].to_numpy()
        
        for _, row in self.node_labels.iterrows():
            node_idx = row['output']
            sample_ids = row[index_col]
            labels = row[label_col]
            node_weights = self.node_list[node_idx]
            
            sample_indices = data[data[index_col].isin(sample_ids)].index
            distances = scipy.spatial.distance.cdist(
                data_n[sample_indices], node_weights.reshape(1, -1), self.distance
            ).flatten()
            
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            
            # Distance-based outliers (spatial anomalies)
            distance_outliers = [
                (sample_ids[i], distances[i], "Far from node center - spatial anomaly")
                for i, dist in enumerate(distances)
                if dist > mean_dist + threshold * std_dist
            ]
            
            # Label-based outliers (classification anomalies)
            label_counts = pd.Series(labels).value_counts()
            majority_label = label_counts.idxmax()
            minority_threshold = 0.2  # Consider minority if less than 20% of node
            
            label_outliers = []
            for i, label in enumerate(labels):
                if label != majority_label:
                    proportion = label_counts[label] / len(labels)
                    if proportion < minority_threshold:
                        explanation = f"Minority class in region - potential class overlap"
                        if dataset_name == 'iris' and majority_label in ['Iris-versicolor', 'Iris-virginica'] and label in ['Iris-versicolor', 'Iris-virginica']:
                            explanation += " (Known Versicolor-Virginica confusion)"
                        label_outliers.append((sample_ids[i], label, explanation))
            
            # Mixed outliers (both spatial and label anomalies)
            mixed_outliers = []
            distance_outlier_ids = [x[0] for x in distance_outliers]
            label_outlier_ids = [x[0] for x in label_outliers]
            mixed_ids = set(distance_outlier_ids) & set(label_outlier_ids)
            
            for sample_id in mixed_ids:
                mixed_outliers.append((sample_id, "Both spatial and classification anomaly - needs investigation"))
            
            outliers.append({
                'node': node_idx,
                'total_samples': len(sample_ids),
                'distance_outliers': distance_outliers,
                'label_outliers': label_outliers,
                'mixed_outliers': mixed_outliers,
                'node_purity': label_counts.max() / len(labels),
                'spatial_compactness': 1 / (1 + std_dist)  # Higher means more compact
            })
        
        return outliers

    def analyze_region(self, center_node_idx, radius, data, label_col, weight_columns, index_col='Id'):
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
                sample_ids = node_row[index_col].iloc[0]
                labels = node_row[label_col].iloc[0]
                node_weights = self.node_list[node_idx]
                sample_indices = data[data[index_col].isin(sample_ids)].index
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

    def identify_boundary_points(self, data, weight_columns, label_col, index_col='Id', max_clusters=3, distance_threshold=0.5):
        """
        Enhanced boundary point identification with explanations.
        Boundary points show where different classes mix and help understand:
        - Which features cause class confusion
        - Where natural class boundaries exist
        - Why certain samples are misclassified
        """
        clusters, segments, _, _ = self.separate_clusters(data, weight_columns, max_clusters)
        node_to_cluster = {}
        for cluster_id, cluster in enumerate(clusters[-1]):
            for node_idx in cluster:
                node_to_cluster[node_idx] = cluster_id
        
        boundary_nodes = set()
        for i, j, dist in segments:
            if node_to_cluster.get(i, -1) != node_to_cluster.get(j, -1):
                boundary_nodes.add(i)
                boundary_nodes.add(j)
        
        boundary_points = []
        data_n = data[weight_columns].to_numpy()
        cluster_centroids = []
        
        for cluster_id in range(len(clusters[-1])):
            cluster_nodes = clusters[-1][cluster_id]
            cluster_weights = np.mean([self.node_list[node_idx] for node_idx in cluster_nodes], axis=0)
            cluster_centroids.append(cluster_weights)
        
        for _, row in self.node_labels.iterrows():
            node_idx = row['output']
            sample_ids = row[index_col]
            labels = row[label_col]
            node_weights = self.node_list[node_idx]
            sample_indices = data[data[index_col].isin(sample_ids)].index
            
            distances_to_centroids = scipy.spatial.distance.cdist(
                data_n[sample_indices], np.array(cluster_centroids), self.distance
            )
            
            for i, sample_id in enumerate(sample_ids):
                distances = distances_to_centroids[i]
                min_dist = np.min(distances)
                second_min_dist = np.min(distances[distances > min_dist]) if len(distances[distances > min_dist]) > 0 else min_dist * 2
                
                # Enhanced boundary detection
                if second_min_dist - min_dist < distance_threshold:
                    sample_features = data_n[sample_indices[i]]
                    feature_diff_node = np.abs(sample_features - node_weights)
                    
                    # Find most discriminative features
                    feature_importance = np.argsort(feature_diff_node)[::-1]
                    top_discriminative_features = [weight_columns[idx] for idx in feature_importance[:3]]
                    
                    # Determine why this is a boundary point
                    explanation = "Close to multiple regions"
                    if dataset_name == 'iris' and labels[i] in ['Iris-versicolor', 'Iris-virginica']:
                        explanation += " - Natural overlap between Versicolor and Virginica"
                    elif node_idx in boundary_nodes:
                        explanation += " - Located at structural boundary between regions"
                    
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
                        'distances_to_centroids': distances,
                        'boundary_score': second_min_dist - min_dist,
                        'explanation': explanation,
                        'top_discriminative_features': top_discriminative_features
                    })
        
        return boundary_points, boundary_nodes, node_to_cluster, clusters

    def compute_quantize_difference(self, data, weight_columns, label_col, index_col='Id'):
        """
        Compute quantize difference to identify pure vs mixed regions.
        This measures how well node weights represent the actual data distribution.
        Low quantize difference indicates pure regions, high indicates mixed regions.
        """
        quantize_differences = {}
        data_n = data[weight_columns].to_numpy()
        
        for _, row in self.node_labels.iterrows():
            node_idx = row['output']
            sample_ids = row[index_col]
            labels = row[label_col]
            node_weights = self.node_list[node_idx]
            
            # Get actual data points assigned to this node
            sample_indices = data[data[index_col].isin(sample_ids)].index
            node_data = data_n[sample_indices]
            
            # Calculate quantize difference as average distance from node weight
            distances = scipy.spatial.distance.cdist(
                node_data, node_weights.reshape(1, -1), self.distance
            ).flatten()
            
            avg_distance = np.mean(distances)
            std_distance = np.std(distances)
            
            # Calculate label heterogeneity
            label_counts = pd.Series(labels).value_counts()
            entropy = 0
            total = len(labels)
            for count in label_counts:
                p = count / total
                if p > 0:
                    entropy -= p * np.log2(p)
            
            # Combine spatial and label information for quantize difference
            quantize_diff = avg_distance * (1 + entropy)  # Higher entropy increases quantize difference
            
            quantize_differences[node_idx] = {
                'quantize_difference': quantize_diff,
                'avg_distance': avg_distance,
                'std_distance': std_distance,
                'entropy': entropy,
                'sample_count': len(sample_ids),
                'unique_labels': len(label_counts),
                'is_pure': entropy == 0,
                'is_mixed': entropy > 0.5
            }
        
        return quantize_differences

def plot_analysis(gsom, output, entropy_dict, clusters, node_to_cluster, outliers, region_entropy, 
                 region_nodes, deviant_points, boundary_points, boundary_nodes, data, label_col, 
                 weight_columns, cluster_names, accuracy, dataset_name, region_analysis, quantize_differences,
                 file_prefix="gsom_regional_analysis"):
    """
    Enhanced visualization focusing on regional analysis and explainability.
    Shows pure vs mixed regions, boundary points, and spatial structure.
    Generates separate PNG files for each plot.
    """
    # Define common parameters
    max_entropy = max(entropy_dict.values(), default=1) if entropy_dict else 1
    n_clusters = len(clusters[-1])
    colors_cluster = ['green', 'red', 'black', 'cyan', 'magenta', 'yellow', 'blue'][:n_clusters]
def plot_analysis(gsom, output, entropy_dict, clusters, node_to_cluster, outliers, region_entropy, 
                 region_nodes, deviant_points, boundary_points, boundary_nodes, data, label_col, 
                 weight_columns, cluster_names, accuracy, dataset_name, region_analysis, quantize_differences,
                 file_prefix="gsom_regional_analysis"):
    """
    Enhanced visualization focusing on regional analysis and explainability.
    Shows pure vs mixed regions, boundary points, and spatial structure.
    Generates separate PNG files for each plot.
    """
    # Define common parameters
    max_entropy = max(entropy_dict.values(), default=1) if entropy_dict else 1
    n_clusters = len(clusters[-1])
    colors_cluster = ['green', 'red', 'black', 'cyan', 'magenta', 'yellow', 'blue'][:n_clusters]
    
    # 1. Main GSOM map with regional analysis
    plt.figure(figsize=(12, 10))
    plt.scatter(gsom.node_coordinate[:gsom.node_count, 0], gsom.node_coordinate[:gsom.node_count, 1], 
                c='gray', s=10, alpha=0.1, label='All Nodes')
    
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
        plt.plot([x1, x2], [y1, y2], color=color, linestyle=line_style, alpha=alpha)
    
    # Enhanced legend with regional information
    legend_elements = []
    for cluster_id, cluster_name in cluster_names.items():
        if cluster_id < len(colors_cluster):
            region_info = region_analysis.get(cluster_id, {})
            region_type = region_info.get('type', 'Unknown')
            color = colors_cluster[cluster_id % len(colors_cluster)]
            
            # Color intensity based on purity
            alpha = 0.3 if region_type == 'Highly Mixed' else 0.6 if region_type == 'Mixed' else 0.9
            
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=color, alpha=alpha,
                                            markersize=8, label=cluster_name))
    
    plt.title(f"GSOM Regional Analysis: Spatial Structure & Class Distribution\n"
             f"Dataset: {dataset_name.capitalize()} | Region Entropy: {region_entropy:.2f} | "
             f"Classification Accuracy: {accuracy:.1%}")
    
    if legend_elements:
        plt.legend(handles=legend_elements + [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', 
                      markersize=8, label='Boundary Node', markeredgecolor='purple'),
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='magenta', 
                      markersize=8, label='Boundary Point', markeredgecolor='black')],
                  bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{file_prefix}_main_map.png", dpi=300, bbox_inches='tight')
    print(f"Main GSOM map saved as {file_prefix}_main_map.png")
    plt.close()

    # 2. Enhanced confusion matrix with regional context
    plt.figure(figsize=(10, 8))
    _, confusion, all_labels, _, _, _, _ = gsom.compute_cluster_purity(data, label_col, weight_columns)
    cluster_labels = [cluster_names.get(i, f'Region {i+1}') for i in range(len(confusion))]
    
    # Create custom colormap for confusion matrix
    confusion_normalized = confusion / confusion.sum(axis=1, keepdims=True)
    confusion_normalized = np.nan_to_num(confusion_normalized)
    
    sns.heatmap(confusion_normalized, annot=confusion.astype(int), fmt='d', cmap='Blues', 
                xticklabels=all_labels, yticklabels=cluster_labels, cbar_kws={'label': 'Proportion'})
    plt.title(f"Regional Confusion Matrix\n(Numbers show sample counts)")
    plt.xlabel("True Class Labels")
    plt.ylabel("GSOM Regions")
    plt.tight_layout()
    plt.savefig(f"{file_prefix}_confusion_matrix.png", dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved as {file_prefix}_confusion_matrix.png")
    plt.close()

    # 3. Regional purity analysis
    plt.figure(figsize=(10, 6))
    purities = [region_analysis[i].get('purity', 0) for i in range(len(clusters[-1]))]
    region_types = [region_analysis[i].get('type', 'Empty') for i in range(len(clusters[-1]))]
    colors_purity = ['darkgreen' if t == 'Pure' else 'lightgreen' if t == 'Mostly Pure' 
                    else 'orange' if t == 'Mixed' else 'red' if t == 'Highly Mixed' else 'gray' 
                    for t in region_types]
    
    bars = plt.bar(range(len(purities)), purities, color=colors_purity, alpha=0.7)
    plt.xlabel("Region ID")
    plt.ylabel("Purity Score")
    plt.title("Regional Purity Analysis\n(Green=Pure, Red=Mixed)")
    plt.ylim(0, 1)
    
    # Add purity values on bars
    for i, (bar, purity) in enumerate(zip(bars, purities)):
        if purity > 0:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{purity:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{file_prefix}_purity_analysis.png", dpi=300, bbox_inches='tight')
    print(f"Purity analysis saved as {file_prefix}_purity_analysis.png")
    plt.close()

    # 4. Quantize difference analysis
    if quantize_differences:
        plt.figure(figsize=(12, 6))
        qd_values = [quantize_differences[node]['quantize_difference'] for node in quantize_differences]
        qd_labels = list(quantize_differences.keys())
        colors_qd = ['red' if quantize_differences[node]['is_mixed'] else 'green' if quantize_differences[node]['is_pure'] else 'yellow' 
                    for node in qd_labels]
        
        plt.bar(range(len(qd_values)), qd_values, color=colors_qd, alpha=0.7)
        plt.xlabel("Node ID")
        plt.ylabel("Quantize Difference")
        plt.title("Quantize Difference by Node\n(Red=Mixed, Green=Pure)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{file_prefix}_quantize_difference.png", dpi=300, bbox_inches='tight')
        print(f"Quantize difference analysis saved as {file_prefix}_quantize_difference.png")
        plt.close()

    # 5. Outlier summary
    plt.figure(figsize=(8, 6))
    total_distance_outliers = sum(len(o['distance_outliers']) for o in outliers)
    total_label_outliers = sum(len(o['label_outliers']) for o in outliers) 
    total_mixed_outliers = sum(len(o['mixed_outliers']) for o in outliers)
    
    outlier_types = ['Distance\nOutliers', 'Label\nOutliers', 'Mixed\nOutliers']
    outlier_counts = [total_distance_outliers, total_label_outliers, total_mixed_outliers]
    colors_outliers = ['skyblue', 'lightcoral', 'purple']
    
    bars_outliers = plt.bar(outlier_types, outlier_counts, color=colors_outliers, alpha=0.7)
    plt.ylabel("Number of Outliers")
    plt.title("Outlier Analysis Summary\nBy Type")
    
    # Add counts on bars
    for bar, count in zip(bars_outliers, outlier_counts):
        if count > 0:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(count), ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{file_prefix}_outlier_summary.png", dpi=300, bbox_inches='tight')
    print(f"Outlier summary saved as {file_prefix}_outlier_summary.png")
    plt.close()

    # 6. Feature differences for boundary points
    if boundary_points:
        plt.figure(figsize=(12, 6))
        variances = data[weight_columns].var()
        top_features = variances.nlargest(5).index.tolist() if len(weight_columns) > 5 else weight_columns
        n_features = len(top_features)
        n_points = min(len(boundary_points), 5)
        bar_width = 0.15
        x = np.arange(n_features)
        
        for i, point in enumerate(boundary_points[:n_points]):
            diffs = point['feature_diff_node'][data[weight_columns].columns.isin(top_features)]
            cluster_name = cluster_names.get(point['cluster'], f"Cluster {point['cluster'] + 1}")
            plt.bar(x + i * bar_width, diffs, bar_width, 
                   label=f"Sample {point['sample_id']} ({cluster_name})")
        
        plt.xticks(x + bar_width * (n_points - 1) / 2, top_features, rotation=45)
        plt.ylabel("Absolute Feature Difference")
        plt.title("Feature Differences for Boundary Points (vs. Assigned Node)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{file_prefix}_boundary_features.png", dpi=300, bbox_inches='tight')
        print(f"Boundary features analysis saved as {file_prefix}_boundary_features.png")
        plt.close()

    print(f"All individual plots saved with prefix: {file_prefix}")

if __name__ == '__main__':
    np.random.seed(1)
    
    # Dataset configuration for Student Depression Dataset
    dataset_configs = {
        'student_depression': {
            'file': 'Student Depression Dataset.csv',
            'index_col': 'id',  # Changed from 'Id' to 'id' to match dataset
            'label_col': 'Depression',
            # Updated to match the actual column names in your dataset
            'weight_columns': [
                'Gender', 'Age', 'Academic Pressure', 'Work Pressure', 'CGPA', 
                'Study Satisfaction', 'Job Satisfaction', 'Work/Study Hours', 
                'Financial Stress', 'Family History of Mental Illness'
            ],
            'dimensions': 10,  # Updated to match the number of weight columns
            'max_clusters': 3,
            'distance': 'euclidean',
            'distance_threshold': 0.5,
            'max_radius': 4
        }
    }

    # Select dataset
    dataset_name = 'student_depression'
    config = dataset_configs[dataset_name]

    # Load dataset
    df = pd.read_csv(config['file'])

    # If 'Id' column does not exist, create it
    if config['index_col'] not in df.columns:
        df[config['index_col']] = range(1, len(df) + 1)

    # Handle categorical variables - encode them to numeric values
    
    # Create a copy for preprocessing
    df_processed = df.copy()
    
    # Encode categorical columns
    categorical_columns = ['Gender', 'Family History of Mental Illness']
    label_encoders = {}
    
    for col in categorical_columns:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            label_encoders[col] = le
    
    # Handle missing values if any
    df_processed = df_processed.fillna(df_processed.mean(numeric_only=True))
    
    # Update the weight columns to only include numeric columns that exist
    available_columns = df_processed.columns.tolist()
    config['weight_columns'] = [col for col in config['weight_columns'] if col in available_columns]
    config['dimensions'] = len(config['weight_columns'])
    
    print(f"Available columns: {available_columns}")
    print(f"Using weight columns: {config['weight_columns']}")
    print(f"Updated dimensions: {config['dimensions']}")

    # Normalize features
    scaler = MinMaxScaler()
    df_processed[config['weight_columns']] = scaler.fit_transform(df_processed[config['weight_columns']])
    
    # Use the processed dataframe
    df = df_processed
    
    # Sample the dataset for testing (use a subset to avoid memory issues)
    # You can remove this line or increase the sample size for full analysis
    df = df.sample(n=1000, random_state=42).reset_index(drop=True)

    print("Dataset shape:", df.shape)
    data_training = df[config['weight_columns']]
    print("Training data head:", data_training.head())
    print("Training data shape:", data_training.shape)

    # Train GSOM with adjusted parameters for the larger dataset
    gsom = GSOM(
        spred_factor=0.95,  # Further increased to reduce growth (much fewer nodes)
        dimensions=config['dimensions'],
        distance=config['distance'],
        max_radius=config['max_radius'],
        initial_node_size=5000  # Increased node capacity even more
    )
    gsom.max_clusters = config['max_clusters']  # Store max_clusters for use in methods
    gsom.fit(data_training.to_numpy(), 30, 15)  # Further reduced iterations
    output = gsom.predict(df, config['index_col'], config['label_col'], config['weight_columns'])
    output_file = f"output_{dataset_name}.csv"
    output.to_csv(output_file, index=False)
    print(f"GSOM training completed. Output saved to {output_file}")
    print("Output shape:", output.shape)
    print("Node Count:", gsom.node_count)

    # Compute cluster purity and regional analysis
    entropy_dict, confusion, all_labels, node_to_cluster, cluster_names, accuracy, region_analysis = gsom.compute_cluster_purity(
        df, config['label_col'], config['weight_columns']
    )

    # Compute quantize differences to identify pure vs mixed regions
    quantize_differences = gsom.compute_quantize_difference(df, config['weight_columns'], config['label_col'], config['index_col'])

    print("=== REGIONAL ANALYSIS ===")
    print("Node Entropy and Regional Information:")
    for node_idx, entropy in entropy_dict.items():
        cluster_id = node_to_cluster.get(node_idx, -1)
        if cluster_id in region_analysis:
            region_info = region_analysis[cluster_id]
            print(f"Node {node_idx}: Entropy = {entropy:.2f}, Region Type = {region_info.get('type', 'Unknown')}")

    print(f"\n=== REGION SUMMARY ===")
    for cluster_id, region_info in region_analysis.items():
        if region_info.get('type') != 'Empty':
            print(f"Region {cluster_id + 1}: {region_info.get('type')} "
                  f"(Purity: {region_info.get('purity', 0):.1%}, "
                  f"Samples: {region_info.get('total_samples', 0)})")
            if 'label_distribution' in region_info:
                print(f"  Label distribution: {region_info['label_distribution']}")

    print(f"\nOverall Classification Accuracy: {accuracy:.1%}")

    print(f"\n=== QUANTIZE DIFFERENCE ANALYSIS ===")
    pure_nodes = [node for node, info in quantize_differences.items() if info['is_pure']]
    mixed_nodes = [node for node, info in quantize_differences.items() if info['is_mixed']]
    print(f"Pure nodes (single class): {pure_nodes}")
    print(f"Mixed nodes (multiple classes): {mixed_nodes}")

    for node_idx, qd_info in quantize_differences.items():
        print(f"Node {node_idx}: QD={qd_info['quantize_difference']:.3f}, "
              f"Entropy={qd_info['entropy']:.2f}, "
              f"Samples={qd_info['sample_count']}, "
              f"Classes={qd_info['unique_labels']}")

    # Enhanced outlier detection with explanations
    outliers = gsom.detect_outliers(df, config['label_col'], config['weight_columns'], config['index_col'], threshold=2.0)
    print(f"\n=== OUTLIER ANALYSIS WITH EXPLANATIONS ===")
    for outlier in outliers:
        print(f"Node {outlier['node']} (Purity: {outlier['node_purity']:.1%}, "
              f"Compactness: {outlier['spatial_compactness']:.2f}):")

        if outlier['distance_outliers']:
            print(f"  Distance Outliers ({len(outlier['distance_outliers'])}):")
            for sample_id, distance, explanation in outlier['distance_outliers']:
                print(f"    Sample {sample_id}: {explanation} (distance: {distance:.3f})")

        if outlier['label_outliers']:
            print(f"  Label Outliers ({len(outlier['label_outliers'])}):")
            for sample_id, label, explanation in outlier['label_outliers']:
                print(f"    Sample {sample_id} ({label}): {explanation}")

        if outlier['mixed_outliers']:
            print(f"  Mixed Outliers ({len(outlier['mixed_outliers'])}):")
            for sample_id, explanation in outlier['mixed_outliers']:
                print(f"    Sample {sample_id}: {explanation}")

    # Analyze a region
    center_node = output.loc[output['hit_count'].idxmax(), 'output']
    region_entropy, region_nodes, deviant_points = gsom.analyze_region(
        center_node, radius=2.0, data=df, label_col=config['label_col'], weight_columns=config['weight_columns'], index_col=config['index_col']
    )
    print(f"\nRegion Analysis (Center Node: {center_node}, Radius: 2.0):")
    print(f"  Region Entropy: {region_entropy:.2f}")
    print(f"  Region Nodes: {region_nodes}")
    print("  Deviant Points:")
    for deviant in deviant_points:
        print(f"    Node {deviant['node']}, Sample ID: {deviant['sample_id']}, Label: {deviant['label']}, Distance: {deviant['distance']:.2f}")

    # Enhanced boundary points with explanations
    boundary_points, boundary_nodes, node_to_cluster, clusters = gsom.identify_boundary_points(
        df, config['weight_columns'], config['label_col'], config['index_col'], max_clusters=config['max_clusters'], 
        distance_threshold=config['distance_threshold']
    )
    print(f"\n=== BOUNDARY POINT ANALYSIS ===")
    print(f"Total Boundary Points Found: {len(boundary_points)}")
    print("These points show where classes naturally overlap and help understand classification confusion.")

    # Group boundary points by explanation
    explanation_groups = {}
    for point in boundary_points:
        explanation = point['explanation']
        if explanation not in explanation_groups:
            explanation_groups[explanation] = []
        explanation_groups[explanation].append(point)

    for explanation, points in explanation_groups.items():
        print(f"\n{explanation} ({len(points)} points):")
        for point in points[:3]:  # Show first 3 examples
            cluster_name = cluster_names.get(point['cluster'], f"Region {point['cluster'] + 1}")
            print(f"  Sample {point['sample_id']} ({point['label']}) in {cluster_name}")
            print(f"    Top discriminative features: {point['top_discriminative_features']}")
            print(f"    Boundary score: {point['boundary_score']:.3f}")
        if len(points) > 3:
            print(f"    ... and {len(points) - 3} more points")

    # Save enhanced boundary points analysis
    boundary_file = f"boundary_points_analysis_{dataset_name}.txt"
    with open(boundary_file, "w") as f:
        f.write(f"GSOM Enhanced Boundary Points Analysis ({dataset_name.capitalize()} Dataset)\n")
        f.write("=" * 60 + "\n\n")
        f.write("UNDERSTANDING CLASS BOUNDARIES AND MIXING PATTERNS\n")
        f.write("-" * 50 + "\n\n")
        f.write(f"Total Boundary Points Found: {len(boundary_points)}\n\n")

        f.write("REGIONAL INFORMATION:\n")
        for cluster_id, region_info in region_analysis.items():
            if region_info.get('type') != 'Empty':
                f.write(f"Region {cluster_id + 1}: {region_info.get('type')} ")
                f.write(f"(Purity: {region_info.get('purity', 0):.1%})\n")
                if 'label_distribution' in region_info:
                    f.write(f"  Classes: {region_info['label_distribution']}\n")
        f.write("\n")

        # Group and explain boundary points
        for explanation, points in explanation_groups.items():
            f.write(f"{explanation.upper()} ({len(points)} points)\n")
            f.write("-" * 40 + "\n")
            for point in points:
                cluster_name = cluster_names.get(point['cluster'], f"Region {point['cluster'] + 1}")
                f.write(f"Sample ID: {point['sample_id']}, Label: {point['label']}, {cluster_name}\n")
                f.write(f"  Boundary Score: {point['boundary_score']:.3f}\n")
                f.write(f"  Key Discriminative Features: {point['top_discriminative_features']}\n")
                f.write(f"  Feature Differences from Node {point['node']}:\n")
                for feature, diff in zip(config['weight_columns'], point['feature_diff_node']):
                    f.write(f"    {feature}: {diff:.3f}\n")
                f.write("-" * 40 + "\n")
            f.write("\n")

    print(f"\nEnhanced boundary points analysis saved to '{boundary_file}'")

    # Save comprehensive regional analysis report
    analysis_file = f"gsom_regional_analysis_{dataset_name}.txt"
    with open(analysis_file, "w") as f:
        f.write(f"GSOM Regional Analysis Report ({dataset_name.capitalize()} Dataset)\n")
        f.write("=" * 60 + "\n\n")
        f.write("UNDERSTANDING DATA STRUCTURE THROUGH SPATIAL ORGANIZATION\n")
        f.write("-" * 55 + "\n\n")

        f.write("DATASET OVERVIEW\n")
        f.write("-" * 20 + "\n")
        f.write(f"Dataset shape: {df.shape}\n")
        f.write(f"Feature columns: {config['weight_columns']}\n")
        f.write(f"Node count: {gsom.node_count}\n")
        f.write(f"Classification accuracy: {accuracy:.1%}\n\n")

        f.write("REGIONAL PURITY ANALYSIS\n")
        f.write("-" * 25 + "\n")
        f.write("Regions are spatial groupings that may contain multiple classes.\n")
        f.write("Pure regions indicate natural class separation, mixed regions show overlap.\n\n")
        for cluster_id, region_info in region_analysis.items():
            if region_info.get('type') != 'Empty':
                f.write(f"Region {cluster_id + 1}: {region_info.get('type')} (Purity: {region_info.get('purity', 0):.1%}, Samples: {region_info.get('total_samples', 0)})\n")
                if 'label_distribution' in region_info:
                    f.write(f"  Label distribution: {region_info['label_distribution']}\n")
        f.write("\n")

        f.write("QUANTIZE DIFFERENCE ANALYSIS\n")
        f.write("-" * 30 + "\n")
        f.write("Quantize difference measures spatial coherence within nodes.\n")
        f.write("Low values indicate tight clustering, high values indicate spreading.\n\n")
        for node_idx, qd_info in quantize_differences.items():
            f.write(f"Node {node_idx}:\n")
            f.write(f"  Quantize Difference: {qd_info['quantize_difference']:.3f}\n")
            f.write(f"  Spatial spread: {qd_info['avg_distance']:.3f} ± {qd_info['std_distance']:.3f}\n")
            f.write(f"  Label entropy: {qd_info['entropy']:.3f}\n")
            f.write(f"  Sample count: {qd_info['sample_count']}\n")
            f.write(f"  Unique classes: {qd_info['unique_labels']}\n")
            f.write(f"  Classification: {'Pure' if qd_info['is_pure'] else 'Mixed' if qd_info['is_mixed'] else 'Intermediate'}\n")
            f.write("\n")

        f.write("OUTLIER EXPLANATIONS\n")
        f.write("-" * 20 + "\n")
        f.write("Understanding why outliers occur provides insights into data structure.\n\n")
        for outlier in outliers:
            f.write(f"Node {outlier['node']} (Purity: {outlier['node_purity']:.1%}, Compactness: {outlier['spatial_compactness']:.2f}):\n")
            if outlier['distance_outliers']:
                f.write(f"  Distance Outliers ({len(outlier['distance_outliers'])}):\n")
                for sample_id, distance, explanation in outlier['distance_outliers']:
                    f.write(f"    Sample {sample_id}: {explanation} (distance: {distance:.3f})\n")
            if outlier['label_outliers']:
                f.write(f"  Label Outliers ({len(outlier['label_outliers'])}):\n")
                for sample_id, label, explanation in outlier['label_outliers']:
                    f.write(f"    Sample {sample_id} ({label}): {explanation}\n")
            if outlier['mixed_outliers']:
                f.write(f"  Mixed Outliers ({len(outlier['mixed_outliers'])}):\n")
                for sample_id, explanation in outlier['mixed_outliers']:
                    f.write(f"    Sample {sample_id}: {explanation}\n")
            f.write("\n")

        f.write("INTERPRETATION NOTES\n")
        f.write("-" * 20 + "\n")
        f.write("- Node weights represent spatial positions and are meaningful for comparison\n")
        f.write("- Regional analysis shows data structure beyond simple classification\n")

    print(f"Comprehensive regional analysis saved to '{analysis_file}'")

    # Create enhanced visualization
    plot_file_prefix = f"gsom_regional_analysis_{dataset_name}"
    plot_analysis(gsom, output, entropy_dict, clusters, node_to_cluster, outliers, region_entropy, 
                  region_nodes, deviant_points, boundary_points, boundary_nodes, df, config['label_col'], 
                  config['weight_columns'], cluster_names, accuracy, dataset_name, region_analysis, 
                  quantize_differences, plot_file_prefix)

    print("Complete")