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

    def separate_clusters(self, data, weight_columns, max_clusters=7):
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
        
        clusters, _, _, _ = self.separate_clusters(data, weight_columns, max_clusters=7)
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
        for cluster_id in range(len(clusters[-1])):
            cluster_label_counts = confusion[cluster_id]
            if cluster_label_counts.sum() > 0:
                dominant_label_idx = np.argmax(cluster_label_counts)
                dominant_label = all_labels[dominant_label_idx]
                dominant_count = cluster_label_counts[dominant_label_idx]
                total_count = cluster_label_counts.sum()
                purity = dominant_count / total_count
                cluster_names[cluster_id] = f"Cluster {cluster_id + 1}: Type {dominant_label} ({purity:.1%})"
            else:
                cluster_names[cluster_id] = f"Cluster {cluster_id + 1}: Empty"
        
        return entropy_dict, confusion, all_labels, node_to_cluster, cluster_names

    def detect_outliers(self, data, label_col, weight_columns, threshold=2.0):
        outliers = []
        data_n = data[weight_columns].to_numpy()
        
        for _, row in self.node_labels.iterrows():
            node_idx = row['output']
            sample_ids = row['Name']  # Now contains animal names
            labels = row[label_col]
            node_weights = self.node_list[node_idx]
            
            sample_indices = data[data['Name'].isin(sample_ids)].index
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
                sample_ids = node_row['Name'].iloc[0]  # Now contains animal names
                labels = node_row[label_col].iloc[0]
                node_weights = self.node_list[node_idx]
                sample_indices = data[data['Name'].isin(sample_ids)].index
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

    def identify_boundary_points(self, data, weight_columns, label_col, max_clusters=7, distance_threshold=0.5):
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
            sample_ids = row['Name']  # Now contains animal names
            labels = row[label_col]
            node_weights = self.node_list[node_idx]
            sample_indices = data[data['Name'].isin(sample_ids)].index
            
            distances_to_centroids = scipy.spatial.distance.cdist(
                data_n[sample_indices], np.array(cluster_centroids), self.distance
            )
            
            for i, sample_id in enumerate(sample_ids):
                distances = distances_to_centroids[i]
                min_dist = np.min(distances)
                second_min_dist = np.min(distances[distances > min_dist])
                
                if second_min_dist - min_dist < distance_threshold:
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

def plot_analysis(gsom, output, entropy_dict, clusters, node_to_cluster, outliers, region_entropy, 
                 region_nodes, deviant_points, boundary_points, boundary_nodes, data, label_col, weight_columns, 
                 cluster_names, file_name="gsom_boundary_analysis_zoo.pdf"):
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    
    max_entropy = max(entropy_dict.values(), default=1) if entropy_dict else 1
    colors_cluster = ['green', 'red', 'black', 'cyan', 'magenta', 'yellow', 'blue']
    
    ax1.scatter(gsom.node_coordinate[:gsom.node_count, 0], gsom.node_coordinate[:gsom.node_count, 1], 
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
        ax1.scatter(x, y, c=color, s=size, alpha=0.5 + 0.5 * (entropy / max_entropy), 
                    marker='D' if node_idx in gsom.node_labels['output'].values else 'o')
        ax1.text(x, y, f"{node_idx}\nE={entropy:.2f}", fontsize=6)
    
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
    
    legend_elements = []
    for cluster_id, cluster_name in cluster_names.items():
        if cluster_id < len(colors_cluster):
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=colors_cluster[cluster_id], 
                                            markersize=8, label=cluster_name))
    
    for node_idx in boundary_nodes:
        x, y = gsom.node_coordinate[node_idx]
        ax1.scatter(x, y, c='none', s=100, edgecolors='purple', linewidth=2, label='Boundary Node' if node_idx == list(boundary_nodes)[0] else "")
    
    for point in boundary_points:
        node_idx = point['node']
        x, y = gsom.node_coordinate[node_idx]
        ax1.scatter(x, y, c='magenta', s=50, marker='*', edgecolors='black', label='Boundary Point' if point == boundary_points[0] else "")
    
    if legend_elements:
        ax1.legend(handles=legend_elements + [plt.Line2D([0], [0], marker='o', color='w', 
                                                        markerfacecolor='purple', markersize=8, 
                                                        label='Boundary Node', markeredgecolor='purple'),
                                            plt.Line2D([0], [0], marker='*', color='w', 
                                                        markerfacecolor='magenta', markersize=8, 
                                                        label='Boundary Point', markeredgecolor='black')],
                  bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    ax1.set_title(f"GSOM Map with Named Clusters, Entropy, and Boundary Points (Region Entropy: {region_entropy:.2f})")
    
    _, confusion, all_labels, _, _ = gsom.compute_cluster_purity(data, label_col, weight_columns)
    cluster_labels = [cluster_names.get(i, f'Cluster {i+1}') for i in range(len(confusion))]
    sns.heatmap(confusion, annot=True, fmt='.0f', cmap='Blues', ax=ax2, 
                xticklabels=all_labels, yticklabels=cluster_labels)
    ax2.set_title("Confusion Matrix with Named Clusters")
    ax2.set_xlabel("Animal Type")
    ax2.set_ylabel("Named Clusters")
    
    if boundary_points:
        # Create a mapping for display names
        feature_display_names = {
            'w1': 'hair', 'w2': 'feathers', 'w3': 'eggs', 'w4': 'milk', 'w4.1': 'airborne',
            'w5': 'aquatic', 'w6': 'predator', 'w7': 'toothed', 'w8': 'backbone', 'w9': 'breathes',
            'w10': 'venomous', 'w11': 'fins', 'w12': 'legs', 'w13': 'tail', 'w14': 'domestic', 'w15': 'catsize'
        }
        
        # Select top 5 features by variance for plotting
        variances = data[weight_columns].var()
        top_features = variances.nlargest(5).index.tolist()
        top_feature_names = [feature_display_names.get(f, f) for f in top_features]
        n_features = len(top_features)
        n_points = min(len(boundary_points), 5)
        bar_width = 0.15
        x = np.arange(n_features)
        
        for i, point in enumerate(boundary_points[:n_points]):
            diffs = point['feature_diff_node'][data[weight_columns].columns.isin(top_features)]
            cluster_name = cluster_names.get(point['cluster'], f"Cluster {point['cluster'] + 1}")
            ax3.bar(x + i * bar_width, diffs, bar_width, 
                   label=f"Sample {point['sample_id']} ({cluster_name})")
        
        ax3.set_xticks(x + bar_width * (n_points - 1) / 2)
        ax3.set_xticklabels(top_feature_names, rotation=45)
        ax3.set_ylabel("Absolute Feature Difference")
        ax3.set_title("Feature Differences for Boundary Points (vs. Assigned Node)")
        ax3.legend()
    
    plt.tight_layout()
    plt.savefig(file_name, bbox_inches='tight')
    print(f"Analysis plot saved as {file_name}")
    plt.close()

if __name__ == '__main__':
    np.random.seed(1)
    # Load Zoo dataset
    df = pd.read_csv('zoo.txt')
    
    # Use Name column as identifier (instead of creating Id column)
    # This makes results more interpretable
    
    # Use original column names as feature columns
    weight_columns = ['w1', 'w2', 'w3', 'w4', 'w4.1', 'w5', 'w6', 'w7', 'w8', 'w9', 'w10', 'w11', 'w12', 'w13', 'w14', 'w15']
    
    # Normalize features
    scaler = MinMaxScaler()
    df[weight_columns] = scaler.fit_transform(df[weight_columns])
    
    print("Dataset shape:", df.shape)
    data_training = df[weight_columns]
    print("Training data head:", data_training.head())
    print("Training data shape:", data_training.shape)
    
    # Train GSOM
    gsom = GSOM(spred_factor=0.83, dimensions=len(weight_columns), max_radius=6, initial_node_size=1000)
    gsom.fit(data_training.to_numpy(), 100, 50)
    output = gsom.predict(df, "Name", "label", weight_columns)
    output.to_csv("output_zoo.csv", index=False)
    print("GSOM training completed.")
    print("Output shape:", output.shape)
    print("Node Count:", gsom.node_count)
    
    # Compute cluster purity
    entropy_dict, confusion, all_labels, node_to_cluster, cluster_names = gsom.compute_cluster_purity(df, "label", weight_columns)
    print("Node Entropy:")
    for node_idx, entropy in entropy_dict.items():
        print(f"Node {node_idx}: Entropy = {entropy:.2f}")
    
    print("\nCluster Names:")
    for cluster_id, cluster_name in cluster_names.items():
        print(f"{cluster_name}")
    
    # Detect outliers
    outliers = gsom.detect_outliers(df, "label", weight_columns, threshold=2.0)
    print("\nOutliers:")
    for outlier in outliers:
        print(f"Node {outlier['node']}:")
        print(f"  Distance Outliers: {outlier['distance_outliers']}")
        print(f"  Minority Outliers: {outlier['minority_outliers']}")
    
    # Analyze a region
    center_node = output.loc[output['hit_count'].idxmax(), 'output']
    region_entropy, region_nodes, deviant_points = gsom.analyze_region(center_node, radius=2.0, data=df, label_col="label", weight_columns=weight_columns)
    print(f"\nRegion Analysis (Center Node: {center_node}, Radius: 2.0):")
    print(f"  Region Entropy: {region_entropy:.2f}")
    print(f"  Region Nodes: {region_nodes}")
    print("  Deviant Points:")
    for deviant in deviant_points:
        print(f"    Node {deviant['node']}, Sample Name: {deviant['sample_id']}, Label: {deviant['label']}, Distance: {deviant['distance']:.2f}")
    
    # Identify boundary points
    boundary_points, boundary_nodes, node_to_cluster, clusters = gsom.identify_boundary_points(df, weight_columns, "label", max_clusters=7, distance_threshold=0.5)
    print("\nBoundary Points:")
    
    # Save boundary points analysis
    with open("boundary_points_analysis_zoo.txt", "w") as f:
        # Create feature display names mapping
        feature_display_names = {
            'w1': 'hair', 'w2': 'feathers', 'w3': 'eggs', 'w4': 'milk', 'w4.1': 'airborne',
            'w5': 'aquatic', 'w6': 'predator', 'w7': 'toothed', 'w8': 'backbone', 'w9': 'breathes',
            'w10': 'venomous', 'w11': 'fins', 'w12': 'legs', 'w13': 'tail', 'w14': 'domestic', 'w15': 'catsize'
        }
        
        f.write("GSOM Boundary Points Analysis\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total Boundary Points Found: {len(boundary_points)}\n\n")
        
        f.write("CLUSTER NAMES:\n")
        for cluster_id, cluster_name in cluster_names.items():
            f.write(f"{cluster_name}\n")
        f.write("\n")
        
        for point in boundary_points:
            cluster_name = cluster_names.get(point['cluster'], f"Cluster {point['cluster'] + 1}")
            animal_name = point['sample_id']  # sample_id now contains the animal name
            f.write(f"Animal: {animal_name}, Node: {point['node']}, {cluster_name}, Label: {point['label']}\n")
            f.write(f"  Feature Differences from Node {point['node']}:\n")
            for i, feature in enumerate(weight_columns):
                display_name = feature_display_names.get(feature, feature)
                diff = point['feature_diff_node'][i]
                f.write(f"    {display_name}: {diff:.4f}\n")
            
            f.write(f"  Feature Differences from Other Clusters:\n")
            for j, diffs in enumerate(point['feature_diff_other_clusters']):
                f.write(f"    Other Cluster {j+1}:\n")
                for i, feature in enumerate(weight_columns):
                    display_name = feature_display_names.get(feature, feature)
                    diff = diffs[i]
                    f.write(f"      {display_name}: {diff:.4f}\n")
            
            f.write(f"  Distances to Cluster Centroids: {[f'{d:.4f}' for d in point['distances_to_centroids']]}\n")
            f.write("-" * 60 + "\n")
    
    print(f"\nBoundary points analysis saved to 'boundary_points_analysis_zoo.txt'")
    
    # Save comprehensive analysis report
    with open("gsom_complete_analysis_zoo.txt", "w") as f:
        f.write("GSOM Complete Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("DATASET INFORMATION\n")
        f.write("-" * 20 + "\n")
        f.write(f"Dataset shape: {df.shape}\n")
        f.write(f"Training data shape: {data_training.shape}\n")
        f.write(f"Feature columns: {weight_columns}\n")
        f.write(f"Node count: {gsom.node_count}\n\n")
        
        f.write("CLUSTER NAMES\n")
        f.write("-" * 20 + "\n")
        for cluster_id, cluster_name in cluster_names.items():
            f.write(f"{cluster_name}\n")
        f.write("\n")
        
        f.write("NODE ENTROPY ANALYSIS\n")
        f.write("-" * 20 + "\n")
        for node_idx, entropy in entropy_dict.items():
            cluster_id = node_to_cluster.get(node_idx, -1)
            cluster_name = cluster_names.get(cluster_id, f"Cluster {cluster_id + 1}")
            f.write(f"Node {node_idx} ({cluster_name}): Entropy = {entropy:.4f}\n")
        f.write("\n")
        
        f.write("OUTLIER DETECTION\n")
        f.write("-" * 20 + "\n")
        for outlier in outliers:
            f.write(f"Node {outlier['node']}:\n")
            f.write(f"  Distance Outliers: {outlier['distance_outliers']}\n")
            f.write(f"  Minority Outliers: {outlier['minority_outliers']}\n")
        f.write("\n")
        
        f.write("REGION ANALYSIS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Center Node: {center_node}, Radius: 2.0\n")
        f.write(f"Region Entropy: {region_entropy:.4f}\n")
        f.write(f"Region Nodes: {region_nodes}\n")
        f.write("Deviant Points:\n")
        for deviant in deviant_points:
            animal_name = deviant['sample_id']  # sample_id now contains the animal name
            f.write(f"  Node {deviant['node']}, Animal: {animal_name}, Label: {deviant['label']}, Distance: {deviant['distance']:.4f}\n")
        f.write("\n")
        
        f.write("BOUNDARY POINTS SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total Boundary Points: {len(boundary_points)}\n")
        f.write(f"Boundary Nodes: {list(boundary_nodes)}\n")
        f.write(f"Number of Clusters: {len(clusters[-1])}\n")
    
    print(f"Complete analysis report saved to 'gsom_complete_analysis_zoo.txt'")
    
    # Visualize results
    plot_analysis(gsom, output, entropy_dict, clusters, node_to_cluster, outliers, region_entropy, 
                  region_nodes, deviant_points, boundary_points, boundary_nodes, df, "label", weight_columns, cluster_names)
    
    print("Complete")