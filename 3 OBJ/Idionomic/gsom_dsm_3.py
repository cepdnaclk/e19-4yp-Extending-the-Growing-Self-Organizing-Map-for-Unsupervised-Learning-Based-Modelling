import numpy as np
import pandas as pd
from scipy.spatial import distance
import scipy
from tqdm import tqdm
import math
from bigtree import Node, findall, find, tree_to_dot
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import networkx as nx
import pydot
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class GSOM:
    def __init__(self, spred_factor, dimensions, distance='euclidean', initialize='random', learning_rate=0.3,
                 smooth_learning_factor=0.8, max_radius=6, FD=0.1, r=3.8, alpha=0.9, initial_node_size=1000):
        """
        GSOM structure: keep dictionary to x,y coordinates and numpy array to keep weights
        :param spred_factor: spread factor of GSOM graph
        :param dimensions: weight vector dimensions
        :param distance: distance method: support scipy.spatial.distance.cdist
        :param initialize: weight vector initialize method
        :param learning_rate: initial training learning rate of weights
        :param smooth_learning_factor: smooth learning factor to change the initial smooth learning rate from training
        :param max_radius: maximum neighbourhood radius
        :param FD: spread weight value
        :param r: learning rate update value
        :param alpha: learning rate update value
        :param initial_node_size: initial node allocation in memory
        """
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

    def build_skeleton(self, data):
        hit_points = []
        out = scipy.spatial.distance.cdist(self.node_list[:self.node_count], data, self.distance)
        winner_indices = out.argmin(axis=0)
        hit_points = np.unique(winner_indices).tolist()
        
        remaining_connections = []
        pos_edges = []
        for path in self.get_paths():
            for i in range(1, len(path) - 1):
                parent = path[i]
                child = path[i + 1]
                parent_idx = parent.node_number
                child_idx = child.node_number
                if parent_idx >= 0 and child_idx >= 0:
                    remaining_connections.append((parent_idx, child_idx))
                    pos_edges.append((parent_idx, child_idx))
        
        pos_nodes = set()
        for _, j in remaining_connections:
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
                    remaining_connections.append((i, nearest))
                    parent_node = find(self.path_tree, lambda node: node.node_number == nearest)
                    if parent_node:
                        Node(str(i), x=self.node_coordinate[i, 0], y=self.node_coordinate[i, 1],
                             node_number=i, distance=min_dist, parent=parent_node)
        
        junctions = []
        for i, j in remaining_connections:
            if i not in hit_points and j in hit_points:
                junctions.append(i)
            elif j not in hit_points and i in hit_points:
                junctions.append(j)
        
        return hit_points, remaining_connections, junctions, pos_edges

    def separate_clusters(self, data, max_clusters=4):
        hit_points, remaining_connections, junctions, pos_edges = self.build_skeleton(data)
        segments = []
        for i, j in remaining_connections:
            if (i in hit_points or i in junctions) and (j in hit_points or j in junctions):
                dist = scipy.spatial.distance.cdist(
                    self.node_list[i].reshape(1, -1),
                    self.node_list[j].reshape(1, -1),
                    self.distance
                )[0][0]
                segments.append((i, j, dist))
        
        segments.sort(key=lambda x: x[2], reverse=True)
        
        G = nx.Graph(remaining_connections)
        clusters = []
        remaining_connections = remaining_connections.copy()
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

    def analyze_internal_characteristics(self, clusters):
        """Analyze mean and variance of node weights within each cluster to represent internal characteristics."""
        cluster_stats = {}
        for cluster_id, cluster_nodes in enumerate(clusters[-1]):
            if cluster_nodes:  # Ensure cluster is not empty
                weights = np.array([self.node_list[node] for node in cluster_nodes if node < self.node_count])
                if weights.size > 0:
                    mean_weights = np.mean(weights, axis=0)
                    variance_weights = np.var(weights, axis=0)
                    cluster_stats[cluster_id] = {
                        'mean_weights': mean_weights,
                        'variance_weights': variance_weights,
                        'node_count': len(cluster_nodes)
                    }
        return cluster_stats

    def model_relationships(self, clusters, segments):
        """Quantify inter-cluster and intra-cluster relationships using average distances."""
        cluster_relationships = {}
        for cluster_id, cluster_nodes in enumerate(clusters[-1]):
            intra_distances = []
            for i in cluster_nodes:
                for j in cluster_nodes:
                    if i < j and i < self.node_count and j < self.node_count:
                        dist = scipy.spatial.distance.cdist(
                            self.node_list[i].reshape(1, -1),
                            self.node_list[j].reshape(1, -1),
                            self.distance
                        )[0][0]
                        intra_distances.append(dist)
            intra_avg = np.mean(intra_distances) if intra_distances else 0.0

            inter_distances = []
            for other_id, other_nodes in enumerate(clusters[-1]):
                if other_id != cluster_id:
                    for i in cluster_nodes:
                        for j in other_nodes:
                            if i < self.node_count and j < self.node_count:
                                dist = scipy.spatial.distance.cdist(
                                    self.node_list[i].reshape(1, -1),
                                    self.node_list[j].reshape(1, -1),
                                    self.distance
                                )[0][0]
                                inter_distances.append(dist)
            inter_avg = np.mean(inter_distances) if inter_distances else float('inf')

            cluster_relationships[cluster_id] = {
                'intra_cluster_distance': intra_avg,
                'inter_cluster_distance': inter_avg,
                'separation_ratio': inter_avg / intra_avg if intra_avg > 0 else float('inf')
            }
        return cluster_relationships

    def model_idionomic_features(self, data, clusters, index_col='Name'):
        """Model idionomic features by associating individual data points with cluster properties."""
        idionomic_features = {}
        if self.output is not None and index_col in self.output.columns:
            for idx, row in self.output.iterrows():
                data_index = row[index_col]
                node_idx = row['output']
                cluster_id = next((i for i, cluster in enumerate(clusters[-1]) if node_idx in cluster), -1)
                if cluster_id != -1:
                    if data_index not in idionomic_features:
                        idionomic_features[data_index] = {}
                    # Extract original data features for this point (excluding Name and label columns)
                    data_point = data[data[index_col] == data_index].iloc[:, 1:17].values[0] if not data.empty else np.zeros(self.dimentions)
                    cluster_nodes = [n for n in clusters[-1][cluster_id] if n < self.node_count]
                    if cluster_nodes:
                        cluster_centroid = np.mean([self.node_list[n] for n in cluster_nodes], axis=0)
                    else:
                        cluster_centroid = np.zeros(self.dimentions)
                    
                    idionomic_features[data_index][cluster_id] = {
                        'data_features': data_point,
                        'node_weight': self.node_list[node_idx] if node_idx < self.node_count else np.zeros(self.dimentions),
                        'distance_to_centroid': scipy.spatial.distance.cdist(
                            data_point.reshape(1, -1),
                            cluster_centroid.reshape(1, -1),
                            self.distance
                        )[0][0]
                    }
        return idionomic_features

def plot(output, index_col, gsom_map=None, file_name="gsom", file_type=".pdf", figure_label="GSOM Map",
         max_text=3, max_length=30, cmap_colors="Paired", show_index=True, n_nodes=180):
    max_count = output["hit_count"].max()
    listed_color_map = _get_color_map(max_count, alpha=0.9, cmap_colors=cmap_colors)
    fig, ax = plt.subplots(figsize=(10, 10))
    if gsom_map:
        ax.plot(gsom_map.node_coordinate[:n_nodes, 0], gsom_map.node_coordinate[:n_nodes, 1], 'o',
                color=listed_color_map.colors[0], markersize=2)
    for index, i in output.iterrows():
        x = i['x']
        y = i['y']
        ax.plot(x, y, 'o', color=listed_color_map.colors[i['hit_count']], markersize=2)
        if show_index:
            if i['hit_count'] > 0:
                label = ", ".join(map(str, i[index_col][0:max_text]))
            else:
                label = ""
            txt = ax.text(x, y, label, ha='left', va='center', wrap=True, fontsize=4)
            txt._get_wrap_line_width = lambda: max_length
    ax.set_title(figure_label)
    plt.savefig(file_name + file_type)
    plt.close()

def _get_color_map(max_count, alpha=0.5, cmap_colors="Reds"):
    np.random.seed(1)
    cmap = plt.get_cmap(cmap_colors, max_count + 1)
    color_list = []
    for ind in range(cmap.N):
        c = []
        for x in cmap(ind)[:3]:
            c.append(x * alpha)
        color_list.append(tuple(c))
    return colors.ListedColormap(color_list, name='gsom_color_list')

def plot_pos(output, index_col, gsom_map=None, file_name="gsom", file_type=".pdf", 
         figure_label="GSOM Map with Paths", max_text=3, max_length=30, 
         cmap_colors="Paired", show_index=True, n_nodes=180):
    """
    Plot GSOM nodes with their clustered data points and paths between nodes.
    """
    max_count = output["hit_count"].max()
    listed_color_map = _get_color_map_pos(max_count, alpha=0.9, cmap_colors=cmap_colors)
    fig, ax = plt.subplots(figsize=(10, 8))
    if gsom_map:
        ax.plot(gsom_map.node_coordinate[:gsom_map.node_count, 0],
                gsom_map.node_coordinate[:gsom_map.node_count, 1],
                'o', color='gray', markersize=2, label='All Nodes')
    if gsom_map:
        paths = gsom_map.get_paths()
        for path in paths:
            if len(path) > 1:
                x_coords = [node.x for node in path]
                y_coords = [node.y for node in path]
                ax.plot(x_coords, y_coords, 'k-', linewidth=0.5, alpha=0.3,
                        label='Node Connections' if path == paths[0] else "")
    for index, i in output.iterrows():
        x = i['x']
        y = i['y']
        ax.plot(x, y, 'o', color=listed_color_map.colors[i['hit_count']],
                markersize=6, label=f'Hit Count {i["hit_count"]}' if i['hit_count'] > 0 else "")
        if show_index and i['hit_count'] > 0:
            label = ", ".join(map(str, i[index_col][0:max_text]))
            txt = ax.text(x, y, label, ha='left', va='center', wrap=True, fontsize=8)
            txt._get_wrap_line_width = lambda: max_length
    ax.set_title(figure_label)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
    plt.savefig(file_name + file_type, bbox_inches='tight')
    plt.show()

def plot_labels(output, index_col, gsom_map=None, file_name="gsom", file_type=".pdf", 
         figure_label="GSOM Map with Paths", max_text=1, max_length=30, 
         cmap_colors="Paired", show_index=True, n_nodes=180):
    """
    Plot GSOM nodes with their clustered data points and paths between nodes.
    """
    max_count = output["hit_count"].max()
    listed_color_map = _get_color_map_pos(max_count, alpha=0.9, cmap_colors=cmap_colors)
    fig, ax = plt.subplots(figsize=(10, 8))
    if gsom_map:
        ax.plot(gsom_map.node_coordinate[:gsom_map.node_count, 0],
                gsom_map.node_coordinate[:gsom_map.node_count, 1],
                'o', color='gray', markersize=2, label='All Nodes')
    if gsom_map:
        paths = gsom_map.get_paths()
        for path in paths:
            if len(path) > 1:
                x_coords = [node.x for node in path]
                y_coords = [node.y for node in path]
                ax.plot(x_coords, y_coords, 'k-', linewidth=0.5, alpha=0.3,
                        label='Node Connections' if path == paths[0] else "")
    for index, i in output.iterrows():
        x = i['x']
        y = i['y']
        ax.plot(x, y, 'o', color=listed_color_map.colors[i['hit_count']],
                markersize=6, label=f'Hit Count {i["hit_count"]}' if i['hit_count'] > 0 else "")
        if show_index and i['hit_count'] > 0:
            label = ", ".join(map(str, i[index_col][0:max_text]))
            txt = ax.text(x, y, label, ha='left', va='center', wrap=True, fontsize=8)
            txt._get_wrap_line_width = lambda: max_length
    ax.set_title(figure_label)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
    plt.savefig("gsom_map_with_labels.png")
    plt.savefig(file_name + file_type, bbox_inches='tight')
    plt.show()

def _get_color_map_pos(max_count, alpha=0.9, cmap_colors="Paired"):
    np.random.seed(1)
    cmap = plt.get_cmap(cmap_colors, max_count + 1)
    color_list = [(c[0] * alpha, c[1] * alpha, c[2] * alpha) for c in cmap(np.arange(cmap.N))]
    return colors.ListedColormap(color_list, name='gsom_color_list')

from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np

def calculate_cluster_metrics(data, gsom, clusters, node_labels, index_col='Name'):
    # Create a mapping from index_col values to data row indices
    df = pd.DataFrame(data, index=gsom.output[index_col].values)  # Assuming gsom.output has the same index_col values
    name_to_index = {name: idx for idx, name in enumerate(df.index)}

    # Map data points to cluster labels
    node_to_cluster = {}
    for cluster_id, cluster in enumerate(clusters[-1]):  # Use the last clustering result
        for node_idx in cluster:
            node_to_cluster[node_idx] = cluster_id

    # Get cluster labels for each data point based on node assignments
    cluster_labels = np.array([-1] * data.shape[0])  # Initialize with -1 for all data points
    for idx, row in gsom.output.iterrows():
        data_index = row[index_col]
        node_idx = row['output']
        if node_idx in node_to_cluster:
            cluster_labels[name_to_index.get(data_index, -1)] = node_to_cluster[node_idx]

    # Filter out any -1 labels (unclustered points, if any)
    valid_indices = cluster_labels != -1
    if np.sum(valid_indices) < 2 or len(np.unique(cluster_labels[valid_indices])) < 2:
        print("Not enough valid clusters for metric calculation.")
        return None, None

    valid_data = data[valid_indices]
    valid_labels = cluster_labels[valid_indices]

    # Calculate Silhouette Score
    try:
        sil_score = silhouette_score(valid_data, valid_labels, metric='euclidean')
    except ValueError as e:
        print(f"Error calculating Silhouette Score: {e}")
        sil_score = None

    # Calculate Davies-Bouldin Index
    try:
        db_score = davies_bouldin_score(valid_data, valid_labels)
    except ValueError as e:
        print(f"Error calculating Davies-Bouldin Score: {e}")
        db_score = None

    return sil_score, db_score

def plot_idionomic_features_simple(gsom, df, clusters, idionomic_features):
    """
    Simple visualization of idionomic features with 1-based cluster numbering (1-7).
    
    Parameters:
    - gsom: Your trained GSOM object
    - df: Original dataframe
    - clusters: Cluster assignments
    - idionomic_features: Dictionary of idionomic features
    """
    
    # 1. Distance to Centroid Box Plot
    distances = []
    cluster_ids = []
    individuals = []
    
    for individual, clusters_data in idionomic_features.items():
        for cluster_id, features in clusters_data.items():
            distances.append(features['distance_to_centroid'])
            cluster_ids.append(cluster_id + 1)  # Convert to 1-based (1-7)
            individuals.append(individual)
    
    plt.figure(figsize=(10, 6))
    df_dist = pd.DataFrame({
        'Cluster': cluster_ids,
        'Distance': distances,
        'Individual': individuals
    })
    
    sns.boxplot(data=df_dist, x='Cluster', y='Distance')
    plt.title('Distance to Cluster Centroid Distribution (Clusters 1-7)')
    plt.xlabel('Cluster ID')
    plt.ylabel('Distance to Centroid')
    plt.savefig('distance_to_centroid_by_cluster.png', dpi=300, bbox_inches='tight')
    plt.savefig('distance_to_centroid_by_cluster.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Feature Profiles for Sample Individuals
    sample_individuals = list(idionomic_features.keys())[:6]  # Show 6 individuals
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    feature_names = ['hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic', 'predator', 
                     'toothed', 'backbone', 'breathes', 'venomous', 'fins', 'legs', 'tail', 'domestic', 'catsize']
    
    for i, individual in enumerate(sample_individuals):
        clusters_data = idionomic_features[individual]
        
        for cluster_id, features in clusters_data.items():
            data_features = features['data_features']
            node_weights = features['node_weight']
            
            x_indices = range(len(data_features))
            axes[i].plot(x_indices, data_features, 'o-', label='Original Data', linewidth=2, markersize=6)
            axes[i].plot(x_indices, node_weights, 's--', label=f'Node Weights (Cluster {cluster_id + 1})', linewidth=2, markersize=4)
            axes[i].set_title(f'{individual} (Cluster {cluster_id + 1})')
            axes[i].set_xlabel('Feature Index')
            axes[i].set_ylabel('Value')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xticks(x_indices[::2])  # Show every other tick to avoid crowding
    
    plt.tight_layout()
    plt.savefig('idionomic_profiles.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. PCA Visualization
    feature_data = []
    labels = []
    cluster_assignments = []
    
    for individual, clusters_data in idionomic_features.items():
        for cluster_id, features in clusters_data.items():
            feature_data.append(features['data_features'])
            labels.append(individual)
            cluster_assignments.append(cluster_id + 1)  # Convert to 1-based
    
    if feature_data:
        feature_matrix = np.array(feature_data)
        
        # Standardize and apply PCA
        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)
        
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(feature_matrix_scaled)
        
        plt.figure(figsize=(10, 8))
        colors = plt.cm.Set3(np.linspace(0, 1, 7))  # 7 clusters
        
        for i, cluster_id in enumerate(range(1, 8)):  # Clusters 1-7
            mask = np.array(cluster_assignments) == cluster_id
            if np.any(mask):
                plt.scatter(pca_result[mask, 0], pca_result[mask, 1], 
                           c=[colors[i]], label=f'Cluster {cluster_id}', alpha=0.7, s=60)
        
        plt.title(f'PCA of Idionomic Features (Clusters 1-7)\n(PC1: {pca.explained_variance_ratio_[0]:.2%}, PC2: {pca.explained_variance_ratio_[1]:.2%})')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('idionomic_pca.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 4. Feature Deviation Analysis
    deviations = []
    for individual, clusters_data in idionomic_features.items():
        for cluster_id, features in clusters_data.items():
            deviation = np.linalg.norm(features['data_features'] - features['node_weight'])
            deviations.append(deviation)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df_dist, x='Cluster', y='Distance')
    plt.title('Distance to Centroid by Cluster (1-7)')
    
    plt.subplot(1, 2, 2)
    plt.hist(deviations, bins=15, alpha=0.7, edgecolor='black')
    plt.title('Distribution of Feature Deviations')
    plt.xlabel('L2 Norm Deviation')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('idionomic_deviations.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. Summary Statistics
    print("\n=== IDIONOMIC FEATURES SUMMARY ===")
    print(f"Total individuals analyzed: {len(set(individuals))}")
    print(f"Clusters represented: {sorted(set(cluster_ids))}")
    print(f"Average distance to centroid: {np.mean(distances):.4f}")
    print(f"Standard deviation of distances: {np.std(distances):.4f}")
    print(f"Average feature deviation: {np.mean(deviations):.4f}")
    
    # Show top and bottom deviating individuals
    deviation_df = pd.DataFrame({
        'Individual': individuals,
        'Cluster': cluster_ids,
        'Distance': distances,
        'Deviation': deviations
    })
    
    print("\nTop 5 individuals with highest distance to centroid:")
    print(deviation_df.nlargest(5, 'Distance')[['Individual', 'Cluster', 'Distance']])
    
    print("\nTop 5 individuals with lowest distance to centroid:")
    print(deviation_df.nsmallest(5, 'Distance')[['Individual', 'Cluster', 'Distance']])
    
    # Save summary to CSV
    deviation_df.to_csv('idionomic_summary.csv', index=False)
    print("\nSummary saved to 'idionomic_summary.csv'")

if __name__ == '__main__':
    np.random.seed(1)
    # Load Zoo dataset
    data_filename = "zoo.txt"
    df = pd.read_csv(data_filename)
    
    print("Dataset shape:", df.shape)
    data_training = df.iloc[:, 1:17]
    print(type(data_training))
    print("Training data head:", data_training.head())
    print("Training data shape:", data_training.shape)
    # Train GSOM
    gsom = GSOM(0.83, 16, max_radius=4, initial_node_size=1000)  # Use 0.25 to match paper
    gsom.fit(data_training.to_numpy(), 100, 50)
    output = gsom.predict(df, "Name", "label")
    output.to_csv("output.csv", index=False)
    print("GSOM training completed.")
    print("Output shape:", output.shape)
    print("Node Count:", gsom.node_count)
    # Get paths of spread
    paths = gsom.get_paths()

    # Export paths of spread to a CSV file
    paths_data = []
    for path in paths:
        node_names = [node.name for node in path]
        node_coords = [(node.x, node.y) for node in path if hasattr(node, 'x') and hasattr(node, 'y')]
        paths_data.append({
            "node_names": ";".join(node_names),
            "node_coords": ";".join([f"({x},{y})" for x, y in node_coords])
        })
    paths_df = pd.DataFrame(paths_data)
    paths_df.to_csv("paths_of_spread.csv", index=False)
    
    # Build skeleton and separate clusters
    clusters, segments, remaining_connections, pos_edges = gsom.separate_clusters(data_training.to_numpy(), max_clusters=7)
    
    # Plot GSOM map
    plot(output, "Name", gsom_map=gsom, file_name="gsom_map", file_type=".pdf", figure_label="GSOM Map")
    plot_pos(output, "Name", gsom_map=gsom, file_name="gsom_with_paths_sk",
             file_type=".pdf", figure_label="GSOM Map with Node Paths", n_nodes=gsom.node_count)
    
    plot_labels(output, "label", gsom_map=gsom, file_name="gsom_with_labels",
             file_type=".pdf", figure_label="GSOM Map with Node Paths", n_nodes=gsom.node_count)
    # Plot skeleton with clusters
    fig, ax = plt.subplots(figsize=(10, 10))
    for i in range(gsom.node_count):
        x, y = gsom.node_coordinate[i]
        if i in gsom.node_labels['output'].values:
            color = 'blue'
            size = 30
            alpha = 0.3
            marker = 'D'  # Diamond marker for output nodes
        else:
            color = 'gray'
            size = 10
            alpha = 0.1
            marker = 'o'  # Circle marker for other nodes
        ax.scatter(x, y, c=color, s=size, marker=marker, alpha=alpha)
    
    from collections import Counter
    overlaps = Counter((i, j) for i, j in remaining_connections)

    # Write overlaps to a CSV file
    overlap_df = pd.DataFrame([(i, j, count) for (i, j), count in overlaps.items()],
                            columns=["node1", "node2", "overlap_count"])
    overlap_df.to_csv("edge_overlaps.csv", index=False)

    counts = np.array(list(overlaps.values()))
    q1 = np.percentile(counts, 25)  # 25th percentile
    median = np.percentile(counts, 50)  # 50th percentile (median)
    q3 = np.percentile(counts, 75)  # 75th percentile
    print(f"Q1 (25th percentile): {q1}")
    print(f"Median (50th percentile): {median}")
    print(f"Q3 (75th percentile): {q3}")
    mean = np.mean(counts)
    std = np.std(counts)
    print(f"Mean: {mean}")
    print(f"Standard Deviation: {std}")

    for i, j in remaining_connections:
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
        
        ax.plot([x1, x2], [y1, y2], color=color, linestyle=line_style, alpha=alpha)
    
    # Analyze internal characteristics
    internal_chars = gsom.analyze_internal_characteristics(clusters)
    for cluster_id, stats in internal_chars.items():
        print(f"Cluster {cluster_id}: Mean Weights = {stats['mean_weights']}, Variance = {stats['variance_weights']}")

    # Model relationships
    relationships = gsom.model_relationships(clusters, segments)
    for cluster_id, rel in relationships.items():
        print(f"Cluster {cluster_id}: Intra-Distance = {rel['intra_cluster_distance']:.2f}, "
              f"Inter-Distance = {rel['inter_cluster_distance']:.2f}, Separation Ratio = {rel['separation_ratio']:.2f}")

    # Model idionomic features
    idionomic = gsom.model_idionomic_features(df, clusters, index_col='Name')
    
    # Save idionomic features to CSV with proper cluster numbering (1-7)
    idionomic_data = []
    for animal, features in idionomic.items():
        for cluster_id, feat in features.items():
            print(f"Animal {animal} in Cluster {cluster_id + 1}: Data Features = {feat['data_features']}, "
                  f"Node Weight = {feat['node_weight']}, Distance to Centroid = {feat['distance_to_centroid']:.2f}")
            
            # Add to CSV data with 1-based cluster numbering
            row = {
                'Animal': animal,
                'Cluster': cluster_id + 1,  # Convert to 1-based
                'Distance_to_Centroid': feat['distance_to_centroid']
            }
            # Add data features as separate columns
            for i, val in enumerate(feat['data_features']):
                row[f'Feature_{i+1}'] = val
            # Add node weights as separate columns
            for i, val in enumerate(feat['node_weight']):
                row[f'Node_Weight_{i+1}'] = val
            idionomic_data.append(row)
    
    # Save to CSV
    idionomic_df = pd.DataFrame(idionomic_data)
    idionomic_df.to_csv("idionomic_features.csv", index=False)
    print("Idionomic features saved to idionomic_features.csv")
    
    # Save a simpler version with just names and features
    feature_names = ['hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic', 'predator', 
                     'toothed', 'backbone', 'breathes', 'venomous', 'fins', 'legs', 'tail', 'domestic', 'catsize']
    simple_data = []
    for animal, features in idionomic.items():
        for cluster_id, feat in features.items():
            row = {'Animal': animal, 'Cluster': cluster_id + 1, 'Distance_to_Centroid': feat['distance_to_centroid']}
            for i, feature_name in enumerate(feature_names):
                row[feature_name] = feat['data_features'][i]
            simple_data.append(row)
    
    simple_df = pd.DataFrame(simple_data)
    simple_df.to_csv("idionomic_features_w_names.csv", index=False)
    print("Idionomic features with feature names saved to idionomic_features_w_names.csv")

    colors = ['green', 'red', 'black', 'cyan']
    print("Clusters found:", len(clusters[-1]))
    # Save clusters to a CSV file
    clusters_data = []
    for idx, cluster in enumerate(clusters[-1]):
        print(f"Cluster {idx + 1}: {len(cluster)} nodes : Color {colors[idx % len(colors)]}")

        clusters_data.append({
            "cluster_id": idx + 1,
            "node_indices": ";".join(map(str, cluster)),
            "color": colors[idx % len(colors)],
            "size": len(cluster)
        })
        
        for node_idx in cluster:
            x, y = gsom.node_coordinate[node_idx]
            ax.scatter(x, y, c=colors[idx % len(colors)], s=20, marker='o', alpha=0.5)
    
    clusters_df = pd.DataFrame(clusters_data)
    clusters_df.to_csv("clusters_zoo.csv", index=False)

    ax.set_title("GSOM Skeleton with Clusters")
    plt.savefig("gsom_skeleton.pdf")
    plt.show()
    
    segment_df = pd.DataFrame(segments, columns=["node1", "node2", "distance"])
    segment_df.to_csv("segment_distances.csv", index=False)
    
    graph = tree_to_dot(gsom.path_tree)
    graph.write_png("path_tree.png")

    sil_score, db_score = calculate_cluster_metrics(data_training.to_numpy(), gsom, clusters, output, index_col='Name')
    print(f"Silhouette Score: {sil_score}")
    print(f"Davies-Bouldin Index: {db_score}")


    # if idionomic:
    print("\n=== VISUALIZING IDIONOMIC FEATURES ===")
    plot_idionomic_features_simple(gsom, df, clusters, idionomic)

    print("Complete")