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
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score, davies_bouldin_score, homogeneity_score, completeness_score, v_measure_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from collections import Counter

class GSOM:
    def __init__(self, spred_factor, dimensions, distance='euclidean', initialize='random', learning_rate=0.3,
                 smooth_learning_factor=0.8, max_radius=3, FD=0.1, r=3.8, alpha=0.9, initial_node_size=1000):
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

    def separate_clusters(self, data, max_clusters=3, distance_threshold=None):
        hit_points, skeleton_connections, junctions, pos_edges = self.build_skeleton(data)
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
        
        # Use distance threshold if provided
        if distance_threshold is None:
            # Estimate threshold as 75th percentile of segment distances
            distances = [s[2] for s in segments]
            distance_threshold = np.percentile(distances, 75) if distances else float('inf')
        
        G = nx.Graph(skeleton_connections)
        clusters = []
        remaining_connections = skeleton_connections.copy()
        for i, j, dist in segments:
            if dist < distance_threshold:
                continue  # Skip edges below threshold to avoid over-fragmentation
            if (i, j) in remaining_connections:
                remaining_connections.remove((i, j))
            if G.has_edge(i, j):
                G.remove_edge(i, j)
                clusters.append(list(nx.connected_components(G)))
                print(f"Removed segment {i}-{j}, Distance: {dist}")
                if len(clusters[-1]) >= max_clusters:
                    break
        return clusters, segments, remaining_connections, pos_edges

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
    cmap = cm.get_cmap(cmap_colors, max_count + 1)
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
    plt.close()

def _get_color_map_pos(max_count, alpha=0.9, cmap_colors="Paired"):
    np.random.seed(1)
    cmap = cm.get_cmap(cmap_colors, max_count + 1)
    color_list = [(c[0] * alpha, c[1] * alpha, c[2] * alpha) for c in cmap(np.arange(cmap.N))]
    return colors.ListedColormap(color_list, name='gsom_color_list')

def compare_clustering_methods(data, true_labels, gsom, clusters, max_clusters=3):
    """
    Compare GSOM with DSM against comprehensive flat clustering methods.
    """
    # Standardize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Get GSOM DSM cluster labels
    dsm_labels = np.full(len(data), -1, dtype=int)  # Initialize with -1 for unassigned points
    cluster_sizes = []
    cluster_purities = []
    for cluster_idx, cluster in enumerate(clusters[-1]):  # Use the final set of clusters from DSM
        cluster_nodes = []
        cluster_true_labels = []
        for node_idx in cluster:
            # Find data points mapped to this node
            data_indices = gsom.output[gsom.output["output"] == node_idx].index
            dsm_labels[data_indices] = cluster_idx
            cluster_nodes.extend(data_indices)
            cluster_true_labels.extend(true_labels[data_indices])
        cluster_sizes.append(len(cluster_nodes))
        # Calculate purity: fraction of the most common true label in the cluster
        if cluster_true_labels:
            most_common_label = Counter(cluster_true_labels).most_common(1)[0]
            purity = most_common_label[1] / len(cluster_true_labels)
        else:
            purity = 0
        cluster_purities.append(purity)
    
    # Get GSOM baseline labels (without DSM)
    gsom_labels = gsom.output["output"].to_numpy()
    
    # Log cluster statistics
    print("\nGSOM+DSM Cluster Statistics:")
    print("Cluster Sizes:", cluster_sizes)
    print("Cluster Purities:", cluster_purities)
    print("Unassigned Points:", np.sum(dsm_labels == -1))
    
    # Initialize clustering methods
    clustering_methods = {}
    
    # K-Means
    kmeans = KMeans(n_clusters=max_clusters, random_state=1, n_init=10)
    clustering_methods['K-Means'] = kmeans.fit_predict(data_scaled)
    
    # Hierarchical Clustering (Agglomerative)
    hierarchical = AgglomerativeClustering(n_clusters=max_clusters, linkage='ward')
    clustering_methods['Hierarchical'] = hierarchical.fit_predict(data_scaled)
    
    # Gaussian Mixture Model
    gmm = GaussianMixture(n_components=max_clusters, random_state=1)
    clustering_methods['GMM'] = gmm.fit_predict(data_scaled)
    
    # Spectral Clustering
    spectral = SpectralClustering(n_clusters=max_clusters, random_state=1, n_init=10)
    clustering_methods['Spectral'] = spectral.fit_predict(data_scaled)
    
    # DBSCAN with multiple eps values
    eps_values = [0.3, 0.5, 0.7, 1.0]
    for eps in eps_values:
        dbscan = DBSCAN(eps=eps, min_samples=5)
        labels = dbscan.fit_predict(data_scaled)
        clustering_methods[f'DBSCAN(eps={eps})'] = labels
    
    # Add GSOM methods
    clustering_methods['GSOM'] = gsom_labels
    clustering_methods['GSOM+DSM'] = dsm_labels
    
    # Comprehensive evaluation metrics
    metrics = {
        'Method': list(clustering_methods.keys()),
        'ARI': [],
        'Homogeneity': [],
        'Completeness': [],
        'V-measure': [],
        'Silhouette': [],
        'Davies-Bouldin': [],
        'Calinski-Harabasz': [],
        'Num_Clusters': [],
        'Noise_Points': []
    }
    
    for method_name in metrics['Method']:
        labels = clustering_methods[method_name]
        
        # Count clusters and noise points
        unique_labels = np.unique(labels)
        num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        noise_points = np.sum(labels == -1)
        
        metrics['Num_Clusters'].append(num_clusters)
        metrics['Noise_Points'].append(noise_points)
        
        # Adjusted Rand Index
        ari = adjusted_rand_score(true_labels, labels)
        metrics['ARI'].append(ari)
        
        # Homogeneity, Completeness, V-measure
        homogeneity = homogeneity_score(true_labels, labels)
        completeness = completeness_score(true_labels, labels)
        v_measure = v_measure_score(true_labels, labels)
        metrics['Homogeneity'].append(homogeneity)
        metrics['Completeness'].append(completeness)
        metrics['V-measure'].append(v_measure)
        
        # Silhouette Score (skip if only one cluster or all noise points)
        if num_clusters > 1 and not (labels == -1).all():
            # For methods with noise points, exclude them from silhouette calculation
            if -1 in labels:
                non_noise_mask = labels != -1
                if np.sum(non_noise_mask) > 1 and len(np.unique(labels[non_noise_mask])) > 1:
                    silhouette = silhouette_score(data_scaled[non_noise_mask], labels[non_noise_mask])
                else:
                    silhouette = np.nan
            else:
                silhouette = silhouette_score(data_scaled, labels)
        else:
            silhouette = np.nan
        metrics['Silhouette'].append(silhouette)
        
        # Davies-Bouldin Index (skip if only one cluster or all noise points)
        if num_clusters > 1 and not (labels == -1).all():
            if -1 in labels:
                non_noise_mask = labels != -1
                if np.sum(non_noise_mask) > 1 and len(np.unique(labels[non_noise_mask])) > 1:
                    db = davies_bouldin_score(data_scaled[non_noise_mask], labels[non_noise_mask])
                else:
                    db = np.nan
            else:
                db = davies_bouldin_score(data_scaled, labels)
        else:
            db = np.nan
        metrics['Davies-Bouldin'].append(db)
        
        # Calinski-Harabasz Index (skip if only one cluster or all noise points)
        if num_clusters > 1 and not (labels == -1).all():
            if -1 in labels:
                non_noise_mask = labels != -1
                if np.sum(non_noise_mask) > 1 and len(np.unique(labels[non_noise_mask])) > 1:
                    ch = calinski_harabasz_score(data_scaled[non_noise_mask], labels[non_noise_mask])
                else:
                    ch = np.nan
            else:
                ch = calinski_harabasz_score(data_scaled, labels)
        else:
            ch = np.nan
        metrics['Calinski-Harabasz'].append(ch)
        
        # Log unique labels
        print(f"{method_name} - Clusters: {num_clusters}, Noise: {noise_points}, Unique Labels: {np.unique(labels)}")
    
    # Save comprehensive metrics to DataFrame and CSV
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv("flat_clustering_comparison_iris.csv", index=False)
    
    print("\nComprehensive Flat Clustering Comparison Metrics:")
    print(metrics_df.round(4))
    
    # Create summary analysis
    print("\n" + "="*80)
    print("FLAT CLUSTERING ANALYSIS SUMMARY")
    print("="*80)
    
    # Best method for each metric (excluding NaN values)
    best_methods = {}
    
    # For ARI, Homogeneity, Completeness, V-measure, Silhouette, Calinski-Harabasz: higher is better
    for metric in ['ARI', 'Homogeneity', 'Completeness', 'V-measure', 'Silhouette', 'Calinski-Harabasz']:
        valid_scores = [(i, score) for i, score in enumerate(metrics[metric]) if not np.isnan(score)]
        if valid_scores:
            best_idx, best_score = max(valid_scores, key=lambda x: x[1])
            best_methods[metric] = (metrics['Method'][best_idx], best_score)
    
    # For Davies-Bouldin: lower is better
    valid_scores = [(i, score) for i, score in enumerate(metrics['Davies-Bouldin']) if not np.isnan(score)]
    if valid_scores:
        best_idx, best_score = min(valid_scores, key=lambda x: x[1])
        best_methods['Davies-Bouldin'] = (metrics['Method'][best_idx], best_score)
    
    print("\nBest performing methods by metric:")
    for metric, (method, score) in best_methods.items():
        print(f"{metric:20s}: {method:15s} (score: {score:.4f})")
    
    # GSOM Performance Analysis
    gsom_idx = metrics['Method'].index('GSOM')
    gsom_dsm_idx = metrics['Method'].index('GSOM+DSM')
    
    print(f"\nGSOM vs GSOM+DSM Comparison:")
    print(f"{'Metric':<20s} {'GSOM':>10s} {'GSOM+DSM':>12s} {'Improvement':>12s}")
    print("-" * 55)
    
    for metric in ['ARI', 'Homogeneity', 'Completeness', 'V-measure', 'Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz']:
        gsom_score = metrics[metric][gsom_idx]
        dsm_score = metrics[metric][gsom_dsm_idx]
        
        if not np.isnan(gsom_score) and not np.isnan(dsm_score):
            if metric == 'Davies-Bouldin':  # Lower is better
                improvement = gsom_score - dsm_score
                improvement_text = f"{improvement:+.4f}"
            else:  # Higher is better
                improvement = dsm_score - gsom_score
                improvement_text = f"{improvement:+.4f}"
            
            print(f"{metric:<20s} {gsom_score:>10.4f} {dsm_score:>12.4f} {improvement_text:>12s}")
        else:
            print(f"{metric:<20s} {'N/A':>10s} {'N/A':>12s} {'N/A':>12s}")
    
    return metrics_df

if __name__ == '__main__':
    np.random.seed(1)
    # Load Iris dataset
    data_filename = "Iris.csv"
    df = pd.read_csv(data_filename)
    
    print("Dataset shape:", df.shape)
    data_training = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    true_labels = df['Species'].astype('category').cat.codes.to_numpy()
    print("Training data head:", data_training.head())
    print("Training data shape:", data_training.shape)
    print("Unique species:", df['Species'].unique())
    
    # Scale data for GSOM
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_training)
    
    # Train GSOM with optimized parameters
    gsom = GSOM(spred_factor=0.1, dimensions=4, max_radius=3, initial_node_size=1000)
    gsom.fit(data_scaled, 200, 50)
    output = gsom.predict(df, "Id", "Species")
    output.to_csv("output_iris.csv", index=False)
    print("GSOM training completed.")
    print("Output shape:", output.shape)
    print("Node Count:", gsom.node_count)
    
    # Get paths of spread
    paths = gsom.get_paths()
    paths_data = []
    for path in paths:
        node_names = [node.name for node in path]
        node_coords = [(node.x, node.y) for node in path if hasattr(node, 'x') and hasattr(node, 'y')]
        paths_data.append({
            "node_names": ";".join(node_names),
            "node_coords": ";".join([f"({x},{y})" for x, y in node_coords])
        })
    paths_df = pd.DataFrame(paths_data)
    paths_df.to_csv("paths_of_spread_iris.csv", index=False)
    
    # Build skeleton and separate clusters (DSM) with distance threshold
    clusters, segments, skeleton_connections, pos_edges = gsom.separate_clusters(data_scaled, max_clusters=3, distance_threshold=None)
    
    # Compare clustering methods
    compare_clustering_methods(data_scaled, true_labels, gsom, clusters, max_clusters=3)
    
    # Plot GSOM map
    plot(output, "Id", gsom_map=gsom, file_name="gsom_map", file_type=".pdf", figure_label="GSOM Map")
    plot_pos(output, "Id", gsom_map=gsom, file_name="gsom_with_paths_sk_iris",
             file_type=".pdf", figure_label="GSOM Map with Node Paths", n_nodes=gsom.node_count)
    
    # Plot skeleton with clusters
    fig, ax = plt.subplots(figsize=(10, 10))
    for i in range(gsom.node_count):
        x, y = gsom.node_coordinate[i]
        if i in gsom.node_labels['output'].values:
            color = 'blue'
            size = 30
            alpha = 0.3
            marker = 'D'
        else:
            color = 'gray'
            size = 10
            alpha = 0.1
            marker = 'o'
        ax.scatter(x, y, c=color, s=size, marker=marker, alpha=alpha)
        ax.text(x, y, str(i), fontsize=6)
    
    overlaps = Counter((i, j) for i, j in skeleton_connections)
    overlap_df = pd.DataFrame([(i, j, count) for (i, j), count in overlaps.items()],
                             columns=["node1", "node2", "overlap_count"])
    overlap_df.to_csv("edge_overlaps_iris.csv", index=False)
    
    counts = np.array(list(overlaps.values()))
    q1 = np.percentile(counts, 25)
    median = np.percentile(counts, 50)
    q3 = np.percentile(counts, 75)
    print(f"Q1 (25th percentile): {q1}")
    print(f"Median (50th percentile): {median}")
    print(f"Q3 (75th percentile): {q3}")
    mean = np.mean(counts)
    std = np.std(counts)
    print(f"Mean: {mean}")
    print(f"Standard Deviation: {std}")
    
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
        ax.plot([x1, x2], [y1, y2], color=color, linestyle=line_style, alpha=alpha)
    
    colors = ['green', 'red', 'black', 'cyan']
    print("Clusters found:", len(clusters[-1]))
    for idx, cluster in enumerate(clusters[-1]):
        print(f"Cluster {idx + 1}: {len(cluster)} nodes : Color {colors[idx % len(colors)]}")
        for node_idx in cluster:
            x, y = gsom.node_coordinate[node_idx]
            ax.scatter(x, y, c=colors[idx % len(colors)], s=20, marker='o', alpha=0.5)
    
    ax.set_title("GSOM Skeleton with Clusters (DSM)")
    plt.savefig("gsom_skeleton_iris_dsm.pdf")
    plt.close()
    
    segment_df = pd.DataFrame(segments, columns=["node1", "node2", "distance"])
    segment_df.to_csv("segment_distances_iris.csv", index=False)
    
    graph = tree_to_dot(gsom.path_tree)
    graph.write_png("path_tree_iris.png")
    
    print("Complete")