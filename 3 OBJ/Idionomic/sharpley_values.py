import numpy as np
import pandas as pd
from scipy.spatial import distance
import scipy
from tqdm import tqdm
import math
from bigtree import Node, find, tree_to_dot
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import networkx as nx
from sklearn.metrics import silhouette_score, davies_bouldin_score
from collections import Counter
import seaborn as sns
import os
import shap  # For Shapley value computation
from functools import partial  # For binding node_idx to value_function

class GSOM:
    def __init__(self, spread_factor, dimensions, distance='euclidean', initialize='random', learning_rate=0.3,
                 smooth_learning_factor=0.8, max_radius=6, FD=0.1, r=3.8, alpha=0.9, initial_node_size=1000):
        self.initial_node_size = initial_node_size
        self.node_count = 0
        self.map = {}
        self.node_list = np.zeros((initial_node_size, dimensions))
        self.node_coordinate = np.zeros((initial_node_size, 2))
        self.node_errors = np.zeros(initial_node_size, dtype=np.longdouble)
        self.spread_factor = spread_factor
        self.growth_threshold = -dimensions * math.log(spread_factor)
        self.FD = FD
        self.R = r
        self.ALPHA = alpha
        self.dimensions = dimensions
        self.distance = distance
        self.initialize = initialize
        self.learning_rate = learning_rate
        self.smooth_learning_factor = smooth_learning_factor
        self.max_radius = max_radius
        self.node_labels = None
        self.output = None
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
            node_weights = np.random.rand(self.dimensions)
        else:
            raise NotImplementedError("Initialization method not supported")
        self.insert_new_node(x, y, node_weights, parent_node=self.path_tree)

    def _get_learning_rate(self, prev_learning_rate):
        return self.ALPHA * (1 - (self.R / self.node_count)) * prev_learning_rate

    def _get_neighbourhood_radius(self, total_iteration, iteration):
        time_constant = total_iteration / math.log(self.max_radius)
        return self.max_radius * math.exp(-iteration / time_constant)

    def _new_weights_for_new_node_in_middle(self, winnerx, winnery, next_nodex, next_nodey):
        return (self.node_list[self.map[(winnerx, winnery)]] + 
                self.node_list[self.map[(next_nodex, next_nodey)]]) * 0.5

    def _new_weights_for_new_node_on_one_side(self, winnerx, winnery, next_nodex, next_nodey):
        return (2 * self.node_list[self.map[(winnerx, winnery)]] - 
                self.node_list[self.map[(next_nodex, next_nodey)]])

    def _new_weights_for_new_node_one_older_neighbour(self, winnerx, winnery):
        weights = self.node_list[self.map[(winnerx, winnery)]]
        return np.full(self.dimensions, (max(weights) + min(weights)) / 2)

    def grow_node(self, wx, wy, x, y, side):
        if (x, y) not in self.map:
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

    def spread_weights(self, x, y):
        for dx, dy in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if (nx, ny) in self.map:
                self.node_errors[self.map[(nx, ny)]] *= (1 + self.FD)
        self.node_errors[self.map[(x, y)]] = self.growth_threshold / 2

    def adjust_weights(self, x, y, rmu_index):
        directions = [(-1, 0, 0), (1, 0, 1), (0, 1, 2), (0, -1, 3)]
        all_neighbors_exist = all((x + dx, y + dy) in self.map for dx, dy, _ in directions)
        if all_neighbors_exist:
            self.spread_weights(x, y)
        else:
            for dx, dy, side in directions:
                self.grow_node(x, y, x + dx, y + dy, side)
        self.node_errors[rmu_index] = self.growth_threshold / 2

    def winner_identification_and_neighbourhood_update(self, data_index, data, radius, learning_rate):
        out = scipy.spatial.distance.cdist(self.node_list[:self.node_count], 
                                        data[data_index, :].reshape(1, self.dimensions), self.distance)
        rmu_index = out.argmin()
        error_val = out.min()
        rmu_x = int(self.node_coordinate[rmu_index][0])
        rmu_y = int(self.node_coordinate[rmu_index][1])

        error = data[data_index] - self.node_list[rmu_index]
        self.node_list[self.map[(rmu_x, rmu_y)]] += learning_rate * error

        mask_size = round(radius)
        for i in range(rmu_x - mask_size, rmu_x + mask_size + 1):
            for j in range(rmu_y - mask_size, rmu_y + mask_size + 1):
                if (i, j) in self.map and (i != rmu_x or j != rmu_y):
                    error = self.node_list[rmu_index] - self.node_list[self.map[(i, j)]]
                    dist = (rmu_x - i)**2 + (rmu_y - j)**2
                    eDist = np.exp(-dist / (2.0 * (radius**2)))
                    self.node_list[self.map[(i, j)]] += learning_rate * eDist * error
        return rmu_index, rmu_x, rmu_y, error_val

    def smooth(self, data, radius, learning_rate):
        for data_index in range(data.shape[0]):
            self.winner_identification_and_neighbourhood_update(data_index, data, radius, learning_rate)

    def grow(self, data, radius, learning_rate):
        for data_index in range(data.shape[0]):
            rmu_index, rmu_x, rmu_y, error_val = self.winner_identification_and_neighbourhood_update(
                data_index, data, radius, learning_rate)
            self.node_errors[rmu_index] += error_val
            if self.node_errors[rmu_index] > self.growth_threshold:
                self.adjust_weights(rmu_x, rmu_y, rmu_index)

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
        out = scipy.spatial.distance.cdist(self.node_list[:self.node_count], data_n, self.distance)
        data_out = pd.DataFrame(data[output_columns])
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

    def separate_clusters(self, data, max_clusters=4):
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

    def analyze_idionomic_features(self, data, index_col, label_col=None, n_samples=100):
        if self.output is None:
            raise ValueError("Run predict() first to assign data points to nodes.")
        
        weight_columns = list(data.columns.values)
        if label_col:
            weight_columns.remove(label_col)
        weight_columns.remove(index_col)
        data_n = data[weight_columns].to_numpy()
        
        # Use existing node assignments from self.output
        winner_indices = self.output["output"].to_numpy()
        
        idionomic_data = []
        
        for idx, (name, row) in enumerate(tqdm(data[[index_col]].iterrows(), total=len(data), desc="Computing Shapley Values")):
            node_idx = winner_indices[idx]
            data_point = data_n[idx].reshape(1, -1)
            
            # Define value function with node_idx bound
            def value_function(data_subset, node_idx=node_idx):
                if data_subset.ndim == 1:
                    data_subset = data_subset.reshape(1, -1)
                distances = scipy.spatial.distance.cdist(
                    self.node_list[node_idx].reshape(1, -1),
                    data_subset,
                    self.distance
                )
                return -distances[0, 0]  # Negative distance as value (higher is better)
            
            # Create explainer for this data point
            explainer = shap.SamplingExplainer(value_function, data_n, feature_names=weight_columns)
            
            # Compute Shapley values
            shap_values = explainer.shap_values(data_point, nsamples=n_samples)
            
            # Normalize to sum to 1
            shap_sum = np.sum(np.abs(shap_values))
            if shap_sum == 0:
                feature_importance = np.ones_like(shap_values) / len(shap_values)
            else:
                feature_importance = np.abs(shap_values) / shap_sum
            
            idionomic_data.append({
                "name": name,
                "node_idx": node_idx,
                "x": self.node_coordinate[node_idx, 0],
                "y": self.node_coordinate[node_idx, 1],
                "feature_importance": feature_importance.tolist(),
                "label": data[label_col][idx] if label_col else None
            })
        
        idionomic_df = pd.DataFrame(idionomic_data)
        idionomic_df.to_csv("idionomic_features.csv", index=False)
        return idionomic_df, weight_columns

def plot_dsm(gsom, output, clusters, segments, pos_edges, idionomic_df, feature_names, index_col, label_col=None, 
             file_name="gsom_dsm", file_type=".pdf", figure_label="GSOM Data Skeleton Model with Idionomic Features (Shapley Values)"):
    max_count = output["hit_count"].max()
    listed_color_map = colors.ListedColormap(
        cm.get_cmap("Paired", max_count + 1)(np.arange(max_count + 1)) * 0.9, 
        name='gsom_color_list'
    )
    
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.scatter(gsom.node_coordinate[:gsom.node_count, 0], 
               gsom.node_coordinate[:gsom.node_count, 1], 
               c='gray', s=10, alpha=0.2, label='All Nodes')
    
    overlaps = Counter((min(i, j), max(i, j)) for i, j, _ in segments)
    q3 = np.percentile(list(overlaps.values()), 75) if overlaps else 1
    for i, j, dist in segments:
        key = (min(i, j), max(i, j))
        if overlaps[key] < q3:
            color, lw, alpha, ls = 'gray', 0.2, 0.3, '--'
        else:
            color = 'black' if (i, j) in pos_edges or (j, i) in pos_edges else 'red'
            lw, alpha, ls = 0.7, 0.5, '-'
        x1, y1 = gsom.node_coordinate[i]
        x2, y2 = gsom.node_coordinate[j]
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, alpha=alpha, linestyle=ls)
    
    cluster_colors = ['green', 'red', 'blue', 'cyan', 'purple', 'orange', 'yellow']
    for idx, cluster in enumerate(clusters[-1]):
        for node_idx in cluster:
            x, y = gsom.node_coordinate[node_idx]
            ax.scatter(x, y, c=[cluster_colors[idx % len(cluster_colors)]], 
                      s=20, alpha=0.5, label=f'Cluster {idx + 1}' if node_idx == min(cluster) else None)
    
    for _, row in output.iterrows():
        x, y = row['x'], row['y']
        hit_count = row['hit_count']
        if hit_count > 0:
            ax.scatter(x, y, c=[listed_color_map.colors[hit_count]], s=50, marker='D', 
                      alpha=0.8, label=f'Hit Count {hit_count}' if hit_count == max_count else None)
            names = ", ".join(map(str, row[index_col][:3]))
            node_idx = row['output']
            node_idionomic = idionomic_df[idionomic_df['node_idx'] == node_idx]
            top_features = []
            for _, idio_row in node_idionomic.iterrows():
                importance = np.array(idio_row['feature_importance'])
                top_idx = np.argmax(importance)
                top_features.append(f"{idio_row['name']}:{feature_names[top_idx]}")
            feature_label = "; ".join(top_features[:3])
            ax.text(x, y + 0.1, names, ha='left', va='bottom', fontsize=6, wrap=True)
            ax.text(x, y - 0.1, feature_label, ha='left', va='top', fontsize=5, wrap=True, color='darkblue')
    
    ax.set_title(figure_label)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.savefig(file_name + file_type, bbox_inches='tight')
    plt.close()

def plot_idionomic_heatmap(idionomic_df, feature_names, file_name="idionomic_heatmap", file_type=".pdf"):
    importance_matrix = np.array(idionomic_df['feature_importance'].tolist())
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(importance_matrix, xticklabels=feature_names, yticklabels=idionomic_df['name'],
                cmap='viridis', cbar_kws={'label': 'Feature Importance (Shapley)'})
    ax.set_title("Idionomic Feature Importance Heatmap (Shapley Values)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(file_name + file_type, bbox_inches='tight')
    plt.close()

def plot_node_feature_importance(idionomic_df, output, feature_names, file_dir="node_feature_plots", file_type=".pdf"):
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    
    for node_idx in output[output['hit_count'] > 0]['output']:
        node_idionomic = idionomic_df[idionomic_df['node_idx'] == node_idx]
        if not node_idionomic.empty:
            importance_matrix = np.array(node_idionomic['feature_importance'].tolist())
            mean_importance = np.mean(importance_matrix, axis=0)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(feature_names, mean_importance, color='skyblue')
            ax.set_title(f"Node {node_idx} Feature Importance (Hit Count: {output[output['output'] == node_idx]['hit_count'].iloc[0]})")
            ax.set_xlabel("Features")
            ax.set_ylabel("Average Shapley Value Importance")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(file_dir, f"node_{node_idx}_features{file_type}"), bbox_inches='tight')
            plt.close()

def calculate_cluster_metrics(data, gsom, clusters, output, index_col):
    cluster_labels = np.array([-1] * data.shape[0])
    node_to_cluster = {node_idx: cluster_id for cluster_id, cluster in enumerate(clusters[-1]) for node_idx in cluster}
    
    for idx, row in output.iterrows():
        node_idx = row['output']
        if node_idx in node_to_cluster:
            for name in row[index_col]:
                data_index = gsom.output[gsom.output[index_col] == name].index[0]
                cluster_labels[data_index] = node_to_cluster[node_idx]
    
    valid_indices = cluster_labels != -1
    if np.sum(valid_indices) < 2 or len(np.unique(cluster_labels[valid_indices])) < 2:
        print("Not enough valid clusters for metric calculation.")
        return None, None
    
    valid_data = data[valid_indices]
    valid_labels = cluster_labels[valid_indices]
    
    try:
        sil_score = silhouette_score(valid_data, valid_labels, metric='euclidean')
    except ValueError as e:
        print(f"Error calculating Silhouette Score: {e}")
        sil_score = None
    
    try:
        db_score = davies_bouldin_score(valid_data, valid_labels)
    except ValueError as e:
        print(f"Error calculating Davies-Bouldin Score: {e}")
        db_score = None
    
    return sil_score, db_score

if __name__ == '__main__':
    np.random.seed(1)
    # Load Zoo dataset
    data_filename = "zoo.txt".replace('\\', '/')
    df = pd.read_csv(data_filename)
    data_training = df.iloc[:, 1:17]
    
    # Train GSOM
    gsom = GSOM(spread_factor=0.83, dimensions=16, max_radius=4, initial_node_size=1000)
    gsom.fit(data_training.to_numpy(), training_iterations=100, smooth_iterations=50)
    output = gsom.predict(df, index_col="Name", label_col="label")
    output.to_csv("gsom_output.csv", index=False)
    print("GSOM training completed. Node Count:", gsom.node_count)
    
    # Build skeleton and separate clusters
    clusters, segments, remaining_connections, pos_edges = gsom.separate_clusters(
        data_training.to_numpy(), max_clusters=7)
    
    # Analyze idionomic features with Shapley values
    idionomic_df, feature_names = gsom.analyze_idionomic_features(df, index_col="Name", label_col="label", n_samples=100)
    print("Idionomic features (Shapley values) saved to idionomic_features.csv")
    
    # Save segments to CSV
    segment_df = pd.DataFrame(segments, columns=["node1", "node2", "distance"])
    segment_df.to_csv("segment_distances.csv", index=False)
    
    # Save clusters to CSV
    clusters_data = [
        {"cluster_id": idx + 1, "node_indices": ";".join(map(str, cluster)), 
         "size": len(cluster), "color": ['green', 'red', 'blue', 'cyan', 'purple', 'orange', 'yellow'][idx % 7]}
        for idx, cluster in enumerate(clusters[-1])
    ]
    clusters_df = pd.DataFrame(clusters_data)
    clusters_df.to_csv("clusters_zoo.csv", index=False)
    
    # Plot DSM with idionomic feature annotations
    plot_dsm(gsom, output, clusters, segments, pos_edges, idionomic_df, feature_names,
             index_col="Name", label_col="label", 
             file_name="gsom_dsm_zoo", file_type=".pdf", 
             figure_label="GSOM Data Skeleton Model with Idionomic Features (Shapley Values)")
    
    # Plot idionomic feature heatmap
    plot_idionomic_heatmap(idionomic_df, feature_names, file_name="idionomic_heatmap_zoo", file_type=".pdf")
    
    # Plot node-level feature importance
    plot_node_feature_importance(idionomic_df, output, feature_names, file_dir="node_feature_plots", file_type=".pdf")
    
    # Calculate and print clustering metrics
    sil_score, db_score = calculate_cluster_metrics(data_training.to_numpy(), gsom, clusters, output, index_col="Name")
    print(f"Silhouette Score: {sil_score}")
    print(f"Davies-Bouldin Index: {db_score}")
    
    # Export path tree to DOT file
    graph = tree_to_dot(gsom.path_tree)
    graph.write_png("path_tree_zoo.png")
    
    print("DSM and all idionomic visualizations (Shapley values) complete.")
