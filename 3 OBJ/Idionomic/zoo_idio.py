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
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        self.dimensions = dimensions
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
        self.path_tree = Node("root", x=0.01, y=0.01, node_number=-1, distance=0)
        self.initialize_GSOM()

    def initialize_GSOM(self):
        for x, y in [(1, 1), (1, 0), (0, 1), (0, 0)]:
            self.insert_node_with_weights(x, y)

    def insert_new_node(self, x, y, weights, parent_node=None):
        if self.node_count >= self.initial_node_size:
            raise MemoryError("Node size out of bound")
        self.map[(x, y)] = self.node_count
        self.node_list[self.node_count] = weights
        self.node_coordinate[self.node_count] = [x, y]
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
        self.node_count += 1
        return new_node

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
        weights = (self.node_list[self.map[(winnerx, winnery)]] + self.node_list[self.map[(next_nodex, next_nodey)]]) * 0.5
        return weights

    def _new_weights_for_new_node_on_one_side(self, winnerx, winnery, next_nodex, next_nodey):
        weights = (2 * self.node_list[self.map[(winnerx, winnery)]] - self.node_list[self.map[(next_nodex, next_nodey)]])
        return weights

    def _new_weights_for_new_node_one_older_neighbour(self, winnerx, winnery):
        weights = np.full(self.dimensions, (max(self.node_list[self.map[(winnerx, winnery)]]) + min(self.node_list[self.map[(winnerx, winnery)]])) / 2)
        return weights

    def grow_node(self, wx, wy, x, y, side):
        if (x, y) not in self.map:
            if side == 0:  # left
                weights = self._new_weights_for_new_node_in_middle(wx, wy, x - 1, y) if (x - 1, y) in self.map else \
                          self._new_weights_for_new_node_on_one_side(wx, wy, wx + 1, wy) if (wx + 1, wy) in self.map else \
                          self._new_weights_for_new_node_on_one_side(wx, wy, wx, wy + 1) if (wx, wy + 1) in self.map else \
                          self._new_weights_for_new_node_on_one_side(wx, wy, wx, wy - 1) if (wx, wy - 1) in self.map else \
                          self._new_weights_for_new_node_one_older_neighbour(wx, wy)
            elif side == 1:  # right
                weights = self._new_weights_for_new_node_in_middle(wx, wy, x + 1, y) if (x + 1, y) in self.map else \
                          self._new_weights_for_new_node_on_one_side(wx, wy, wx - 1, wy) if (wx - 1, wy) in self.map else \
                          self._new_weights_for_new_node_on_one_side(wx, wy, wx, wy + 1) if (wx, wy + 1) in self.map else \
                          self._new_weights_for_new_node_on_one_side(wx, wy, wx, wy - 1) if (wx, wy - 1) in self.map else \
                          self._new_weights_for_new_node_one_older_neighbour(wx, wy)
            elif side == 2:  # top
                weights = self._new_weights_for_new_node_in_middle(wx, wy, x, y + 1) if (x, y + 1) in self.map else \
                          self._new_weights_for_new_node_on_one_side(wx, wy, wx, wy - 1) if (wx, wy - 1) in self.map else \
                          self._new_weights_for_new_node_on_one_side(wx, wy, wx + 1, wy) if (wx + 1, wy) in self.map else \
                          self._new_weights_for_new_node_on_one_side(wx, wy, wx - 1, wy) if (wx - 1, wy) in self.map else \
                          self._new_weights_for_new_node_one_older_neighbour(wx, wy)
            elif side == 3:  # bottom
                weights = self._new_weights_for_new_node_in_middle(wx, wy, x, y - 1) if (x, y - 1) in self.map else \
                          self._new_weights_for_new_node_on_one_side(wx, wy, wx, wy + 1) if (wx, wy + 1) in self.map else \
                          self._new_weights_for_new_node_on_one_side(wx, wy, wx + 1, wy) if (wx + 1, wy) in self.map else \
                          self._new_weights_for_new_node_on_one_side(wx, wy, wx - 1, wy) if (wx - 1, wy) in self.map else \
                          self._new_weights_for_new_node_one_older_neighbour(wx, wy)
            else:
                raise ValueError("Invalid side specified")

            weights = np.clip(weights, 0.0, 1.0)
            parent_node = find(self.path_tree, lambda node: node.x == wx and node.y == wy)
            self.insert_new_node(x, y, weights, parent_node=parent_node)

    def spread_wights(self, x, y):
        leftx, lefty = x - 1, y
        rightx, righty = x + 1, y
        topx, topy = x, y + 1
        bottomx, bottomy = x, y - 1
        self.node_errors[self.map[(x, y)]] = self.groth_threshold / 2
        for (nx, ny) in [(leftx, lefty), (rightx, righty), (topx, topy), (bottomx, bottomy)]:
            if (nx, ny) in self.map:
                self.node_errors[self.map[(nx, ny)]] *= (1 + self.FD)

    # def adjust_weights(self, x, y, rmu_index):
    #     leftx, lefty = x - 1, y
    #     rightx, righty = x + 1, y
    #     topx, topy = x, y + 1
    #     bottomx, bottomy = x, y - 1
    #     if all((nx, ny) in self.map for (nx, ny) in [(leftx, lefty), (rightx, righty), (topx, topy), (bottomx, bottomy)]):
    #         self.spread_weights(x, y)
    #     else:
    #         for (nx, ny), side in [(leftx, lefty, 0), (rightx, righty, 1), (topx, topy, 2), (bottomx, bottomy, 3)]:
    #             self.grow_node(x, y, nx, ny, side)
    #     self.node_errors[rmu_index] = self.groth_threshold / 2

    def adjust_weights(self, x, y, rmu_index):
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
        out = scipy.spatial.distance.cdist(self.node_list[:self.node_count], data[data_index].reshape(1, self.dimensions), self.distance)
        rmu_index = out.argmin()
        error_val = out.min()
        rmu_x, rmu_y = self.node_coordinate[rmu_index]

        error = data[data_index] - self.node_list[rmu_index]
        self.node_list[rmu_index] += learning_rate * error

        mask_size = round(radius)
        for i in range(int(rmu_x) - mask_size, int(rmu_x) + mask_size + 1):
            for j in range(int(rmu_y) - mask_size, int(rmu_y) + mask_size + 1):
                if (i, j) in self.map and (i != rmu_x or j != rmu_y):
                    dist = (rmu_x - i)**2 + (rmu_y - j)**2
                    eDistance = np.exp(-dist / (2.0 * (radius**2)))
                    error = self.node_list[rmu_index] - self.node_list[self.map[(i, j)]]
                    self.node_list[self.map[(i, j)]] += learning_rate * eDistance * error
        return rmu_index, rmu_x, rmu_y, error_val

    def smooth(self, data, radius, learning_rate):
        for data_index in range(data.shape[0]):
            self.winner_identification_and_neighbourhood_update(data_index, data, radius, learning_rate)

    def grow(self, data, radius, learning_rate):
        for data_index in range(data.shape[0]):
            rmu_index, rmu_x, rmu_y, error_val = self.winner_identification_and_neighbourhood_update(data_index, data, radius, learning_rate)
            self.node_errors[rmu_index] += error_val
            if self.node_errors[rmu_index] > self.groth_threshold:
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
        individual_mapping = {}  # Map node indices to individual data point indices
        for data_idx, node_idx in enumerate(winner_indices):
            if node_idx in individual_mapping:
                individual_mapping[node_idx].append(data_idx)
            else:
                individual_mapping[node_idx] = [data_idx]

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
        
        logging.info(f"Skeleton connections: {skeleton_connections}")
        logging.info(f"Individual mapping: {individual_mapping}")
        return hit_points, skeleton_connections, junctions, pos_edges, individual_mapping

    def separate_clusters(self, data, max_clusters=4):
        hit_points, skeleton_connections, junctions, pos_edges, individual_mapping = self.build_skeleton(data)
        segments = []
        valid_skeleton_connections = []
        for i, j in skeleton_connections:
            if 0 <= i < self.node_count and 0 <= j < self.node_count:  # Bounds checking
                if (i in hit_points or i in junctions) and (j in hit_points or j in junctions):
                    dist = scipy.spatial.distance.cdist(
                        self.node_list[i].reshape(1, -1),
                        self.node_list[j].reshape(1, -1),
                        self.distance
                    )[0][0]
                    segments.append((i, j, dist))
                    valid_skeleton_connections.append((i, j))
            else:
                logging.warning(f"Invalid index pair ({i}, {j}) skipped in skeleton_connections")
        
        segments.sort(key=lambda x: x[2], reverse=True)
        
        G = nx.Graph(valid_skeleton_connections)
        clusters = []
        remaining_connections = valid_skeleton_connections.copy()
        for i, j, dist in segments:
            if (i, j) in remaining_connections and G.has_edge(i, j):
                remaining_connections.remove((i, j))
                G.remove_edge(i, j)
                clusters.append(list(nx.connected_components(G)))
                logging.info(f"Removed segment {i}-{j}, Distance: {dist}")
                if len(clusters) >= max_clusters:  # Check number of cluster sets
                    break
        
        # Flatten clusters[-1] to get individual node indices
        hit_points = [item for sublist in clusters[-1] for item in sublist] if clusters else hit_points
        return clusters, segments, remaining_connections, pos_edges, individual_mapping
def plot(output, index_col, gsom_map=None, hit_points=None, skeleton_connections=None, junctions=None,
         pos_edges=None, file_name="gsom", file_type=".png", figure_label="GSOM Map with Skeleton",
         max_text=3, max_length=30, cmap_colors="Paired", show_index=True, overlap_threshold=None):
    max_count = output["hit_count"].max()
    listed_color_map = _get_color_map(max_count, alpha=0.9, cmap_colors=cmap_colors)
    fig, ax = plt.subplots(figsize=(12, 10))
    
    if gsom_map:
        ax.scatter(gsom_map.node_coordinate[:gsom_map.node_count, 0],
                   gsom_map.node_coordinate[:gsom_map.node_count, 1],
                   c='gray', s=20, alpha=0.5, label='All Nodes')
    
    if skeleton_connections and gsom_map:
        from collections import Counter
        overlaps = Counter((i, j) for i, j in skeleton_connections if 0 <= i < gsom_map.node_count and 0 <= j < gsom_map.node_count)
        logging.info(f"Edge overlaps: {overlaps}")
        
        total_overlaps = sum(count - 1 for count in overlaps.values() if count > 1)
        logging.info(f"Total number of overlapping edges (count > 1): {total_overlaps}")
        top_overlaps = overlaps.most_common(3)
        logging.info(f"Top 3 overlapping edges: {top_overlaps}")
        
        if overlap_threshold is None:
            counts = np.array(list(overlaps.values()))
            if len(counts) > 0:
                overlap_threshold = np.percentile(counts, 25)
                logging.info(f"Calculated overlap threshold (25th percentile): {overlap_threshold}")
            else:
                overlap_threshold = 0
                logging.warning("No edges in skeleton_connections. Setting threshold to 0.")
        
        plotted_edges = set()
        for (i, j), count in overlaps.items():
            if count > overlap_threshold and 0 <= i < gsom_map.node_count and 0 <= j < gsom_map.node_count:
                edge = tuple(sorted((i, j)))
                if edge not in plotted_edges:
                    plotted_edges.add(edge)
                    x1, y1 = gsom_map.node_coordinate[i]
                    x2, y2 = gsom_map.node_coordinate[j]
                    if not (np.isnan(x1) or np.isnan(x2) or np.isnan(y1) or np.isnan(y2)):
                        color = 'black' if (i, j) in pos_edges or (j, i) in pos_edges else 'red'
                        alpha = 0.5 if (i, j) in pos_edges or (j, i) in pos_edges else 0.2
                        linewidth = 2.0 if (i, j) in pos_edges or (j, i) in pos_edges else 1.0
                        ax.plot([x1, x2], [y1, y2], color=color, linestyle='-', alpha=alpha, linewidth=linewidth)
    
    for index, i in output.iterrows():
        x = i['x']
        y = i['y']
        if hit_points and i['output'] in hit_points:
            ax.scatter(x, y, c=[listed_color_map.colors[i['hit_count']]], s=100, 
                       marker='o', label=f'Hit Point (Count: {i["hit_count"]})' if index == 0 else "")
            if show_index and i['hit_count'] > 0:
                label = ", ".join(map(str, i[index_col][0:max_text]))
                ax.text(x + 0.1, y + 0.1, label, ha='left', va='center', fontsize=8)
    
    if junctions and gsom_map:
        print(f"Junctions: {junctions}")
        for j in junctions:
            if isinstance(j, (int, np.integer)) and 0 <= j < gsom_map.node_count:
                x, y = gsom_map.node_coordinate[j]
                if not (np.isnan(x) or np.isnan(y)):
                    ax.scatter(x, y, c='green', s=150, marker='s', label='Junction' if j == junctions[0] else "")
    
    ax.set_title(figure_label)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.savefig(file_name + file_type, bbox_inches='tight')
    plt.show()
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

def plot_pos(output, index_col, gsom_map=None, file_name="gsom", file_type=".png", 
             figure_label="GSOM Map with Paths", max_text=3, max_length=30, 
             cmap_colors="Paired", show_index=True, n_nodes=None):
    if n_nodes is None:
        n_nodes = gsom_map.node_count if gsom_map else 180
    max_count = output["hit_count"].max()
    listed_color_map = _get_color_map_pos(max_count, alpha=0.9, cmap_colors=cmap_colors)
    fig, ax = plt.subplots(figsize=(10, 8))
    if gsom_map:
        ax.scatter(gsom_map.node_coordinate[:n_nodes, 0],
                   gsom_map.node_coordinate[:n_nodes, 1],
                   c='gray', s=20, alpha=0.5, label='All Nodes')
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
        ax.scatter(x, y, c=listed_color_map.colors[i['hit_count']],
                   s=50 if i['hit_count'] > 0 else 20, alpha=0.7,
                   label=f'Hit Count {i["hit_count"]}' if i['hit_count'] > 0 and index == 0 else "")
        if show_index and i['hit_count'] > 0:
            label = ", ".join(map(str, i[index_col][0:max_text]))
            ax.text(x + 0.1, y + 0.1, label, ha='left', va='center', fontsize=8, wrap=True)
    ax.set_title(figure_label)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
    plt.savefig(file_name + file_type, bbox_inches='tight')
    plt.show()

def _get_color_map_pos(max_count, alpha=0.9, cmap_colors="Paired"):
    np.random.seed(1)
    cmap = cm.get_cmap(cmap_colors, max_count + 1)
    color_list = [(c[0] * alpha, c[1] * alpha, c[2] * alpha) for c in cmap(np.arange(cmap.N))]
    return colors.ListedColormap(color_list, name='gsom_color_list')

if __name__ == '__main__':
    np.random.seed(1)
    # Load Zoo dataset
    data_filename = "example/data/zoo.txt".replace('\\', '/')
    df = pd.read_csv(data_filename)
    
    print("Dataset shape:", df.shape)
    data_training = df.iloc[:, 1:17].to_numpy()
    print("Training data shape:", data_training.shape)
    
    # Train GSOM
    gsom = GSOM(0.83, 16, max_radius=4, initial_node_size=1000)
    gsom.fit(data_training, 100, 50)
    output = gsom.predict(df, "Name", "label")
    output.to_csv("output.csv", index=False)
    print("GSOM training completed.")
    print("Output shape:", output.shape)
    print("Node Count:", gsom.node_count)
    
    # Get paths of spread
    paths = gsom.get_paths()
    paths_df = pd.DataFrame({"node_names": [";".join([node.name for node in path]) for path in paths],
                             "node_coords": [";".join([f"({node.x},{node.y})" for node in path if hasattr(node, 'x')]) for path in paths]})
    paths_df.to_csv("paths_of_spread.csv", index=False)
    
    # Build skeleton and separate clusters with individual mapping
    clusters, segments, remaining_connections, pos_edges, individual_mapping = gsom.separate_clusters(data_training, max_clusters=7)
    
    # Plot GSOM map with skeleton, clusters, and individual mappings
    plot(output, "Name", gsom_map=gsom, hit_points=clusters[-1], skeleton_connections=remaining_connections,
         junctions=clusters[-1], pos_edges=pos_edges, file_name="gsom_skeleton", file_type=".png",
         figure_label="GSOM Skeleton with Clusters and Individuals")
    
    fig, ax = plt.subplots(figsize=(10, 10))
    # Visualize individual mappings
    colors = ['green', 'red', 'blue', 'cyan', 'magenta', 'yellow', 'orange']
    for node_idx, data_indices in individual_mapping.items():
        x, y = gsom.node_coordinate[node_idx]
        num_individuals = len(data_indices)
        color_idx = min(num_individuals, len(colors) - 1)
        ax.scatter(x, y, c=colors[color_idx], s=50 * num_individuals, alpha=0.6,
                   label=f'Node {node_idx} (Individuals: {num_individuals})' if node_idx == list(individual_mapping.keys())[0] else "")
    ax.set_title("Individual Mappings in GSOM")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.grid(True)
    plt.tight_layout()
    plt.xticks(rotation=45)
    
    # Save individual mappings to CSV
    individual_df = pd.DataFrame([(node_idx, ";".join(map(str, indices)), len(indices))
                                 for node_idx, indices in individual_mapping.items()],
                                columns=["node_idx", "data_indices", "num_individuals"])
    individual_df.to_csv("individual_mappings.csv", index=False)
    
    plt.savefig("gsom_individual_mappings.png", bbox_inches='tight')
    plt.show()
    
    # Export path tree to DOT file
    graph = tree_to_dot(gsom.path_tree)
    graph.write_png("path_tree.png")

    print("Complete")