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
        # for i, j, dist in segments:
        #     if (i, j) in remaining_connections:
        #         remaining_connections.remove((i, j))
        #     G.remove_edge(i, j)
        #     clusters.append(list(nx.connected_components(G)))
        #     print(f"Removed segment {i}-{j}, Distance: {dist}")
        #     if len(clusters[-1]) >= max_clusters:
        #         break
        for i, j, dist in segments:
            if (i, j) in remaining_connections:
                remaining_connections.remove((i, j))
            # Only remove the edge if it exists
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

def _get_color_map_pos(max_count, alpha=0.9, cmap_colors="Paired"):
    np.random.seed(1)
    cmap = cm.get_cmap(cmap_colors, max_count + 1)
    color_list = [(c[0] * alpha, c[1] * alpha, c[2] * alpha) for c in cmap(np.arange(cmap.N))]
    return colors.ListedColormap(color_list, name='gsom_color_list')


from bigtree import Node, tree_to_dot
import pandas as pd

def generate_filtered_gsom_tree(gsom, output_csv="output.csv", output_path="filtered_tree.png"):
    """
    Generate and export a GSOM path tree that only includes nodes with original data mappings.
    Works with bigtree structure.
    """
    print("Generating filtered GSOM path tree...")

    # Load output.csv and get node indices with data
    output_df = pd.read_csv(output_csv)
    data_nodes = set(output_df['output'])
    label_map = output_df.set_index('output')['Name'].to_dict()

    # Recursively clone only relevant tree nodes
    def clone_relevant_nodes(node):
        if node.name == "root":
            new_node = Node("root", x=node.x, y=node.y, node_number=node.node_number)
        else:
            try:
                node_index = int(node.name)
            except:
                return None
            include_node = node_index in data_nodes
            label = label_map.get(node_index, "")
            new_name = f"{node_index}\n{label}" if include_node else node.name
            new_node = Node(new_name, x=node.x, y=node.y, node_number=node.node_number)

        for child in node.children:
            filtered_child = clone_relevant_nodes(child)
            if filtered_child:
                filtered_child.parent = new_node

        # Keep node if it's in data or has children
        if new_node.name == "root" or int(node.name) in data_nodes or new_node.children:
            return new_node
        return None

    filtered_root = clone_relevant_nodes(gsom.path_tree)

    if filtered_root:
        dot_graph = tree_to_dot(filtered_root)
        dot_graph.write_png(output_path)
        print(f"✅ Filtered path tree saved to: {output_path}")
    else:
        print("⚠️ No data-mapped nodes found to export.")

if __name__ == '__main__':
    import os
    np.random.seed(1)

    # Define output folder and create if it doesn't exist
    output_folder = "output_zoo"
    os.makedirs(output_folder, exist_ok=True)

    # Load Zoo dataset
    data_filename = "example/data/zoo.txt".replace('\\', '/')
    df = pd.read_csv(data_filename)

    print("Dataset shape:", df.shape)
    data_training = df.iloc[:, 1:17]
    print(type(data_training))
    print("Training data head:", data_training.head())
    print("Training data shape:", data_training.shape)

    # Train GSOM
    gsom = GSOM(0.83, 16, max_radius=4, initial_node_size=1000)
    gsom.fit(data_training.to_numpy(), 100, 50)
    output = gsom.predict(df, "Name", "label")
    output.to_csv(os.path.join(output_folder, "output.csv"), index=False)
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
    paths_df.to_csv(os.path.join(output_folder, "paths_of_spread.csv"), index=False)

    # Build skeleton and separate clusters
    clusters, segments, skeleton_connections, pos_edges = gsom.separate_clusters(data_training.to_numpy(), max_clusters=7)

    # Plot GSOM map and paths
    plot(output, "Name", gsom_map=gsom, file_name=os.path.join(output_folder, "gsom_map"))
    plot_pos(output, "Name", gsom_map=gsom, file_name=os.path.join(output_folder, "gsom_with_paths_sk"),
             figure_label="GSOM Map with Node Paths", n_nodes=gsom.node_count)

    # Plot skeleton with clusters
    fig, ax = plt.subplots(figsize=(10, 10))
    for i in range(gsom.node_count):
        x, y = gsom.node_coordinate[i]
        if i in gsom.node_labels['output'].values:
            ax.scatter(x, y, c='blue', s=30, marker='D', alpha=0.3)
        else:
            ax.scatter(x, y, c='gray', s=10, marker='o', alpha=0.1)
        ax.text(x, y, str(i), fontsize=6)

    from collections import Counter
    overlaps = Counter((i, j) for i, j in skeleton_connections)

    # Save overlaps
    overlap_df = pd.DataFrame([(i, j, count) for (i, j), count in overlaps.items()],
                              columns=["node1", "node2", "overlap_count"])
    overlap_df.to_csv(os.path.join(output_folder, "edge_overlaps.csv"), index=False)

    counts = np.array(list(overlaps.values()))
    q1 = np.percentile(counts, 25)
    median = np.percentile(counts, 50)
    q3 = np.percentile(counts, 75)
    print(f"Q1: {q1}, Median: {median}, Q3: {q3}")
    print(f"Mean: {np.mean(counts)}, Std Dev: {np.std(counts)}")

    for i, j in skeleton_connections:
        if overlaps[(i, j)] < q3:
            color, alpha, line_style = 'gray', 0.3, '--'
        else:
            color = 'black' if (i, j) in pos_edges or (j, i) in pos_edges else 'red'
            alpha = 0.5 if color == 'black' else 0.1
            line_style = '-'
        x1, y1 = gsom.node_coordinate[i]
        x2, y2 = gsom.node_coordinate[j]
        ax.plot([x1, x2], [y1, y2], color=color, linestyle=line_style, alpha=alpha)

    # Save clusters
    colors_list = ['green', 'red', 'black', 'cyan']
    clusters_data = []
    for idx, cluster in enumerate(clusters[-1]):
        color = colors_list[idx % len(colors_list)]
        clusters_data.append({
            "cluster_id": idx + 1,
            "node_indices": ";".join(map(str, cluster)),
            "color": color,
            "size": len(cluster)
        })
        for node_idx in cluster:
            x, y = gsom.node_coordinate[node_idx]
            ax.scatter(x, y, c=color, s=20, marker='o', alpha=0.5)
    clusters_df = pd.DataFrame(clusters_data)
    clusters_df.to_csv(os.path.join(output_folder, "clusters_zoo.csv"), index=False)

    ax.set_title("GSOM Skeleton with Clusters")
    plt.savefig(os.path.join(output_folder, "gsom_skeleton.pdf"))
    plt.show()

    # Save segment distances
    segment_df = pd.DataFrame(segments, columns=["node1", "node2", "distance"])
    segment_df.to_csv(os.path.join(output_folder, "segment_distances.csv"), index=False)

    # Export full path tree
    graph = tree_to_dot(gsom.path_tree)
    graph.write_png(os.path.join(output_folder, "path_tree.png"))

    # Export filtered tree (only original data-mapped nodes)
    generate_filtered_gsom_tree(gsom,
                                 output_csv=os.path.join(output_folder, "output.csv"),
                                 output_path=os.path.join(output_folder, "filtered_tree.png"))

    print("Complete")
