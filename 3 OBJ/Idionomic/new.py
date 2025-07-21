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
import logging
import shutil
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from collections import Counter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# GSOM class with DSM enhancements
class GSOM:
    def __init__(self, spred_factor, dimensions, distance='euclidean', initialize='random', learning_rate=0.3,
                 smooth_learning_factor=0.8, max_radius=6, FD=0.1, r=3.8, alpha=0.9, initial_node_size=1000):
        self.initial_node_size = initial_node_size
        self.node_count = 0
        self.map = {}
        self.node_list = np.zeros((self.initial_node_size, dimensions), dtype=np.float64)
        self.node_coordinate = np.zeros((self.initial_node_size, 2), dtype=np.float64)
        self.node_errors = np.zeros(self.initial_node_size, dtype=np.longdouble)
        self.spred_factor = spred_factor
        self.growth_threshold = -dimensions * math.log(self.spred_factor)
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
        self.path_tree = None
        self.initialize_GSOM()

    def initialize_GSOM(self):
        self.path_tree = Node("root", x=0.01, y=0.01, node_number=-1, distance=0)
        for x, y in [(1, 1), (1, 0), (0, 1), (0, 0)]:
            self.insert_node_with_weights(x, y)

    def insert_new_node(self, x, y, weights, parent_node=None):
        if self.node_count >= self.initial_node_size:
            raise MemoryError("Node size out of bound")
        if x < 0 or y < 0:
            logging.warning(f"Skipping node creation at ({x}, {y}): Negative coordinates")
            return
        self.map[(x, y)] = self.node_count
        self.node_list[self.node_count] = weights.astype(np.float64)
        self.node_coordinate[self.node_count][0] = x
        self.node_coordinate[self.node_count][1] = y
        
        distance_from_parent = 0
        new_node = Node(str(self.node_count), x=x, y=y, node_number=self.node_count, distance=distance_from_parent)
        if parent_node is not None:
            if (parent_node.x, parent_node.y) in self.map:
                distance_from_parent = scipy.spatial.distance.cdist(
                    weights.reshape(1, -1).astype(np.float64),
                    self.node_list[self.map[(parent_node.x, parent_node.y)]].reshape(1, -1).astype(np.float64),
                    self.distance
                )[0][0]
                new_node.distance = distance_from_parent
            new_node.parent = parent_node
        else:
            new_node.parent = self.path_tree
        self.node_count += 1

    def insert_node_with_weights(self, x, y):
        if self.initialize == 'random':
            node_weights = np.random.rand(self.dimensions).astype(np.float64)
        else:
            raise NotImplementedError("Initialization method not supported")
        self.insert_new_node(x, y, node_weights, parent_node=self.path_tree)

    def _get_learning_rate(self, prev_learning_rate):
        return self.ALPHA * (1 - (self.R / max(1, self.node_count))) * prev_learning_rate

    def _get_neighbourhood_radius(self, total_iteration, iteration):
        time_constant = total_iteration / math.log(self.max_radius)
        return self.max_radius * math.exp(-iteration / time_constant)

    def _new_weights_for_new_node_in_middle(self, winnerx, winnery, next_nodex, next_nodey):
        weights = (self.node_list[self.map[(winnerx, winnery)]] + self.node_list[
            self.map[(next_nodex, next_nodey)]]) * 0.5
        return weights.astype(np.float64)

    def _new_weights_for_new_node_on_one_side(self, winnerx, winnery, next_nodex, next_nodey):
        weights = (2 * self.node_list[self.map[(winnerx, winnery)]] - self.node_list[
            self.map[(next_nodex, next_nodey)]])
        return weights.astype(np.float64)

    def _new_weights_for_new_node_one_older_neighbour(self, winnerx, winnery):
        weights = np.full(self.dimensions, (max(self.node_list[self.map[(winnerx, winnery)]]) + min(
            self.node_list[self.map[(winnerx, winnery)]])) / 2, dtype=np.float64)
        return weights

    def grow_node(self, wx, wy, x, y, side):
        if (x, y) not in self.map:
            if x < 0 or y < 0:
                logging.warning(f"Skipping node growth at ({x}, {y}): Negative coordinates")
                return
            if side == '0':  # left
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
            elif side == '1':  # right
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
            elif side == '2':  # top
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
            elif side == '3':  # bottom
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
                raise ValueError(f"Invalid side specified: {side}")
            
            weights = weights.astype(np.float64)
            weights[weights < 0] = 0.0
            weights[weights > 1] = 1.0
            parent_node = find(self.path_tree, lambda node: node.x == wx and node.y == wy)
            if parent_node is None:
                parent_node = self.path_tree
            self.insert_new_node(x, y, weights, parent_node=parent_node)

    def spread_weights(self, x, y):
        leftx, lefty = x - 1, y
        rightx, righty = x + 1, y
        topx, topy = x, y + 1
        bottomx, bottomy = x, y - 1
        self.node_errors[self.map[(x, y)]] = self.growth_threshold / 2
        if (leftx, lefty) in self.map:
            self.node_errors[self.map[(leftx, lefty)]] *= (1 + self.FD)
        if (rightx, righty) in self.map:
            self.node_errors[self.map[(rightx, righty)]] *= (1 + self.FD)
        if (topx, topy) in self.map:
            self.node_errors[self.map[(topx, topy)]] *= (1 + self.FD)
        if (bottomx, bottomy) in self.map:
            self.node_errors[self.map[(bottomx, bottomy)]] *= (1 + self.FD)

    def adjust_weights(self, x, y, rmu_index):
        leftx, lefty = x - 1, y
        rightx, righty = x + 1, y
        topx, topy = x, y + 1
        bottomx, bottomy = x, y - 1
        if (leftx, lefty) in self.map and (rightx, righty) in self.map and (topx, topy) in self.map and (bottomx, bottomy) in self.map:
            self.spread_weights(x, y)
        else:
            if leftx >= 0 and lefty >= 0:
                self.grow_node(x, y, leftx, lefty, '0')
            if rightx >= 0 and righty >= 0:
                self.grow_node(x, y, rightx, righty, '1')
            if topx >= 0 and topy >= 0:
                self.grow_node(x, y, topx, topy, '2')
            if bottomx >= 0 and bottomy >= 0:
                self.grow_node(x, y, bottomx, bottomy, '3')
        self.node_errors[rmu_index] = self.growth_threshold / 2

    def winner_identification_and_neighbourhood_update(self, data_index, data, radius, learning_rate):
        out = scipy.spatial.distance.cdist(
            self.node_list[:self.node_count].astype(np.float64),
            data[data_index, :].reshape(1, self.dimensions).astype(np.float64),
            self.distance
        )
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
                    distance = (rmu_x - i) ** 2 + (rmu_y - j) ** 2
                    eDistance = np.exp(-1.0 * distance / (2.0 * (radius ** 2)))
                    self.node_list[self.map[(i, j)]] += learning_rate * eDistance * error
        return rmu_index, rmu_x, rmu_y, error_val

    def smooth(self, data, radius, learning_rate):
        for data_index in range(data.shape[0]):
            self.winner_identification_and_neighbourhood_update(data_index, data, radius, learning_rate)

    def grow(self, data, radius, learning_rate):
        for data_index in range(data.shape[0]):
            rmu_index, rmu_x, rmu_y, error_val = self.winner_identification_and_neighbourhood_update(
                data_index, data, radius, learning_rate
            )
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
        
        if index_col not in weight_columns:
            raise ValueError(f"Index column '{index_col}' not found in DataFrame columns: {weight_columns}")
        
        weight_columns.remove(index_col)
        
        if label_col:
            if label_col not in data.columns:
                raise ValueError(f"Label column '{label_col}' not found in DataFrame columns: {data.columns}")
            weight_columns.remove(label_col)
            output_columns.append(label_col)
        
        data_n = data[weight_columns].to_numpy().astype(float)
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
        def collect_paths(node, current_path):
            current_path.append(node)
            if not node.children:
                paths.append(current_path[:])
            for child in node.children:
                collect_paths(child, current_path)
            current_path.pop()
        collect_paths(self.path_tree, [])
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
        individual_mapping = {}  # Map node indices to individual data point indices
        for data_idx, node_idx in enumerate(winner_indices):
            if node_idx in individual_mapping:
                individual_mapping[node_idx].append(data_idx)
            else:
                individual_mapping[node_idx] = [data_idx]
        
        for i in hit_points:
            if i not in pos_nodes:
                min_dist = float('inf')
                nearest = None
                for j in pos_nodes:
                    dist = scipy.spatial.distance.cdist(
                        self.node_list[i].reshape(1, -1).astype(np.float64),
                        self.node_list[j].reshape(1, -1).astype(np.float64),
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
        
        return hit_points, skeleton_connections, junctions, pos_edges, individual_mapping

def plot(output, index_col, gsom_map=None, hit_points=None, skeleton_connections=None, 
         junctions=None, pos_edges=None, data=None, individual_mapping=None, file_name="gsom", 
         file_type=".png", figure_label="GSOM Map with Skeleton", max_text=3, max_length=30, 
         cmap_colors="Paired", show_index=True, overlap_threshold=None):
    max_count = output["hit_count"].max()
    listed_color_map = _get_color_map(max_count, alpha=0.9, cmap_colors=cmap_colors)
    fig, ax = plt.subplots(figsize=(12, 10))
    
    if gsom_map:
        ax.scatter(gsom_map.node_coordinate[:gsom_map.node_count, 0],
                   gsom_map.node_coordinate[:gsom_map.node_count, 1],
                   c='gray', s=20, alpha=0.5, label='All Nodes')
    
    if skeleton_connections and gsom_map:
        overlaps = Counter((i, j) for i, j in skeleton_connections)
        logging.info(f"Edge overlaps: {overlaps}")
        
        total_overlaps = sum(count - 1 for count in overlaps.values() if count > 1)
        logging.info(f"Total number of overlapping edges (count > 1): {total_overlaps}")
        top_overlaps = overlaps.most_common(3)
        logging.info(f"Top 3 overlapping edges: {top_overlaps}")
        
        if overlap_threshold is None:
            counts = np.array(list(overlaps.values()))
            if len(counts) > 0:
                overlap_threshold = np.percentile(counts, 75)
                logging.info(f"Calculated overlap threshold (75th percentile): {overlap_threshold}")
            else:
                overlap_threshold = 0
                logging.warning("No edges in skeleton_connections. Setting threshold to 0")
        
        plotted_edges = set()
        for (i, j), count in overlaps.items():
            if count > overlap_threshold:
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
    
    if hit_points and gsom_map:
        for index, i in output.iterrows():
            x = i['x']
            y = i['y']
            if i['output'] in hit_points:
                ax.scatter(x, y, c=[listed_color_map.colors[i['hit_count']]], s=100, 
                           marker='o', label=f'Hit Point (Count: {i["hit_count"]})' if index == 0 else "")
                if show_index and i['hit_count'] > 0:
                    label = ", ".join(map(str, i[index_col][0:max_text]))
                    ax.text(x + 0.1, y + 0.1, label, ha='left', va='center', fontsize=8)
    
    if junctions and gsom_map:
        for j in junctions:
            x, y = gsom_map.node_coordinate[j]
            if not (np.isnan(x) or np.isnan(y)):
                ax.scatter(x, y, c='green', s=150, marker='s', label='Junction' if j == junctions[0] else "")
    
    if individual_mapping and gsom_map:
        for node_idx, data_indices in individual_mapping.items():
            x, y = gsom_map.node_coordinate[node_idx]
            if not (np.isnan(x) or np.isnan(y)):
                ax.scatter(x, y, c=f'C{len(data_indices) % 10}', s=50 + len(data_indices) * 10,
                           label=f'Node {node_idx} (Individuals: {len(data_indices)})' if node_idx == list(individual_mapping.keys())[0] else "",
                           alpha=0.7)
    
    ax.set_title(figure_label)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.savefig(file_name + file_type, bbox_inches='tight')
    plt.show()

def _get_color_map(max_count, alpha=1.0, cmap_colors="Paired"):
    cmap = cm.get_cmap(cmap_colors)
    colors = [cmap(i / max(max_count, 1)) for i in range(max_count + 1)]
    for i in range(len(colors)):
        colors[i] = list(colors[i])
        colors[i][3] = alpha
    return colors

if __name__ == '__main__':
    np.random.seed(1)
    try:
        # Check if Graphviz is installed
        if not shutil.which('dot'):
            logging.warning("Graphviz 'dot' not found in PATH. Skipping path_tree.png export.")

        # Load the CSV file and sample 200 rows
        data_filename = "example/data/zoo.txt".replace('\\', '/')  # Replace with your dataset path
        df = pd.read_csv(data_filename)
        if len(df) < 200:
            print(f"Warning: DataFrame has only {len(df)} rows, which is less than 200.")
            df_sampled = df
        else:
            df_sampled = df.sample(n=200, random_state=42)
        df = df_sampled

        # Separate features and target, keeping a copy for merging
        df_processed = df.copy()
        features = df_processed.drop(columns=['NObeyesdad'])  # Adjust target column as needed
        labels = df_processed['NObeyesdad']

        # Define categorical and numerical columns (adjust based on your dataset)
        categorical_binary = ['Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
        categorical_ordinal = ['CAEC', 'CALC']
        categorical_nominal = ['MTRANS']
        numerical = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

        # 1. Encode binary categorical variables
        label_encoders = {}
        for col in categorical_binary:
            label_encoders[col] = LabelEncoder()
            features[col] = label_encoders[col].fit_transform(features[col])

        # 2. Encode ordinal categorical variables
        caec_order = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
        calc_order = {'no': 0, 'Sometimes': 1, 'Frequently': 2}
        features['CAEC'] = features['CAEC'].map(caec_order)
        features['CALC'] = features['CALC'].map(calc_order)

        # 3. One-hot encode nominal categorical variables
        features = pd.get_dummies(features, columns=categorical_nominal, prefix='MTRANS')

        # Normalize numerical features
        scaler = MinMaxScaler()
        features[numerical] = scaler.fit_transform(features[numerical])

        # Convert all columns to float
        features = features.astype(float)

        # Add Index column
        features['Index'] = features.index

        # Merge preprocessed features back with NObeyesdad
        df_processed = features.copy()
        df_processed['NObeyesdad'] = labels

        # Convert to numpy array for training
        data_training = df_processed.drop(columns=['Index', 'NObeyesdad']).to_numpy()
        print(f"Training data shape: {data_training.shape}")
        print(f"Training data dtype: {data_training.dtype}")

        # Train GSOM
        gsom = GSOM(spred_factor=0.25, dimensions=data_training.shape[1], max_radius=4, initial_node_size=1000)
        gsom.fit(data_training, training_iterations=100, smooth_iterations=50)
        logging.info(f"Total nodes created: {gsom.node_count}")
        
        # Predict clusters
        output = gsom.predict(df_processed, "Index", label_col="NObeyesdad")
        output.to_csv("obesity_output.csv", index=False)
        
        # Build skeleton with individual mapping
        hit_points, skeleton_connections, junctions, pos_edges, individual_mapping = gsom.build_skeleton(data_training)
        
        # Print paths of spread
        logging.info("\nPaths of Spread:")
        paths = gsom.get_paths()
        for path in paths:
            node_names = [node.name for node in path]
            node_coords = [f"({node.x:.2f}, {node.y:.2f})" for node in path if hasattr(node, 'x') and hasattr(node, 'y')]
            logging.info(f"Path: {node_names} -> Coordinates: {node_coords}")
        
        # Visualize GSOM map with paths and individual mapping
        plot(output, "Index", gsom_map=gsom, hit_points=hit_points, skeleton_connections=skeleton_connections,
             junctions=junctions, pos_edges=pos_edges, individual_mapping=individual_mapping,
             file_name="obesity_gsom_with_paths", file_type=".png", figure_label="GSOM Map with Node Paths")
        
        # Visualize skeleton with thresholded edges and individual mapping
        plot(output, "NObeyesdad", gsom_map=gsom, hit_points=hit_points, skeleton_connections=skeleton_connections,
             junctions=junctions, pos_edges=pos_edges, individual_mapping=individual_mapping,
             file_name="obesity_gsom_skeleton", file_type=".png", figure_label="GSOM Skeleton", overlap_threshold=6)
        
        # Export path_tree to PNG if Graphviz is available
        if shutil.which('dot'):
            try:
                graph = tree_to_dot(gsom.path_tree)
                graph.write_png("obesity_path_tree.png")
                logging.info("Exported path_tree to obesity_path_tree.png")
            except Exception as e:
                logging.error(f"Failed to export path_tree: {str(e)}")
        else:
            logging.info("Path tree export skipped due to missing Graphviz.")
        
        print("Complete")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        print(f"Error: {str(e)}")