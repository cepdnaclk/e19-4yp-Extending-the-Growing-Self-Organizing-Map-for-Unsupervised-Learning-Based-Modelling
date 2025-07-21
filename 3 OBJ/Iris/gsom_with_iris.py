import numpy as np
import pandas as pd
from scipy.spatial import distance
import scipy
from tqdm import tqdm
import math
from bigtree import Node, find
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors

# Update data filename for Iris dataset
data_filename = "example/data/iris.csv".replace('\\', '/')



class GSOM:
    def __init__(self, spred_factor, dimensions, distance='euclidean', initialize='random', learning_rate=0.3,
                 smooth_learning_factor=0.8, max_radius=6, FD=0.1, r=3.8, alpha=0.9, initial_node_size=50000):
        """
        GSOM structure: keeps dictionary of x,y coordinates and numpy arrays for weights.
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
        self.path_tree = None
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
                )
                new_node.distance = distance_from_parent[0][0]
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
        return self.max_radius * math.exp(-iteration / time_constant)

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
        if (leftx, lefty) in self.map and (rightx, righty) in self.map and (topx, topy) in self.map and (bottomx, bottomy) in self.map:
            self.spread_wights(x, y)
        else:
            self.grow_node(x, y, leftx, lefty, 0)
            self.grow_node(x, y, rightx, righty, 1)
            self.grow_node(x, y, topx, topy, 2)
            self.grow_node(x, y, bottomx, bottomy, 3)
        self.node_errors[rmu_index] = self.groth_threshold / 2

    def winner_identification_and_neighbourhood_update(self, data_index, data, radius, learning_rate):
        out = scipy.spatial.distance.cdist(
            self.node_list[:self.node_count],
            data[data_index, :].reshape(1, self.dimentions),
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
        """
        Manually get all paths from the root to leaf nodes in the path_tree.
        Returns a list of paths, where each path is a list of Node objects.
        """
        paths = []
        def collect_paths(node, current_path):
            current_path.append(node)
            if not node.children:  # Leaf node
                paths.append(current_path[:])
            for child in node.children:
                collect_paths(child, current_path)
            current_path.pop()
        collect_paths(self.path_tree, [])
        return paths

def plot(output, index_col, gsom_map=None, file_name="gsom", file_type=".pdf", 
         figure_label="GSOM Map with Paths", max_text=3, max_length=30, 
         cmap_colors="Paired", show_index=True, n_nodes=180):
    """
    Plot GSOM nodes with their clustered data points and paths between nodes.
    Modified to fix legend overlap by moving it outside the plot and simplifying entries.
    """
    max_count = output["hit_count"].max()
    listed_color_map = _get_color_map(max_count, alpha=0.9, cmap_colors=cmap_colors)
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot all nodes
    if gsom_map:
        ax.plot(gsom_map.node_coordinate[:gsom_map.node_count, 0],
                gsom_map.node_coordinate[:gsom_map.node_count, 1],
                'o', color='gray', markersize=2, label='All Nodes')

    # Plot paths between nodes
    if gsom_map:
        paths = gsom_map.get_paths()
        for path in paths:
            if len(path) > 1:
                x_coords = [node.x for node in path]
                y_coords = [node.y for node in path]
                ax.plot(x_coords, y_coords, 'k-', linewidth=0.5, alpha=0.3,
                        label='Node Connections' if path == paths[0] else "")

    # Plot output nodes with hits and collect unique hit counts for the legend
    hit_count_colors = {}
    for index, i in output.iterrows():
        x = i['x']
        y = i['y']
        hit_count = i['hit_count']
        color = listed_color_map.colors[hit_count]
        ax.plot(x, y, 'o', color=color, markersize=6)
        if hit_count > 0 and hit_count not in hit_count_colors:
            hit_count_colors[hit_count] = color
        if show_index and hit_count > 0:
            label = ", ".join(map(str, i[index_col][0:max_text]))
            txt = ax.text(x, y, label, ha='left', va='center', wrap=True, fontsize=8)
            txt._get_wrap_line_width = lambda: max_length

    # Create legend entries for hit counts
    for hit_count, color in sorted(hit_count_colors.items()):
        ax.plot([], [], 'o', color=color, markersize=6, label=f'Hit Count {hit_count}')

    ax.set_title(figure_label)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(file_name + file_type, bbox_inches='tight')
    plt.show()

def _get_color_map(max_count, alpha=0.9, cmap_colors="Paired"):
    np.random.seed(1)
    cmap = cm.get_cmap(cmap_colors, max_count + 1)
    color_list = [(c[0] * alpha, c[1] * alpha, c[2] * alpha) for c in cmap(np.arange(cmap.N))]
    return colors.ListedColormap(color_list, name='gsom_color_list')

if __name__ == '__main__':
    np.random.seed(1)
    try:
        df = pd.read_csv(data_filename)
        print("Dataset shape:", df.shape)
        data_training = df.iloc[:, 1:5]
        print("Training data shape:", data_training.shape)
        print("Training data sample:\n", data_training.head())

        gsom = GSOM(0.83, 4, max_radius=4, initial_node_size=50000)
        gsom.fit(data_training.to_numpy(), 100, 50)
        print(gsom.node_count)
        output = gsom.predict(df, "Id", "Species")
        output.to_csv("output.csv", index=False)
        plot(output, "Id", gsom_map=gsom, file_name="gsom_with_paths_iris",
             file_type=".pdf", figure_label="GSOM Map with Node Paths", n_nodes=gsom.node_count)
        print("Complete")

    except FileNotFoundError:
        print(f"Error: Data file issue. Ensure ucimlrepo is installed and accessible.")
    except Exception as e:
        print(f"Error: {str(e)}")