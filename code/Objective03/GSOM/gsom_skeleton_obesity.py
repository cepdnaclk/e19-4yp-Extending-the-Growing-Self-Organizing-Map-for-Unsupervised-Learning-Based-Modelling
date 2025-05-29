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

    def insert_node_with_weights(self, x, y):
        if self.initialize == 'random':
            node_weights = np.random.rand(self.dimentions)
        else:
            raise NotImplementedError("Initialization method not supported")
        self.insert_new_node(x, y, node_weights, parent_node=self.path_tree)

    def insert_new_node(self, x, y, weights, parent_node=None):
        if self.node_count >= self.initial_node_size:
            new_size = self.initial_node_size * 2
            print(f"Expanding memory: {self.initial_node_size} → {new_size}")
            self.node_list = np.vstack([self.node_list, np.zeros((self.initial_node_size, self.dimentions))])
            self.node_coordinate = np.vstack([self.node_coordinate, np.zeros((self.initial_node_size, 2))])
            self.node_errors = np.concatenate([self.node_errors, np.zeros(self.initial_node_size, dtype=np.longdouble)])
            self.initial_node_size = new_size

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

    def fit(self, data, training_iterations, smooth_iterations):
        current_learning_rate = self.learning_rate
        for i in tqdm(range(training_iterations), desc="Growing"):
            radius_exp = self._get_neighbourhood_radius(training_iterations, i)
            if i != 0:
                current_learning_rate = self._get_learning_rate(current_learning_rate)
            self._grow(data, radius_exp, current_learning_rate)

        current_learning_rate = self.learning_rate * self.smooth_learning_factor
        for i in tqdm(range(smooth_iterations), desc="Smoothing"):
            radius_exp = self._get_neighbourhood_radius(smooth_iterations, i)
            if i != 0:
                current_learning_rate = self._get_learning_rate(current_learning_rate)
            self._smooth(data, radius_exp, current_learning_rate)

    def _grow(self, data, radius, learning_rate):
        for data_index in range(data.shape[0]):
            distances = scipy.spatial.distance.cdist(self.node_list[:self.node_count], data[data_index:data_index+1], self.distance)
            winner = np.argmin(distances)
            self.node_list[winner] += learning_rate * (data[data_index] - self.node_list[winner])

    def _smooth(self, data, radius, learning_rate):
        for data_index in range(data.shape[0]):
            distances = scipy.spatial.distance.cdist(self.node_list[:self.node_count], data[data_index:data_index+1], self.distance)
            winner = np.argmin(distances)
            self.node_list[winner] += learning_rate * (data[data_index] - self.node_list[winner])

    def _get_learning_rate(self, prev_learning_rate):
        return self.ALPHA * (1 - (self.R / self.node_count)) * prev_learning_rate

    def _get_neighbourhood_radius(self, total_iteration, iteration):
        time_constant = total_iteration / math.log(self.max_radius)
        return self.max_radius * math.exp(- iteration / time_constant)

    def predict(self, data, index_col, label_col=None, feature_columns=None):
        weight_columns = list(data.columns.values)
        output_columns = [index_col]

        if label_col:
            if label_col in weight_columns:
                weight_columns.remove(label_col)
                output_columns.append(label_col)
            else:
                print(f"Warning: label column '{label_col}' not found in DataFrame. Skipping it.")

        if index_col in weight_columns:
            weight_columns.remove(index_col)
        else:
            print(f"Warning: index column '{index_col}' not found in weight columns. Proceeding anyway.")

        if feature_columns is not None:
            data_n = data[feature_columns].to_numpy(dtype=np.float64)
        else:
            data_n = data.select_dtypes(include=['number']).to_numpy(dtype=np.float64)

        data_out = pd.DataFrame(data[output_columns])
        out = scipy.spatial.distance.cdist(self.node_list[:self.node_count], data_n, self.distance)
        data_out["output"] = out.argmin(axis=0)

        grp_output = data_out.groupby("output")
        dn = grp_output[index_col].apply(list).reset_index()
        dn = dn.set_index("output")

        if label_col and label_col in data.columns:
            dn[label_col] = grp_output[label_col].apply(list)

        dn = dn.reset_index()
        dn["hit_count"] = dn[index_col].apply(lambda x: len(x))
        dn["x"] = dn["output"].apply(lambda x: self.node_coordinate[x, 0])
        dn["y"] = dn["output"].apply(lambda x: self.node_coordinate[x, 1])
        self.node_labels = dn
        self.output = data_out
        return self.node_labels

if __name__ == '__main__':
    np.random.seed(1)
    df = pd.read_csv("../Data/ObesityDataSet_raw_and_data_sinthetic.csv")
    print("Original Dataset shape:", df.shape)

    numeric_df = df.select_dtypes(include=['number'])
    data_np = numeric_df.to_numpy(dtype=np.float64)
    training_columns = numeric_df.columns

    gsom = GSOM(spred_factor=0.83, dimensions=data_np.shape[1], max_radius=4, initial_node_size=1000)
    gsom.fit(data_np, training_iterations=100, smooth_iterations=50)

    output = gsom.predict(df, index_col="Age", label_col="NObeyesdad", feature_columns=training_columns)
    output.to_csv("output.csv", index=False)

    print("GSOM training completed.")
    print("Output shape:", output.shape)
    print("Node Count:", gsom.node_count)


