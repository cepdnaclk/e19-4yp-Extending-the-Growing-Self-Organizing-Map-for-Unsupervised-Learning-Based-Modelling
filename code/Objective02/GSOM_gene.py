import numpy as np
import pandas as pd
from scipy.spatial import distance
import scipy
from tqdm import tqdm
import math
from bigtree import Node, findall, find
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from collections import Counter
import ast



data_filename = "example/data/iris.csv".replace('\\', '/')


class GSOM:

    def __init__(self, spred_factor, dimensions, distance='euclidean', initialize='random', learning_rate=0.3,
                 smooth_learning_factor=0.8,
                 max_radius=6, FD=0.1, r=3.8, alpha=0.9, initial_node_size=30000):
        """
        GSOM structure:
        keep dictionary to x,y coordinates and numpy array to keep weights
        :param spred_factor: spread factor of GSOM graph
        :param dimensions: weight vector dimensions
        :param distance: distance method: support scipy.spatial.distance.cdist
        :param initialize: weight vector initialize method
        :param learning_rate: initial training learning rate of weights
        :param smooth_learning_factor: smooth learning factor to change the initial smooth learning rate from training
        :param max_radius: maximum neighbourhood radius
        :param FD: spread weight value #TODO: check this from paper
        :param r: learning rate update value #TODO: check this from paper
        :param alpha: learning rate update value #TODO: check this from paper
        :param initial_node_size: initial node allocation in memory
        """
        self.initial_node_size = initial_node_size
        self.node_count = 0  # Keep current GSOM node count
        self.map = {}
        self.node_list = np.zeros((self.initial_node_size, dimensions))  # initialize node allocation in memory
        self.node_coordinate = np.zeros((self.initial_node_size, 2))  # initialize node coordinate in memory
        self.node_errors = np.zeros(self.initial_node_size, dtype=np.longdouble)  # initialize node error in memory
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
        self.node_labels = None  # Keep the prediction GSOM nodes
        self.output = None # keep the cluster id of each data point
        # HTM sequence learning parameters
        self.predictive = None  # Keep the prediction of the next sequence value (HTM predictive state)
        self.active = None  # Keep the activation of the current sequence value (HTM active state)
        self.sequence_weights = None  # Sequence weight matrix. This has the dimensions node count*column height
        self.path_tree = {}  # To store the root nodes of path trees using bigtree
        self.initialize_GSOM()


    def initialize_GSOM(self):
        self.path_tree = Node("root", x=0.01, y=0.01, node_number=-1, distance=0) #initialize the root node
        
        for x, y in [(1, 1), (1, 0), (0, 1), (0, 0)]:
            self.insert_node_with_weights(x, y)
        

    def insert_new_node(self, x, y, weights, parent_node=None):
        if self.node_count > self.initial_node_size:
            print("node size out of bound")
            # TODO:resize the nodes
        self.map[(x, y)] = self.node_count
        self.node_list[self.node_count] = weights
        self.node_coordinate[self.node_count][0] = x
        self.node_coordinate[self.node_count][1] = y
        
        distance_from_parent=0
        new_node = Node(str(self.node_count), x=x, y=y, node_number=self.node_count, distance=distance_from_parent)

        if parent_node is not None:
            if (parent_node.x, parent_node.y) in self.map:
                distance_from_parent = scipy.spatial.distance.cdist(weights.reshape(1, -1), self.node_list[self.map[(parent_node.x, parent_node.y)]].reshape(1, -1), self.distance)
                new_node.distance = distance_from_parent[0][0]
                
            new_node.parent = parent_node
            #print(f'parent node: {parent_node} child node: {new_node} diance: {distance_from_parent}')
        else:
            raise ValueError("Parent node is not provided")

        self.node_count += 1

    def insert_node_with_weights(self, x, y):
        if self.initialize == 'random':
            node_weights = np.random.rand(self.dimentions)
        else:
            raise NotImplementedError("Initialization method not supported")
            # TODO:: add other initialize methods
        
        #insert a new node into root level
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
        """
        grow new node if not exist on x,y coordinates using the winner node weight(wx,wy)
        check the side of the winner new node add in following order (left, right, top and bottom)
        new node N
        winner node W
        Other nodes O
        left
        =============
        1 O-N-W
        -------------
        2 N-W-O
        -------------
        3   O
            |
          N-W
        -------------
        4 N-W
            |
            O
        -------------
        =============
        right
        =============
        1 W-N-O
        -------------
        2 o-W-N
        -------------
        3 O
          |
          W-N
        -------------
        4 W-N
          |
          O
        -------------
        =============
        top
        ===============
        1 O
          |
          N
          |
          W
        -------------
        1 N
          |
          W
          |
          O
        -------------
        3 N
          |
          W-N
        -------------
        4 N
          |
        O-N
        -------------
        =============
        :param wx:
        :param wy:
        :param x:
        :param y:
        :param side:
        """
        if not (x, y) in self.map:
            #print(f'adding new node to ({wx},{wy}) at ({x},{y}) side:', side)
            if side == 0:  # add new node to left of winner
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
            elif side == 1:  # add new node to right of winner
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
            elif side == 2:  # add new node to top of winner
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
            elif side == 3:  # add new node to bottom of winner
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
                
            # clip the wight between (0,1)
            weights[weights < 0] = 0.0
            weights[weights > 1] = 1.0
            
            parent_node = find(self.path_tree, lambda node: node.x==wx and node.y==wy)
            self.insert_new_node(x, y, weights, parent_node=parent_node)

    def spread_wights(self, x, y):
        leftx, lefty = x - 1, y
        rightx, righty = x + 1, y
        topx, topy = x, y + 1
        bottomx, bottomy = x, y - 1
        self.node_errors[self.map[(x, y)]] = self.groth_threshold/2   #TODO check this value if different in Rashmika's
                                                                    # version and paper version (paper t/2 Rashmika t)
        self.node_errors[self.map[(leftx, lefty)]] *= (1 + self.FD)
        self.node_errors[self.map[(rightx, righty)]] *= (1 + self.FD)
        self.node_errors[self.map[(topx, topy)]] *= (1 + self.FD)
        self.node_errors[self.map[(bottomx, bottomy)]] *= (1 + self.FD)

    def adjust_wights(self, x, y, rmu_index):
        leftx, lefty = x - 1, y
        rightx, righty = x + 1, y
        topx, topy = x, y + 1
        bottomx, bottomy = x, y - 1
        # Check all neighbours exist and spread the weights
        if (leftx, lefty) in self.map \
                and (rightx, righty) in self.map \
                and (topx, topy) in self.map \
                and (bottomx, bottomy) in self.map:
            self.spread_wights(x, y)
        else:
        # Grow new nodes for the four sides
            self.grow_node(x, y, leftx, lefty, 0)
            self.grow_node(x, y, rightx, righty, 1)
            self.grow_node(x, y, topx, topy, 2)
            self.grow_node(x, y, bottomx, bottomy, 3)
        self.node_errors[rmu_index] = self.groth_threshold/2 #TODO check the need of setting the error to zero after weight adaptation

    def winner_identification_and_neighbourhood_update(self, data_index, data, radius, learning_rate):
        out = scipy.spatial.distance.cdist(self.node_list[:self.node_count], data[data_index, :].reshape(1, self.dimentions), self.distance)
        rmu_index = out.argmin()  # get winner node index
        error_val = out.min()
        # get winner node coordinates
        rmu_x = int(self.node_coordinate[rmu_index][0])
        rmu_y = int(self.node_coordinate[rmu_index][1])

        # Update winner error
        error = data[data_index] - self.node_list[rmu_index]
        self.node_list[self.map[(rmu_x, rmu_y)]] = self.node_list[self.map[(rmu_x, rmu_y)]] + learning_rate * error

        # Get integer radius value
        mask_size = round(radius)

        # Iterate over the winner node radius(neighbourhood)
        for i in range(rmu_x - mask_size, rmu_x + mask_size):
            for j in range(rmu_y - mask_size, rmu_y + mask_size):
                # Check neighbour coordinate in the map not winner coordinates
                if (i, j) in self.map and (i != rmu_x and j != rmu_y):
                    # get error between winner and neighbour
                    error = self.node_list[rmu_index] - self.node_list[self.map[(i, j)]]
                    distance = (rmu_x - i) * (rmu_x - i) + (rmu_y - j) * (rmu_y - j)
                    eDistance = np.exp(-1.0 * distance / (2.0 * (radius * radius)))  # influence from distance

                    # Update neighbour error
                    self.node_list[self.map[(i, j)]] = self.node_list[self.map[(i, j)]] \
                                                       + learning_rate * eDistance * error
        return rmu_index, rmu_x, rmu_y, error_val

    def smooth(self, data, radius, learning_rate):
        # Iterate all data points
        for data_index in range(data.shape[0]):
            self.winner_identification_and_neighbourhood_update(data_index, data, radius, learning_rate)

    def grow(self, data, radius, learning_rate):
        # Iterate all data points
        for data_index in range(data.shape[0]):
            rmu_index, rmu_x, rmu_y, error_val = self.winner_identification_and_neighbourhood_update(data_index, data, radius, learning_rate)

            # winner node weight update and grow
            self.node_errors[rmu_index] += error_val
            if self.node_errors[rmu_index] > self.groth_threshold:
                self.adjust_wights(rmu_x, rmu_y, rmu_index)

    def fit(self, data, training_iterations, smooth_iterations):
        """
        method to train the GSOM map
        :param data:
        :param training_iterations:
        :param smooth_iterations:
        """
        current_learning_rate = self.learning_rate
        # growing iterations
        for i in tqdm(range(training_iterations)):
            radius_exp = self._get_neighbourhood_radius(training_iterations, i)
            if i != 0:
                current_learning_rate = self._get_learning_rate(current_learning_rate)

            self.grow(data, radius_exp, current_learning_rate)

        # smoothing iterations
        current_learning_rate = self.learning_rate * self.smooth_learning_factor
        for i in tqdm(range(smooth_iterations)):
            radius_exp = self._get_neighbourhood_radius(training_iterations, i)
            if i != 0:
                current_learning_rate = self._get_learning_rate(current_learning_rate)

            self.smooth(data, radius_exp, current_learning_rate)
        # Identify winners
        out = scipy.spatial.distance.cdist(self.node_list[:self.node_count], data, self.distance)
        return out.argmin(axis=0)

    def predict(self, data, index_col, label_col=None):
        """
        Identify the winner nodes for test dataset
        Predict the winning node for each data point and create a pandas dataframe
        need to provide both index column and label column
        :param data:
        :param index_col:
        :param label_col:
        :return:
        """

        # Prepare the dataset - remove label and index column
        weight_columns = list(data.columns.values)
        output_columns = [index_col]
        if label_col:
            weight_columns.remove(label_col)
            output_columns.append(label_col)

        weight_columns.remove(index_col)
        data_n = data[weight_columns].to_numpy()
        data_out = pd.DataFrame(data[output_columns])
        # Identify winners
        out = scipy.spatial.distance.cdist(self.node_list[:self.node_count], data_n, self.distance)
        data_out["output"] = out.argmin(axis=0)

        grp_output =data_out.groupby("output")
        dn = grp_output[index_col].apply(list).reset_index()
        dn = dn.set_index("output")
        if label_col:
            dn[label_col] = grp_output[label_col].apply(list)
        dn = dn.reset_index()
        dn["hit_count"] = dn[index_col].apply(lambda x: len(x))
        dn["x"] = dn["output"].apply(lambda x: self.node_coordinate[x, 0])
        dn["y"] = dn["output"].apply(lambda x: self.node_coordinate[x, 1])
        hit_max_count = dn["hit_count"].max()
        self.node_labels = dn
        # display map
        #plot(self.node_labels, index_col)
        self.output = data_out

        return self.node_labels
        
        
    def get_paths(self):
        paths = []
        paths.extend(self.path_tree.get_paths())
        return paths


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
    from scipy.spatial.distance import pdist
    from sklearn.metrics import silhouette_score
    import pandas as pd
    import numpy as np
    from collections import Counter
    import ast
    import os

    np.random.seed(1)

    # Step 1: Load the normalized gene expression dataset
    df = pd.read_csv("example/data/GSE/GSE5281_normalized_gene_expression.csv", index_col=0)
    print("Dataset shape:", df.shape)

    # Transpose: samples as rows, genes as features
    df = df.T
    df.index.name = "Sample_ID"

    # Step 2: Assign temporary labels
    df["Id"] = df.index
    df["Species"] = ["Alzheimer's Disease" if i < 87 else "Control" for i in range(161)]

    # Step 3: Prepare inputs
    features = df.drop(columns=["Id", "Species"])
    full_input = pd.concat([features, df[["Id", "Species"]]], axis=1)

    # Step 4: Train GSOM
    gsom = GSOM(spred_factor=0.83, dimensions=features.shape[1], distance='euclidean', max_radius=4)
    gsom.fit(features.to_numpy(), training_iterations=100, smooth_iterations=50)

    # Step 5: Predict & save
    output = gsom.predict(full_input, index_col="Id", label_col="Species")
    output.to_csv("output_gse5281.csv", index=False)

    # Step 6: Active GSOM nodes
    df_out = pd.read_csv("output_gse5281.csv")
    active_nodes = df_out[df_out["hit_count"] > 0].copy()
    active_nodes["Species"] = active_nodes["Species"].apply(ast.literal_eval)

    # ✅ Optional: Filter sparse GSOM nodes
    active_nodes = active_nodes[active_nodes["hit_count"] >= 2].copy()

    # Step 7: Hierarchical clustering - evaluate best linkage
    X = active_nodes[["x", "y"]].to_numpy()
    print("📊 Cophenetic Correlation Coefficients for Linkage Methods:")
    best_coph = 0
    best_method = None
    best_Z = None

    for method in ['single', 'complete', 'average', 'ward']:
        Z_temp = linkage(X, method=method)
        coph_corr, _ = cophenet(Z_temp, pdist(X))
        print(f"{method.capitalize():<10} CCC: {coph_corr:.4f}")
        if coph_corr > best_coph:
            best_coph = coph_corr
            best_method = method
            best_Z = Z_temp

    Z = best_Z
    print(f"\n✅ Best linkage method: {best_method} with CCC = {best_coph:.4f}")

    # Step 8: Assign clusters
    active_nodes["cluster"] = fcluster(Z, 2, criterion='maxclust')

    # Step 9: Silhouette Scores (Nodes & Samples)
    os.makedirs("results", exist_ok=True)

    # 9.1 Silhouette Score for GSOM nodes
    X_nodes = active_nodes[["x", "y"]].to_numpy()
    labels_nodes = active_nodes["cluster"].to_numpy()

    sil_node_score = None
    if len(set(labels_nodes)) > 1:
        sil_node_score = silhouette_score(X_nodes, labels_nodes)
        print(f"📐 Silhouette Score (GSOM Nodes): {sil_node_score:.4f}")
    else:
        print("⚠️ Not enough node clusters to compute Silhouette Score.")

    # 9.2 Silhouette Score for Samples (mapped from GSOM node clusters)
    data_points = pd.read_csv("output_gse5281.csv")
    node_clusters = pd.read_csv("gsom_node_clusters_gse5281.csv") if os.path.exists("gsom_node_clusters_gse5281.csv") else active_nodes
    merged = pd.merge(data_points, node_clusters[["output", "cluster"]], on="output", how="left")

    # Drop samples with NaN cluster assignment
    merged_valid = merged.dropna(subset=["cluster"]).copy()
    X_samples = merged_valid[["x", "y"]].to_numpy()
    labels_samples = merged_valid["cluster"].astype(int).to_numpy()


    sil_sample_score = None
    if len(set(labels_samples)) > 1:
        sil_sample_score = silhouette_score(X_samples, labels_samples)
        print(f"📐 Silhouette Score (Samples): {sil_sample_score:.4f}")
    else:
        print("⚠️ Not enough sample clusters to compute Silhouette Score.")

    # Step 10: Save metrics
    with open("results/silhouette_scores.txt", "w") as f:
        f.write("Silhouette Score Results\n")
        f.write("========================\n")
        f.write(f"Best Linkage Method: {best_method}\n")
        f.write(f"Cophenetic Correlation Coefficient (CCC): {best_coph:.4f}\n\n")
        if sil_node_score is not None:
            f.write(f"Silhouette Score (GSOM Nodes): {sil_node_score:.4f}\n")
        else:
            f.write("Silhouette Score (GSOM Nodes): Not enough clusters.\n")
        if sil_sample_score is not None:
            f.write(f"Silhouette Score (Samples): {sil_sample_score:.4f}\n")
        else:
            f.write("Silhouette Score (Samples): Not enough clusters.\n")

    print("✅ Silhouette scores saved to 'results/silhouette_scores.txt'")

    # Step 11: Label summary
    def formatted_label(row):
        counter = Counter(row["Species"])
        ad = counter.get("Alzheimer's Disease", 0)
        ctrl = counter.get("Control", 0)
        total = ad + ctrl
        dominant = "AD" if ad > ctrl else "Control"
        percent = (max(ad, ctrl) / total) * 100 if total > 0 else 0
        return f"Cluster {row['cluster']} | {dominant} ({percent:.1f}%) | N={total}"

    active_nodes["label_summary"] = active_nodes.apply(formatted_label, axis=1)
    active_nodes.to_csv("gsom_node_clusters_gse5281.csv", index=False)

    # Step 12: Plot dendrogram
    plt.figure(figsize=(18, 8))
    dendrogram(
        Z,
        labels=active_nodes["label_summary"].values,
        leaf_rotation=90,
        leaf_font_size=10
    )
    plt.title(f"Hierarchical Clustering on GSOM Nodes (Linkage: {best_method})")
    plt.xlabel("Clustered GSOM Nodes (AD vs Control)")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig("hierarchical_clustering_gse5281_annotated.png")
    plt.close()

    # Step 13: Map each sample to its node's cluster
    merged.to_csv("gse5281_sample_cluster_mapping.csv", index=False)

    print("✅ Dendrogram saved as 'hierarchical_clustering_gse5281_annotated.png'")
    print("✅ Node clusters saved as 'gsom_node_clusters_gse5281.csv'")
    print("✅ Sample-cluster mapping saved as 'gse5281_sample_cluster_mapping.csv'")
