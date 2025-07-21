import numpy as np
import pandas as pd
from scipy.spatial import distance
import scipy
from tqdm import tqdm
import math
from bigtree import Node, findall, find
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

class GSOM:
    def __init__(self, spread_factor, dimensions, distance='euclidean', initialize='random', learning_rate=0.3,
                    smooth_learning_factor=0.8, max_radius=6, FD=0.1, r=3.8, alpha=0.9, initial_node_size=30000):
        self.initial_node_size = initial_node_size
        self.node_count = 0
        self.map = {}
        self.node_list = np.zeros((self.initial_node_size, dimensions))
        self.node_coordinate = np.zeros((self.initial_node_size, 2))
        self.node_errors = np.zeros(self.initial_node_size, dtype=np.longdouble)
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
            raise ValueError("Node size out of bound")
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
        weights = np.full(self.dimensions, (max(self.node_list[self.map[(winnerx, winnery)]]) + min(
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

    def spread_weights(self, x, y):
        neighbors = [(x - 1, y), (x + 1, y), (x, y + 1), (x, y - 1)]
        self.node_errors[self.map[(x, y)]] = self.growth_threshold / 2
        for nx, ny in neighbors:
            if (nx, ny) in self.map:
                self.node_errors[self.map[(nx, ny)]] *= (1 + self.FD)

    def adjust_weights(self, x, y, rmu_index):
        neighbors = [(x - 1, y), (x + 1, y), (x, y + 1), (x, y - 1)]
        if all((nx, ny) in self.map for nx, ny in neighbors):
            self.spread_weights(x, y)
        else:
            self.grow_node(x, y, x - 1, y, 0)
            self.grow_node(x, y, x + 1, y, 1)
            self.grow_node(x, y, x, y + 1, 2)
            self.grow_node(x, y, x, y - 1, 3)
        self.node_errors[rmu_index] = self.growth_threshold / 2

    def winner_identification_and_neighbourhood_update(self, data_index, data, radius, learning_rate):
        out = scipy.spatial.distance.cdist(self.node_list[:self.node_count], data[data_index, :].reshape(1, self.dimensions), self.distance)
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
                    eDistance = np.exp(-distance / (2.0 * (radius ** 2)))
                    self.node_list[self.map[(i, j)]] += learning_rate * eDistance * error
        return rmu_index, rmu_x, rmu_y, error_val

    def smooth(self, data, radius, learning_rate):
        for data_index in range(data.shape[0]):
            self.winner_identification_and_neighbourhood_update(data_index, data, radius, learning_rate)

    def grow(self, data, radius, learning_rate):
        for data_index in range(data.shape[0]):
            rmu_index, rmu_x, rmu_y, error_val = self.winner_identification_and_neighbourhood_update(data_index, data, radius, learning_rate)
            self.node_errors[rmu_index] += error_val
            if self.node_errors[rmu_index] > self.growth_threshold:
                self.adjust_weights(rmu_x, rmu_y, rmu_index)

    def fit(self, data, training_iterations, smooth_iterations):
        current_learning_rate = self.learning_rate
        for i in tqdm(range(training_iterations)):
            radius_exp = self._get_neighbourhood_radius(training_iterations, i)
            if i != 0:
                current_learning_rate = self._get_learning_rate(current_learning_rate)
            self.grow(data, radius_exp, current_learning_rate)
        current_learning_rate = self.learning_rate * self.smooth_learning_factor
        for i in tqdm(range(smooth_iterations)):
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

    def build_data_skeleton(self, data, k_multiplier=1.0, index_col='Name', label_col='label', max_clusters=7, min_variance_reduction=0.1):
        """
        Build Data Skeleton Model (DSM) using a top-down cluster separation approach with intra-cluster distance variance.
        Identify Paths of Spread, Hit Points, Junctions, and Path Segments, then split based on variance.
        """
        # Step 1: Identify Hit Points (nodes with data points mapped)
        hit_points = self.node_labels[self.node_labels['hit_count'] > 0][['output', 'x', 'y', 'hit_count', index_col]].copy()
        hit_points['node_number'] = hit_points['output']
        
        # Merge true labels into node_labels
        if label_col in data.columns and index_col in data.columns:
            self.node_labels = self.node_labels.merge(
                data[[index_col, label_col]].drop_duplicates(),
                on=index_col,
                how='left'
            )
            hit_points = self.node_labels[self.node_labels['hit_count'] > 0][['output', 'x', 'y', 'hit_count', index_col, label_col]].copy()
            hit_points['node_number'] = hit_points['output']
        
        # Step 2: Get Paths of Spread (POS) from path_tree
        paths = []
        for node in findall(self.path_tree, lambda node: node.node_number != -1):
            path = []
            current = node
            while current.node_number != -1:
                path.append((current.node_number, current.x, current.y, current.distance))
                current = current.parent
            paths.append(path[::-1])  # Reverse to start from root

        # Step 3: Identify Junctions (nodes connecting external hit points to POS)
        junctions = []
        for idx, row in hit_points.iterrows():
            node_num = row['node_number']
            node = find(self.path_tree, lambda n: n.node_number == node_num)
            if node and node.children:
                for child in node.children:
                    if child.node_number not in hit_points['node_number'].values:
                        junctions.append((child.node_number, child.x, child.y))

        # Step 4: Compute Path Segments and their distances
        path_segments = []
        segment_df = pd.DataFrame(columns=['node1', 'node2', 'distance'], dtype=float)
        for path in paths:
            for i in range(len(path) - 1):
                node1, x1, y1, _ = path[i]
                node2, x2, y2, dist = path[i + 1]
                if node1 in hit_points['node_number'].values or node2 in hit_points['node_number'].values:
                    weight_dist = scipy.spatial.distance.cdist(
                        self.node_list[node1].reshape(1, -1),
                        self.node_list[node2].reshape(1, -1),
                        self.distance
                    )[0][0]
                    path_segments.append((node1, node2, weight_dist))
                    segment_df = pd.concat([segment_df, pd.DataFrame({'node1': [node1], 'node2': [node2], 'distance': [weight_dist]})], ignore_index=True)

        for junction in junctions:
            j_node, j_x, j_y = junction
            parent = find(self.path_tree, lambda n: n.node_number == j_node).parent
            if parent.node_number in hit_points['node_number'].values:
                weight_dist = scipy.spatial.distance.cdist(
                    self.node_list[j_node].reshape(1, -1),
                    self.node_list[parent.node_number].reshape(1, -1),
                    self.distance
                )[0][0]
                path_segments.append((j_node, parent.node_number, weight_dist))
                segment_df = pd.concat([segment_df, pd.DataFrame({'node1': [j_node], 'node2': [parent.node_number], 'distance': [weight_dist]})], ignore_index=True)

        # Step 5: Top-Down Cluster Separation with Variance
        def compute_variance(nodes, segments):
            if not nodes or len(nodes) < 2:
                return 0.0
            coords = np.array([
                [self.node_coordinate[int(n), 0], self.node_coordinate[int(n), 1]]
                for n in nodes if isinstance(n, (int, np.integer, float)) and not isinstance(n, bool)
            ])
            dist_matrix = scipy.spatial.distance.cdist(coords, coords, 'euclidean')
            distances = dist_matrix[np.triu_indices(len(nodes), k=1)]
            return np.var(distances) if len(distances) > 0 else 0.0

        def split_cluster(nodes, segments, current_clusters):
            if len(current_clusters) >= max_clusters or not segments:
                return current_clusters
            # Compute variance for the current cluster
            current_variance = compute_variance(nodes, segments)
            best_reduction = 0.0
            best_split = None
            best_cluster1, best_cluster2 = None, None
            
            # Try splitting at each segment
            for seg in segments:
                node1, node2 = seg['node1'], seg['node2']
                if node1 in nodes and node2 in nodes:
                    # Perform BFS to split
                    cluster1, cluster2 = set(), set()
                    visited = set()
                    queue = [node1]
                    while queue:
                        current = queue.pop(0)
                        if current not in visited:
                            visited.add(current)
                            cluster1.add(current)
                            for s in segments:
                                if s['node1'] == current and s['node2'] not in visited:
                                    queue.append(s['node2'])
                                elif s['node2'] == current and s['node1'] not in visited:
                                    queue.append(s['node1'])
                    cluster2 = set(nodes) - cluster1
                    
                    # Compute variance after split
                    var1 = compute_variance(list(cluster1), [s for s in segments if s['node1'] in cluster1 and s['node2'] in cluster1])
                    var2 = compute_variance(list(cluster2), [s for s in segments if s['node1'] in cluster2 and s['node2'] in cluster2])
                    total_variance = (len(cluster1) * var1 + len(cluster2) * var2) / len(nodes) if len(nodes) > 0 else 0.0
                    reduction = current_variance - total_variance
                    
                    if reduction > best_reduction:
                        best_reduction = reduction
                        best_split = seg
                        best_cluster1, best_cluster2 = cluster1, cluster2
            
            if best_reduction > min_variance_reduction:
                # Remove the best split segment
                segments = [s for s in segments if s != best_split]
                current_clusters.append(list(best_cluster1))
                if best_cluster2:
                    current_clusters.append(list(best_cluster2))
                # Recurse on new clusters
                for cluster in current_clusters[-2:]:
                    relevant_segments = [s for s in segments if s['node1'] in cluster and s['node2'] in cluster]
                    split_cluster(cluster, relevant_segments, current_clusters)
            return current_clusters

        # Initialize with all nodes
        all_nodes = set(self.node_labels['output'])
        clusters = []
        if not segment_df.empty:
            clusters = split_cluster(all_nodes, segment_df.to_dict('records'), clusters)

        # Assign cluster labels
        cluster_labels = {}
        for cluster_id, cluster_nodes in enumerate(clusters):
            for node in cluster_nodes:
                cluster_labels[node] = cluster_id
        self.node_labels['cluster'] = self.node_labels['output'].map(cluster_labels).fillna(-1).astype(int)
        if self.output is not None and "output" in self.output.columns:
            self.output['cluster'] = self.output['output'].map(cluster_labels).fillna(-1).astype(int)

        # Step 6: Model idionomic features
        if not self.output.empty and index_col in data.columns:
            data_n = data.drop(columns=[index_col]).copy()
            data_n[index_col] = data[index_col]
            data_n = data_n.merge(self.output[['output', 'cluster']], left_on=index_col, right_on=index_col, how='left')
            weight_columns = [col for col in data.columns if col != index_col]
            cluster_features = data_n.groupby('cluster')[weight_columns].mean().reset_index()
            cluster_features.to_csv("cluster_features.csv", index=False)
        
        return paths, hit_points, junctions, path_segments, {i: cluster for i, cluster in enumerate(clusters)}
    
    def visualize_clusters(self, filename="clusters_visualization.png"):
        """
        Visualize clusters and true labels on the GSOM map using a scatter plot with a combined legend.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Ensure required columns exist
        if 'cluster' not in self.node_labels.columns or 'label' not in self.node_labels.columns:
            raise ValueError("Clusters or true labels have not been computed. Run build_data_skeleton with a dataset containing a label column.")

        # Prepare data
        plot_data = self.node_labels.dropna(subset=['x', 'y', 'cluster', 'label'])
        if plot_data.empty:
            raise ValueError("No data available for visualization.")
        # Ensure 'label' is scalar by taking the first value if it's a list
        plot_data['label'] = plot_data['label'].apply(lambda x: x[0] if isinstance(x, list) else x)

        # Create scatter plot with two hues
        plt.figure(figsize=(12, 10), dpi=300)
        scatter = sns.scatterplot(data=plot_data, x='x', y='y', hue='cluster', style='label', 
                                palette='tab20', s=100, alpha=0.6, markers=['o', 's', '^', 'D', '*', 'p', 'X'],)
        
        # Customize plot
        plt.title("GSOM Clusters vs True Labels Visualization")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.grid(True)  # Add grid as requested
        
        # Create a combined legend
        handles, labels = scatter.get_legend_handles_labels()
        legend_labels = []
        for h, l in zip(handles, labels):
            cluster_label = f"Cluster {l.split(',')[0].replace('cluster=', '')}" if 'cluster=' in l else l
            true_label_idx = int(l.split(',')[0].replace('cluster=', '')) if 'cluster=' in l else None
            true_label = plot_data[plot_data['cluster'] == true_label_idx]['label'].iloc[0] if true_label_idx is not None else l
            legend_labels.append(f"{cluster_label}, True Label {true_label}")
        
        plt.legend(handles, legend_labels, title="Cluster / True Label", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(filename)
        plt.close()
        print(f"Cluster visualization saved as {filename}")

    def visualize_skeleton(self, paths, hit_points, junctions, index_col, figure_label="GSOM with DSM Skeleton"):
        """
        Visualize the GSOM map with DSM skeleton, highlighting POS, hit points, and junctions.
        """
        plt.figure(figsize=(10, 12), dpi=300)
        fig, ax = plt.subplots()
        
        # Plot all GSOM nodes
        sns.scatterplot(
            x=self.node_coordinate[:self.node_count, 0],
            y=self.node_coordinate[:self.node_count, 1],
            color=sns.color_palette()[7],
            alpha=0.3,
            s=10,
            ax=ax
        )
        
        # Plot hit points
        sns.scatterplot(
            x=hit_points['x'],
            y=hit_points['y'],
            size=hit_points['hit_count'],
            color=sns.color_palette()[0],
            ax=ax
        )
        # Add legend manually if needed:
        hit_points_handle = ax.scatter([], [], color=sns.color_palette()[0], s=30, label='Hit Points')
        
        # Plot junctions
        if junctions:
            j_nodes, j_x, j_y = zip(*junctions)
            sns.scatterplot(
                x=j_x,
                y=j_y,
                color=sns.color_palette()[1],
                s=50,
                marker='s',
                ax=ax
            )
            # Add legend manually if needed:
            junctions_handle = ax.scatter([], [], color=sns.color_palette()[1], s=50, marker='s', label='Junctions')
        
        # Plot Paths of Spread
        for path in paths:
            x_coords = [node[1] for node in path]
            y_coords = [node[2] for node in path]
            plt.plot(x_coords, y_coords, color=sns.color_palette()[2], alpha=0.5)

        # Add labels for hit points
        for _, row in hit_points.iterrows():
            if row['hit_count'] > 0:
                label = ", ".join(map(str, row[index_col][:3]))
                ax.text(row['x'] + 0.3, row['y'] + 0.3, label, ha='left', va='center', fontsize=5)
        
        ax.set_title(figure_label)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.legend(handles=[hit_points_handle, junctions_handle])
        plt.savefig('gsom_dsm_skeleton.png')
        plt.close()

    def evaluate_clusters(self, data, index_col):
        """
        Evaluate clustering quality using Silhouette Score and Davies-Bouldin Index.
        """
        weight_columns = list(data.columns.values)
        weight_columns.remove(index_col)
        data_n = data[weight_columns].to_numpy()
        labels = self.output['cluster'].to_numpy()
        valid_indices = labels != -1
        if len(np.unique(labels[valid_indices])) > 1:
            sil_score = silhouette_score(data_n[valid_indices], labels[valid_indices])
            db_score = davies_bouldin_score(data_n[valid_indices], labels[valid_indices])
            return sil_score, db_score
        return None, None

    def analyze_idionomic_features(self, data, index_col):
        """
        Analyze idionomic (individual-specific) features by computing mean feature values per cluster.
        """
        weight_columns = list(data.columns.values)
        weight_columns.remove(index_col)
        data_n = data[weight_columns].copy()
        data_n['cluster'] = self.output['cluster']
        cluster_features = data_n.groupby('cluster')[weight_columns].mean()
        return cluster_features

    def visualize_idionomic_heatmap(self, cluster_features, filename="idionomic_heatmap.png"):
        plt.figure(figsize=(12, 8), dpi=300)
        sns.heatmap(cluster_features, annot=True, cmap="coolwarm", center=0, fmt=".2f")
        plt.title("Idionomic Features: Mean Values per Cluster")
        plt.xlabel("Features")
        plt.ylabel("Cluster")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def visualize_idionomic_bar(self, cluster_features, filename="idionomic_bar.png"):
        plt.figure(figsize=(14, 8), dpi=300)
        cluster_features.plot(kind='bar', stacked=False)
        plt.title("Idionomic Features: Mean Values per Cluster")
        plt.xlabel("Cluster")
        plt.ylabel("Mean Feature Value (Standardized)")
        plt.legend(title="Features", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def visualize_idionomic_top_features(self, cluster_features, top_n=5, filename="idionomic_top_features.png"):
        plt.figure(figsize=(12, 8), dpi=300)
        for cluster in cluster_features.index:
            top_features = cluster_features.loc[cluster].abs().sort_values(ascending=False).head(top_n)
            plt.barh([f"Cluster {cluster}: {feat}" for feat in top_features.index], top_features.values, label=f'Cluster {cluster}')
        plt.title(f"Top {top_n} Idionomic Features per Cluster")
        plt.xlabel("Absolute Mean Feature Value")
        plt.ylabel("Feature")
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

if __name__ == '__main__':
    np.random.seed(16)

    data_filename = "example/data/zoo.txt".replace('\\', '/')
    df = pd.read_csv(data_filename)

    # Load and preprocess data
    # df = pd.read_csv('problematic_internet_behaviors.csv')
    # df['IV'] = df['IV'] + '_' + df['DV']
    # df.drop(columns=['DV', 'SE'], inplace=True)
    # df = df.sort_values(by=['ID', 'IV'])
    # df = df.pivot(index='ID', columns='IV', values='Score').reset_index()
    # cols = df.drop(columns=['ID']).columns
    # df[cols] = df[cols].astype(float)
    # scaler = StandardScaler()
    # df[cols] = scaler.fit_transform(df[cols])
    print(len(df.columns))
    # Train GSOM
    data_train = df.iloc[:, 1:17]
    gsom_map = GSOM(spread_factor=0.75, dimensions=16)
    gsom_map.fit(data_train.to_numpy(), training_iterations=100, smooth_iterations=50)
    map_points = gsom_map.predict(df, "Name", "label")
    
    # Build and visualize DSM
    paths, hit_points, junctions, path_segments, clusters = gsom_map.build_data_skeleton(data_train, k_multiplier=5.0, index_col="Name", label_col="label")
    gsom_map.visualize_skeleton(paths, hit_points, junctions, index_col="Name")
    
    num_clusters = len(gsom_map.output['cluster'].value_counts()) - (1 if -1 in gsom_map.output['cluster'].values else 0)
    print(f"Number of clusters: {num_clusters}")
    gsom_map.visualize_clusters()

    # Evaluate clustering
    sil_score, db_score = gsom_map.evaluate_clusters(df, "Name")
    print(f"Silhouette Score: {sil_score:.3f}" if sil_score is not None else "Silhouette Score: Not enough clusters")
    print(f"Davies-Bouldin Score: {db_score:.3f}" if db_score is not None else "Davies-Bouldin Score: Not enough clusters")
    
    # Analyze idionomic features
    cluster_features = gsom_map.analyze_idionomic_features(df, "Name")
    gsom_map.visualize_idionomic_heatmap(cluster_features)
    gsom_map.visualize_idionomic_bar(cluster_features)
    gsom_map.visualize_idionomic_top_features(cluster_features, top_n=5)

    print("\nIdionomic Features (Mean per Cluster):")
    # print(cluster_features)
    cluster_features.to_csv("cluster_features.csv")
    
    print("DSM analysis complete. Skeleton visualization saved as 'gsom_dsm_skeleton.png'.")