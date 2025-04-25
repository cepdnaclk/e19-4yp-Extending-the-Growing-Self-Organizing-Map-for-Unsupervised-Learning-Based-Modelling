import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean, cdist
from minisom import MiniSom
import os
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

# Note: GSOM library is less standard. Assuming a placeholder implementation.
# If a specific GSOM library exists (e.g., 'gsom'), replace the placeholder.
# class GSOM:
#     def __init__(self, grid_size, input_dim, spread_factor, learning_rate):
#         self.grid_size = grid_size
#         self.input_dim = input_dim
#         self.spread_factor = spread_factor
#         self.learning_rate = learning_rate
#         # Placeholder weights and lattice
#         self.weights = np.random.rand(grid_size * grid_size, input_dim)
#         self.lattice_positions = np.array([[i, j] for i in range(grid_size) for j in range(grid_size)])
#         self.connections = []  # Placeholder for grown connections

#     def train(self, data, iterations, smoothing_iterations):
#         # Placeholder training logic (replace with actual GSOM implementation)
#         pass

#     def get_weights(self):
#         return self.weights

#     def get_lattice_positions(self):
#         return self.lattice_positions

#     def get_connections(self):
#         return self.connections

def zrehen_measure(weights, lattice_positions, neighbors):
    """
    Implements Zrehen Measure (ZM) as described in the paper (Page 7).
    Purpose: Quantifies local consistency by counting 'intruders' that disrupt the geometric
    arrangement of neighboring neurons' weight vectors. Lower ZM indicates better topology
    preservation (ZM = 0 for perfect preservation). Applied to Shapes and FCPS datasets to
    evaluate SOM/GSOM maps (Pages 12–17).

    Parameters:
    - weights: numpy array (N, d), N=neurons, d=input dimensions (2 for Shapes, 2/3 for FCPS).
    - lattice_positions: numpy array (N, 2), 2D coordinates in map space.
    - neighbors: list of lists, neighbors[i] contains indices of neurons adjacent to i.

    Returns:
    - zm: Normalized Zrehen Measure.
    """
    N = len(weights)
    intruder_count = 0
    for i in range(N):
        for j in neighbors[i]:
            if j > i:
                w_i, w_j = weights[i], weights[j]
                dist_ij_sq = euclidean(w_i, w_j) ** 2
                for k in range(N):
                    if k != i and k != j:
                        w_k = weights[k]
                        dist_ik_sq = euclidean(w_i, w_k) ** 2
                        dist_jk_sq = euclidean(w_j, w_k) ** 2
                        if dist_ik_sq + dist_jk_sq <= dist_ij_sq:
                            intruder_count += 1
    zm = intruder_count / N
    return zm

def c_measure(weights, lattice_positions, p=2):
    """
    Implements C-Measure (CM) as described in the paper (Page 8).
    Purpose: Quantifies overall topological alignment using a cost function summing the product
    of input and map space distances. Lower CM indicates better preservation. Applied to Shapes
    and FCPS datasets (Pages 12–17). Uses minimal wiring scheme.

    Parameters:
    - weights: numpy array (N, d), N=neurons, d=input dimensions.
    - lattice_positions: numpy array (N, 2), 2D coordinates in map space.
    - p: exponent for map space distance (d_A(i,j) = ||i-j||^p), default p=2.

    Returns:
    - cm: Normalized C-Measure.

    Assumptions:
    - Uses Euclidean distances for input space (d_V) due to ambiguity in the paper’s binary
      connectivity function (Page 8).
    """
    N = len(weights)
    cost = 0.0
    input_distances = cdist(weights, weights, metric='euclidean')
    map_distances = cdist(lattice_positions, lattice_positions, metric='euclidean') ** p
    for i in range(N):
        for j in range(N):
            if i != j:
                cost += input_distances[i, j] * map_distances[i, j]
    cm = cost / N
    return cm

def topographic_product(weights, lattice_positions):
    """
    Implements Topographic Product (TP) as described in the paper (Page 8–9).
    Purpose: Measures mismatch between neighborhood structures in input and map spaces to
    assess dimensionality suitability. TP ≈ 0 indicates good preservation. Applied to Shapes
    and FCPS datasets (Pages 12–17).

    Parameters:
    - weights: numpy array (N, d), N=neurons, d=input dimensions.
    - lattice_positions: numpy array (N, 2), 2D coordinates in map space.

    Returns:
    - tp: Topographic Product.

    Assumptions:
    - Summation over k=1 to N-1, as implied by the formula (Page 9).
    """
    N = len(weights)
    total_log = 0.0
    input_distances = cdist(weights, weights, metric='euclidean')
    map_distances = cdist(lattice_positions, lattice_positions, metric='euclidean')
    for j in range(N):
        input_neighbors = np.argsort(input_distances[j])[1:]
        map_neighbors = np.argsort(map_distances[j])[1:]
        for k in range(1, N):
            product = 1.0
            for i in range(1, k+1):
                n_v = input_neighbors[i-1]
                n_a = map_neighbors[i-1]
                num = input_distances[j, n_v] * map_distances[j, n_a]
                denom = input_distances[j, n_a] * map_distances[j, n_v]
                if denom > 0:
                    product *= num / denom
            if product > 0:
                total_log += np.log(product ** (1 / (2 * k)))
    tp = total_log / (N**2 - N)
    return tp

def topographic_error(data, weights, lattice_positions, neighbors):
    """
    Implements Topographic Error (TE) as described in the paper (Page 9).
    Purpose: Measures the proportion of data points whose first and second BMUs are not
    adjacent, evaluating mapping continuity. Lower TE indicates better preservation.
    Applied to Shapes and FCPS datasets (Pages 12–17).

    Parameters:
    - data: numpy array (K, d), K=data points, d=input dimensions.
    - weights: numpy array (N, d), N=neurons, d=input dimensions.
    - lattice_positions: numpy array (N, 2), 2D coordinates in map space.
    - neighbors: list of lists, neighbors[i] contains indices of adjacent neurons.

    Returns:
    - te: Topographic Error.
    """
    K = len(data)
    error_count = 0
    distances = cdist(data, weights, metric='euclidean')
    for i in range(K):
        sorted_indices = np.argsort(distances[i])
        bmu1, bmu2 = sorted_indices[0], sorted_indices[1]
        if bmu2 not in neighbors[bmu1]:
            error_count += 1
    te = error_count / K
    return te

def get_neighbors(lattice_positions, grid_shape):
    """
    Computes neighbors for a 2D rectangular lattice (SOM), as per the paper (Page 4).
    Parameters:
    - lattice_positions: numpy array (N, 2), 2D coordinates.
    - grid_shape: tuple (rows, cols), lattice shape.
    Returns:
    - neighbors: list of lists, neighbors[i] contains indices of adjacent neurons.
    """
    rows, cols = grid_shape
    N = len(lattice_positions)
    neighbors = [[] for _ in range(N)]
    for i in range(N):
        x, y = lattice_positions[i]
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols:
                neighbor_idx = np.where((lattice_positions[:, 0] == nx) & 
                                       (lattice_positions[:, 1] == ny))[0]
                if len(neighbor_idx) > 0:
                    neighbors[i].append(neighbor_idx[0])
    return neighbors

def get_neighbors_gsom(connections):
    """
    Computes neighbors for GSOM based on grown connections (Page 2).
    Parameters:
    - connections: list of tuples (i, j), edges between neurons.
    Returns:
    - neighbors: list of lists, neighbors[i] contains indices of adjacent neurons.

    Assumptions:
    - GSOM provides a list of connections from its grown structure, not specified in the
      paper but assumed for dynamic SOMs (Page 5).
    """
    N = max(max(i, j) for i, j in connections) + 1
    neighbors = [[] for _ in range(N)]
    for i, j in connections:
        neighbors[i].append(j)
        neighbors[j].append(i)
    return neighbors

def load_kaggle_dataset(dataset_slug, output_dir):
    """
    Downloads a Kaggle dataset using the Kaggle API.
    Parameters:
    - dataset_slug: str, Kaggle dataset identifier (e.g., 'username/shapes-dataset').
    - output_dir: str, directory to save the dataset.

    Instructions:
    1. Install kaggle: `pip install kaggle`
    2. Set up Kaggle API credentials:
       - Go to Kaggle.com > Your Profile > Account > Create New API Token.
       - Download kaggle.json and place it in ~/.kaggle/kaggle.json (Linux/Mac)
         or %USERPROFILE%\.kaggle\kaggle.json (Windows).
       - Run: `chmod 600 ~/.kaggle/kaggle.json` (Linux/Mac).
    3. Ensure the dataset exists on Kaggle by searching for 'Shapes dataset' or
       'FCPS dataset' and noting the slug (e.g., 'username/shapes-dataset').
    """
    os.makedirs(output_dir, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset_slug, path=output_dir, unzip=True)

def preprocess_dataset(file_path, is_3d=False):
    """
    Preprocesses a dataset file (CSV or similar) into a NumPy array.
    Parameters:
    - file_path: str, path to the dataset file.
    - is_3d: bool, True for 3D datasets (FCPS), False for 2D (Shapes, some FCPS).
    Returns:
    - data: numpy array (n, d), d=2 or 3.

    Assumptions:
    - Datasets are CSVs with columns 'x', 'y' (2D) or 'x', 'y', 'z' (3D), possibly with
      class labels (Page 11). Labels are ignored, as topology measures use coordinates.
    """
    df = pd.read_csv(file_path)
    cols = ['x', 'y', 'z'] if is_3d else ['x', 'y']
    data = df[cols[:2 if not is_3d else 3]].values
    return data

def main():
    """
    Tests topology preservation measures on Shapes and FCPS datasets using SOM and GSOM,
    as described in the paper (Pages 9–17).

    Instructions for Loading Datasets:
    - Shapes Dataset: 6 2D datasets (circle, triangle, rectangle, square, trapezoid, hexagon)
      (Page 10). Assume available on Kaggle as a single dataset with files
      'circle.csv', 'triangle.csv', etc., each with ~441 points (based on square, Page 10).
    - FCPS Dataset: 10 datasets (Hepta, Lsun, Tetra, Target, Chainlink, Atom, EngyTime,
      TwoDiamonds, WingNut, GolfBall) (Page 11). Assume available on Kaggle with files
      'hepta.csv', etc., with sizes and dimensions as in Table 1 (Page 11).
    - Kaggle Slugs: Replace 'username/shapes-dataset' and 'username/fcps-dataset' with
      actual slugs found on Kaggle by searching for the datasets.
    - Download: Use load_kaggle_dataset() to download and unzip files to 'data/shapes'
      and 'data/fcps'.

    Preprocessing:
    - Shapes: 2D coordinates (x, y), ~441 points per dataset.
    - FCPS: 2D (Lsun, Target, EngyTime, TwoDiamonds, WingNut) or 3D (Hepta, Tetra,
      Chainlink, Atom, GolfBall) coordinates. Ignore class labels (Page 11).
    - Normalize data to [0,1] to match SOM/GSOM input requirements, as implied by the
      paper’s synthetic data (Page 10).

    Assumptions:
    - Datasets are CSVs with 'x', 'y' (2D) or 'x', 'y', 'z' (3D) columns.
    - SOM uses a 10x10 grid (100 neurons), GSOM grows to a similar size (Page 12).
    - GSOM implementation is a placeholder; replace with a library like 'gsom' if available.
    - GSOM connections are assumed available from the grown structure (Page 5).
    """
    # Define datasets and their properties (Page 10–11)
    shapes_datasets = ['circle', 'triangle', 'rectangle', 'square', 'trapezoid', 'hexagon']
    fcps_datasets = [
        ('hepta', True), ('lsun', False), ('tetra', True), ('target', False),
        ('chainlink', True), ('atom', True), ('engytime', False),
        ('twodiamonds', False), ('wingnut', False), ('golfball', True)
    ]

    # Download datasets from Kaggle
    shapes_dir = 'data/shapes'
    fcps_dir = 'data/fcps'
    load_kaggle_dataset('username/shapes-dataset', shapes_dir)  # Replace with actual slug
    load_kaggle_dataset('username/fcps-dataset', fcps_dir)  # Replace with actual slug

    # SOM and GSOM parameters (Page 12)
    grid_size = 10  # 10x10 grid for SOM
    som_iterations = 60
    som_learning_rate = 0.01
    gsom_spread_factor = 0.5  # Example SF from [0.05, 0.15, 0.5, 0.83]
    gsom_learning_rate = 0.3
    gsom_iterations = 60
    gsom_smoothing = 30

    # Process each dataset
    for dataset in shapes_datasets + [name for name, _ in fcps_datasets]:
        # Load and preprocess data
        is_3d = any(dataset == name and is_3d for name, is_3d in fcps_datasets)
        file_path = f'{shapes_dir}/{dataset}.csv' if dataset in shapes_datasets else f'{fcps_dir}/{dataset}.csv'
        data = preprocess_dataset(file_path, is_3d)
        # Normalize data to [0,1]
        data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-10)

        # Train SOM
        som = MiniSom(grid_size, grid_size, data.shape[1], sigma=1.0, learning_rate=som_learning_rate)
        som.train_random(data, som_iterations)
        weights_som = som.get_weights().reshape(-1, data.shape[1])
        lattice_som = np.array([[i, j] for i in range(grid_size) for j in range(grid_size)])
        neighbors_som = get_neighbors(lattice_som, (grid_size, grid_size))

        # Train GSOM (placeholder)
        gsom = GSOM(grid_size, data.shape[1], gsom_spread_factor, gsom_learning_rate)
        gsom.train(data, gsom_iterations, gsom_smoothing)
        weights_gsom = gsom.get_weights()
        lattice_gsom = gsom.get_lattice_positions()
        connections_gsom = gsom.get_connections()  # Assume provided by GSOM
        neighbors_gsom = get_neighbors_gsom(connections_gsom)

        # Compute topology preservation measures
        zm_som = zrehen_measure(weights_som, lattice_som, neighbors_som)
        cm_som = c_measure(weights_som, lattice_som)
        tp_som = topographic_product(weights_som, lattice_som)
        te_som = topographic_error(data, weights_som, lattice_som, neighbors_som)

        zm_gsom = zrehen_measure(weights_gsom, lattice_gsom, neighbors_gsom)
        cm_gsom = c_measure(weights_gsom, lattice_gsom)
        tp_gsom = topographic_product(weights_gsom, lattice_gsom)
        te_gsom = topographic_error(data, weights_gsom, lattice_gsom, neighbors_gsom)

        # Print results
        print(f"Dataset: {dataset}")
        print("SOM Results:")
        print(f"  ZM: {zm_som:.4f}, CM: {cm_som:.4f}, TP: {tp_som:.4f}, TE: {te_som:.4f}")
        print("GSOM Results:")
        print(f"  ZM: {zm_gsom:.4f}, CM: {cm_gsom:.4f}, TP: {tp_gsom:.4f}, TE: {te_gsom:.4f}")

if __name__ == "__main__":
    main()