import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean, cdist
from minisom import MiniSom
import gsom
import os

def zrehen_measure(weights, lattice_positions, neighbors, neurons):
    """
    Implements Zrehen Measure (ZM) as provided (Page 7).
    Purpose: Quantifies local consistency by counting 'intruders' that disrupt the geometric
    arrangement of neighboring neurons' weight vectors. Lower ZM indicates better topology
    preservation. Applied to Shapes and FCPS datasets to compare SOM/GSOM (Pages 12–17).
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

def c_measure(weights, lattice_positions,neurons, p=2):
    """
    Implements C-Measure (CM) as provided (Page 8).
    Purpose: Quantifies overall topological alignment using a cost function summing the product
    of input and map space distances. Lower CM indicates better preservation. Applied to
    Shapes and FCPS datasets (Pages 12–17).
    """
    N = len(weights)
    cost = 0.0
    input_distances = cdist(weights, weights, metric='euclidean')
    map_distances = cdist(lattice_positions, lattice_positions, metric='euclidean') ** p
    for i in range(N):
        for j in range(N):
            if i != j:
                cost += input_distances[i, j] * map_distances[i, j]
    cm = cost / neurons
    return cm

def topographic_product(weights, lattice_positions):
    """
    Implements Topographic Product (TP) as provided (Page 8–9).
    Purpose: Measures mismatch between neighborhood structures to assess dimensionality
    suitability. TP ≈ 0 indicates good preservation. Applied to Shapes and FCPS datasets
    (Pages 12–17).
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
    Implements Topographic Error (TE) as provided (Page 9).
    Purpose: Measures the proportion of data points whose first and second BMUs are not
    adjacent, evaluating mapping continuity. Lower TE indicates better preservation.
    Applied to Shapes and FCPS datasets (Pages 12–17).
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
    Computes neighbors for a 2D rectangular lattice (SOM), as provided (Page 4).
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

def get_neighbors_gsom(gsom_map):
    """
    Computes neighbors for GSOM based on grown connections (Page 2).
    Parameters:
    - gsom_map: GSOM object from the gsom library.
    Returns:
    - neighbors: list of lists, neighbors[i] contains indices of adjacent neurons.

    Assumptions:
    - GSOM’s map dictionary and node coordinates define the lattice structure (Page 5).
    - Nodes are adjacent if their coordinates differ by ±1 in x or y (based on grid growth).
    """
    node_coords = gsom_map.node_coordinate[:gsom_map.node_count]
    N = gsom_map.node_count
    neighbors = [[] for _ in range(N)]
    for i in range(N):
        x_i, y_i = node_coords[i]
        for j in range(N):
            if i != j:
                x_j, y_j = node_coords[j]
                if (abs(x_i - x_j) == 1 and y_i == y_j) or (abs(y_i - y_j) == 1 and x_i == x_j):
                    neighbors[i].append(j)
    return neighbors

def load_dataset(file_path, is_3d=False):
    """
    Loads a dataset file (CSV) into a NumPy array.
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
    # Normalize to [0,1] to match SOM/GSOM input requirements (implied, Page 10)
    data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-10)
    return data

def main():
    """
    Tests topology preservation measures (ZM, CM, TP, TE) on Shapes and FCPS datasets
    using SOM (minisom) and GSOM (gsom library), as described in the paper (Pages 9–17).

    Datasets (Pages 10–11):
    - Shapes: 6 2D datasets (circle, triangle, rectangle, square, trapezoid, hexagon),
      ~441 points each (Page 10). Assumed to be locally available CSVs in 'data/shapes'
      (e.g., 'circle.csv').
    - FCPS: 10 datasets (Hepta, Lsun, Tetra, Target, Chainlink, Atom, EngyTime, TwoDiamonds,
      WingNut, GolfBall), with sizes and dimensions in Table 1 (Page 11). Assumed to be
      downloadable via DOI (Page 21) or locally available as CSVs in 'data/fcps' (e.g., 'hepta.csv').

    Instructions for Obtaining Datasets:
    - Shapes: Synthetically generated (Page 10). Obtain from the paper’s authors or generate
      locally (not done here, as per user request). Place CSVs in 'data/shapes' with names
      'circle.csv', etc., each with 'x', 'y' columns.
    - FCPS: Available via DOI https://doi.org/10.1016/j.dib.2020.105501 (Page 21).
      1. Visit the Data in Brief article to download CSVs from the associated repository
         (e.g., University of Marburg’s clustering benchmark).
      2. Place CSVs in 'data/fcps' with names 'hepta.csv', etc.
      3. Alternatively, check repositories like GitHub for FCPS datasets.
    - Format: CSVs with 'x', 'y' (2D) or 'x', 'y', 'z' (3D) columns, possibly with class
      labels (ignored, Page 11).

    Preprocessing:
    - Load CSVs, extract coordinates, normalize to [0,1] (implied for synthetic data, Page 10).
    - Shapes: 2D (n, 2). FCPS: 2D or 3D (n, 2 or n, 3) per Table 1.

    SOM/GSOM Parameters (Page 12):
    - SOM: 60 iterations, learning rate 0.01, 10x10 grid (assumed, as SOM matches GSOM’s
      neuron count).
    - GSOM: 60 iterations, 30 smoothing iterations, learning rate 0.3, spread factor 0.5
      (example from [0.05, 0.15, 0.5, 0.83]).

    Assumptions:
    - Datasets are CSVs in 'data/shapes' and 'data/fcps' directories.
    - GSOM library matches the provided implementation, providing node_list, node_coordinate,
      and node_count.
    - GSOM connections are inferred from coordinate adjacency (Page 5).
    - 10x10 grid for SOM, adjustable for GSOM’s grown structure.
    """
    # Define datasets (Pages 10–11)
    shapes_datasets = ['circle', 'triangle', 'rectangle', 'square', 'trapezoid', 'hexagon']
    fcps_datasets = [
        ('hepta', True), ('lsun', False), ('tetra', True), ('target', False),
        ('chainlink', True), ('atom', True), ('engytime', False),
        ('twodiamonds', False), ('wingnut', False), ('golfball', True)
    ]

    # SOM and GSOM parameters (Page 12)
    grid_size = 10  # 10x10 grid for SOM
    grid_shape = (grid_size, grid_size)
    som_iterations = 60
    som_learning_rate = 0.01
    gsom_spread_factor = 0.5
    gsom_learning_rate = 0.3
    gsom_iterations = 60
    gsom_smoothing = 30
    gsom_max_radius = 4  # Matches provided GSOM example

    # SOM and GSOM parameters and results to the output and save to a CSV file



    # Create directories for datasets
    os.makedirs('data/shapes', exist_ok=True)
    os.makedirs('data/fcps', exist_ok=True)

    results = []
    # Process each dataset
    for dataset in shapes_datasets + [name for name, _ in fcps_datasets]:
        # Load data
        is_3d = any(dataset == name and is_3d for name, is_3d in fcps_datasets)
        file_path = f'data/shapes/{dataset}.csv' if dataset in shapes_datasets else f'data/fcps/{dataset}.csv'
        if not os.path.exists(file_path):
            print(f"Dataset {dataset} not found at {file_path}. Please download or provide.")
            continue
        data = load_dataset(file_path, is_3d)

        # Train SOM
        som = MiniSom(grid_size, grid_size, data.shape[1], sigma=1.0, learning_rate=som_learning_rate)
        som.train_random(data, som_iterations)
        weights_som = som.get_weights().reshape(-1, data.shape[1])
        lattice_som = np.array([[i, j] for i in range(grid_size) for j in range(grid_size)])
        neighbors_som = get_neighbors(lattice_som, grid_shape)

        # Train GSOM
        gsom_map = gsom.GSOM(spred_factor=gsom_spread_factor, 
                             dimensions=data.shape[1], 
                             learning_rate=gsom_learning_rate, 
                             max_radius=gsom_max_radius)
        gsom_map.fit(data, gsom_iterations, gsom_smoothing)
        weights_gsom = gsom_map.node_list[:gsom_map.node_count]
        lattice_gsom = gsom_map.node_coordinate[:gsom_map.node_count]
        neighbors_gsom = get_neighbors_gsom(gsom_map)
        neurons = gsom_map.node_count

        # Compute topology preservation measures
        zm_som = zrehen_measure(weights_som, lattice_som, neighbors_som, grid_size**2)
        cm_som = c_measure(weights_som, lattice_som,grid_size**2, p=2)
        tp_som = topographic_product(weights_som, lattice_som)
        te_som = topographic_error(data, weights_som, lattice_som, neighbors_som)

        zm_gsom = zrehen_measure(weights_gsom, lattice_gsom, neighbors_gsom,neurons)
        cm_gsom = c_measure(weights_gsom, lattice_gsom,neurons, p=2)
        tp_gsom = topographic_product(weights_gsom, lattice_gsom)
        te_gsom = topographic_error(data, weights_gsom, lattice_gsom, neighbors_gsom)


        # Append results to the list
        results.append({
            'Dataset': dataset,
            'Model': 'SOM',
            'ZM': zm_som,
            'CM': cm_som,
            'TP': tp_som,
            'TE': te_som
        })
        results.append({
            'Dataset': dataset,
            'Model': 'GSOM',
            'ZM': zm_gsom,
            'CM': cm_gsom,
            'TP': tp_gsom,
            'TE': te_gsom
        })
        # Print results
        print(f"\nDataset: {dataset}")
        print("SOM Results:")
        print(f"  ZM: {zm_som:.4f}, CM: {cm_som:.4f}, TP: {tp_som:.4f}, TE: {te_som:.4f}")
        print("GSOM Results:")
        print(f"  ZM: {zm_gsom:.4f}, CM: {cm_gsom:.4f}, TP: {tp_gsom:.4f}, TE: {te_gsom:.4f}")

        # Save results to a CSV file
        results_df = pd.DataFrame(results)
        results_df.to_csv('topology_measures_results.csv', index=False)
        print("\nResults saved to 'topology_measures_results.csv'")

if __name__ == "__main__":
    main()