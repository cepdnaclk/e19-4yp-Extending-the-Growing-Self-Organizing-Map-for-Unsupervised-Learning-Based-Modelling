import os
import pandas as pd
import numpy as np
from minisom import MiniSom
import gsom  # Assuming GSOM is a custom module
from scipy.spatial.distance import euclidean, cdist, cityblock
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from itertools import combinations

def zrehen_measure_new(weights, grid):
    """
    Compute Zrehen Measure (ZM) for topology preservation.
    Args:
        weights: (n_neurons, input_dim) array of weight vectors
        grid: (n_neurons, 2) array of grid coordinates
    Returns:
        zm: Normalized Zrehen Measure
    """
    n_neurons = weights.shape[0]
    intruders = 0
    
    # Define neighbors in output space (Manhattan distance <= 1)
    def is_neighbor(i, j):
        return cityblock(grid[i], grid[j]) <= 1
    
    # def is_neighbor(i, j, k=4):
    #     distances = np.array([euclidean(weights[i], weights[m]) for m in range(n_neurons)])
    #     k_nearest = np.argsort(distances)[:k]
    #     return j in k_nearest
    
    # Check all pairs of neighboring neurons (iterate over all unique pairs of neurons r, r')
    for r, r_prime in combinations(range(n_neurons), 2):
        # Check if r and r' are neighbors
        if is_neighbor(r, r_prime):
            # Compute distances
            w_r = weights[r]
            w_r_prime = weights[r_prime]
            dist_rr_prime = euclidean(w_r, w_r_prime) ** 2
            
            # Check for intruders (Iterates over all other neurons r" not equal to r or r')
            for r_double_prime in range(n_neurons):
                if r_double_prime != r and r_double_prime != r_prime:
                    w_r_double_prime = weights[r_double_prime]
                    dist_r_r_double_prime = euclidean(w_r, w_r_double_prime) ** 2
                    dist_r_prime_r_double_prime = euclidean(w_r_prime, w_r_double_prime) ** 2
                    
                    # Intruder condition
                    if (dist_r_r_double_prime + dist_r_prime_r_double_prime) <= dist_rr_prime:
                        intruders += 1
    
    # Normalize by number of neurons ()
    zm = intruders / n_neurons
    return zm

# C-Measure (CM) with Minimal Wiring Scheme
def c_measure_new(weights, grid, p=1):
    """
    Compute C-Measure (CM) for topology preservation using minimal wiring scheme.
    Args:
        weights: (n_neurons, input_dim) array of weight vectors
        grid: (n_neurons, 2 or 3) array of grid coordinates
        p: Power for output space distance (default=1, Manhattan distance)
    Returns:
        cm: Normalized C-Measure
    """
    n_neurons = weights.shape[0]
    print(f"Number of neurons: {n_neurons}")
    
    # Initialize cost
    cost = 0
    
    # # Define neighbors in input space (simplified: use small Euclidean distance threshold)
    # def is_input_neighbor(i, j, threshold=0.1):
    #     return euclidean(weights[i], weights[j]) <= threshold
    
    def is_input_neighbor(i, j, k=4):
        distances = np.array([euclidean(weights[i], weights[m]) for m in range(n_neurons)])
        k_nearest = np.argsort(distances)[:k]
        return j in k_nearest
    
    # def is_input_neighbor(i, j):
    #     tri = Delaunay(weights)
    #     # Check if i, j share an edge in the triangulation
    #     for simplex in tri.simplices:
    #         if {i, j}.issubset(set(simplex)):
    #             return True
    #     return False
    
    # Compute cost for all pairs
    for i, j in combinations(range(n_neurons), 2):
        d_A = cityblock(grid[i], grid[j]) ** p  # Output space distance
        d_V = 1 if is_input_neighbor(i, j) else 0  # Binary connectivity in input space
        cost += d_V * d_A
    
    # Normalize by number of neurons
    cm = cost / n_neurons
    return cm

def topographic_product_new(weights, grid, K=None):
    """
    Compute Topographic Product (TP) for topology preservation.
    Args:
        weights: (n_neurons, input_dim) array of weight vectors
        grid: (n_neurons, 2) array of grid coordinates
        K: Number of nearest neighbors to consider (default: N-1)
    Returns:
        tp: Topographic Product
    """
    n_neurons = weights.shape[0]
    K = n_neurons - 1 if K is None else min(K, n_neurons - 1)
    P = 0

    for j in range(n_neurons):
        # Compute distances
        d_V = np.array([euclidean(weights[j], weights[l]) for l in range(n_neurons)])
        d_A = np.array([cityblock(grid[j], grid[l]) for l in range(n_neurons)])
        
        # Nearest neighbors (excluding self)
        n_V = np.argsort(d_V)[1:K+1]  # k-th nearest in V
        n_A = np.argsort(d_A)[1:K+1]  # k-th nearest in A
        
        for k in range(K):
            # Compute product of Q1 * Q2 for l=1 to k
            product = 1
            for l in range(k + 1):
                Q1_l = d_V[n_A[l]] / d_V[n_V[l]] if d_V[n_V[l]] != 0 else 1
                Q2_l = d_A[n_A[l]] / d_A[n_V[l]] if d_A[n_V[l]] != 0 else 1
                product *= Q1_l * Q2_l
            P3 = product ** (1 / (2 * (k + 1))) # if product > 0 else 1
            
            # Sum log(P3)
            P += np.log(P3) if P3 != 1 else 0
    
    # Normalize
    tp = P / (n_neurons * K)
    return tp

def topographic_error_new(data, weights, grid):
    """
    Compute Topographic Error (TE) for topology preservation.
    Args:
        data: (n_points, input_dim) array of input data points
        weights: (n_neurons, input_dim) array of weight vectors
        grid: (n_neurons, 2) array of grid coordinates
    Returns:
        te: Topographic Error
    """
    n_points = data.shape[0]
    errors = 0
    
    # Define adjacency in output space (Manhattan distance <= 1)
    def is_adjacent(i, j):
        return cityblock(grid[i], grid[j]) <= 1
    
    # For each data point, find first and second BMUs
    for x in data:
        # Compute distances to all weights
        distances = np.array([euclidean(x, w) for w in weights])
        # Find indices of first and second BMUs
        bmu1_idx = np.argmin(distances)
        distances[bmu1_idx] = np.inf  # Exclude first BMU
        bmu2_idx = np.argmin(distances)
        
        # Check if BMUs are adjacent
        if not is_adjacent(bmu1_idx, bmu2_idx):
            errors += 1
    
    # Compute TE
    te = errors / n_points
    return te

# def get_neighbors(lattice_positions, grid_shape):
#     """
#     Computes neighbors for a 2D rectangular lattice (SOM), as provided (Page 4).
#     """
#     rows, cols = grid_shape
#     N = len(lattice_positions)
#     neighbors = [[] for _ in range(N)]
#     for i in range(N):
#         x, y = lattice_positions[i]
#         for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
#             nx, ny = x + dx, y + dy
#             if 0 <= nx < rows and 0 <= ny < cols:
#                 neighbor_idx = np.where((lattice_positions[:, 0] == nx) & 
#                                        (lattice_positions[:, 1] == ny))[0]
#                 if len(neighbor_idx) > 0:
#                     neighbors[i].append(neighbor_idx[0])
#     return neighbors

# def get_neighbors_gsom(gsom_map):
#     """
#     Computes neighbors for GSOM based on grown connections (Page 2).
#     """
#     node_coords = gsom_map.node_coordinate[:gsom_map.node_count]
#     N = gsom_map.node_count
#     neighbors = [[] for _ in range(N)]
#     for i in range(N):
#         x_i, y_i = node_coords[i]
#         for j in range(N):
#             if i != j:
#                 x_j, y_j = node_coords[j]
#                 if (abs(x_i - x_j) == 1 and y_i == y_j) or (abs(y_i - y_j) == 1 and x_i == x_j):
#                     neighbors[i].append(j)
#     return neighbors

def load_dataset(file_path, is_3d=False):
    """
    Loads a dataset file (CSV) into a NumPy array.
    """
    df = pd.read_csv(file_path)
    cols = ['x', 'y', 'z'] if is_3d else ['x', 'y']
    data = df[cols[:2 if not is_3d else 3]].values
    # Normalize to [0,1]
    data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-10)
    return data

def main(dataset_name, is_3d=False):
    """
    Tests topology preservation measures (ZM, CM, TP, TE) on a single dataset
    using SOM (minisom) and GSOM (gsom library), saving results and parameters to Excel.
    """
    # SOM and GSOM parameters
    grid_size = 20  # Size of the grid for SOM 
    grid_shape = (grid_size, grid_size)
    som_iterations = 600
    som_learning_rate = 0.05
    gsom_spread_factor = 0.3
    gsom_learning_rate = 0.05
    gsom_iterations = 60
    gsom_smoothing = 30
    gsom_max_radius = 4

    # Define parameters
    parameters = {
        'Grid Size': grid_size,
        'SOM Iterations': som_iterations,
        'SOM Learning Rate': som_learning_rate,
        'GSOM Spread Factor': gsom_spread_factor,
        'GSOM Learning Rate': gsom_learning_rate,
        'GSOM Iterations': gsom_iterations,
        'GSOM Smoothing': gsom_smoothing,
        'GSOM Max Radius': gsom_max_radius
    }

    # Create directories for datasets
    os.makedirs('data/shapes', exist_ok=True)
    os.makedirs('data/fcps', exist_ok=True)

    # Determine dataset type and path
    shapes_datasets = ['circle', 'triangle', 'rectangle', 'square', 'trapezoid', 'hexagon']
    file_path = f'data/shapes/{dataset_name}.csv' if dataset_name in shapes_datasets else f'data/fcps/{dataset_name}.csv'

    # Load data
    if not os.path.exists(file_path):
        print(f"Dataset {dataset_name} not found at {file_path}. Please download or provide.")
        return
    data = load_dataset(file_path, is_3d)
    # Train SOM
    som = MiniSom(grid_size, grid_size, data.shape[1], sigma=1.0, learning_rate=som_learning_rate)
    som.train_random(data, som_iterations)
    weights_som = som.get_weights().reshape(-1, data.shape[1])
    # lattice_som = np.array([[i, j] for i in range(grid_size) for j in range(grid_size)])
    grid_x, grid_y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    lattice_som = np.vstack((grid_x.flatten(), grid_y.flatten())).T
    print(f"Number of neurons in SOM: {weights_som.shape[0]} ")

    # Train GSOM
    gsom_map = gsom.GSOM(spred_factor=gsom_spread_factor, 
                         dimensions=data.shape[1], 
                         learning_rate=gsom_learning_rate, 
                         max_radius=gsom_max_radius)
    gsom_map.fit(data, gsom_iterations, gsom_smoothing)
    weights_gsom = gsom_map.node_list[:gsom_map.node_count]
    lattice_gsom = gsom_map.node_coordinate[:gsom_map.node_count]
    neurons = gsom_map.node_count
    print(f"Number of neurons in GSOM: {neurons}")


    # Compute topology preservation measures
    results = []

    cm_som_new = c_measure_new(weights_som, lattice_som, p=1)
    zm_som_new = zrehen_measure_new(weights_som, lattice_som)
    tp_som_new = topographic_product_new(weights_som, lattice_som)
    te_som_new = topographic_error_new(data, weights_som, lattice_som)

    cm_gsom_new = c_measure_new(weights_gsom, lattice_gsom, p=1)
    zm_gsom_new = zrehen_measure_new(weights_gsom, lattice_gsom)
    tp_gsom_new = topographic_product_new(weights_gsom, lattice_gsom)
    te_gsom_new = topographic_error_new(data, weights_gsom, lattice_gsom)


    # Append results to the list
    results.append({
        'Dataset': dataset_name,
        'Model': 'SOM',
        'Grid Size': grid_size,
        'TE_new': te_som_new,
        'CM_new': cm_som_new,
        'ZM_new': zm_som_new,
        'TP_new': tp_som_new
    })
    results.append({
        'Dataset': dataset_name,
        'Model': 'GSOM',
        'Neurons': neurons,
        'CM_new': cm_gsom_new,
        'ZM_new': zm_gsom_new,
        'TP_new': tp_gsom_new,
        'TE_new': te_gsom_new

    })

    # Print parameters
    print("\nParameters:")
    for key, value in parameters.items():
        print(f"  {key}: {value}")

    print("\nResults:")
    for result in results:
        print(f"  {result['Model']} - Dataset: {result['Dataset']}")
        print(f"   ZM_new:{result['ZM_new']:.4f},CM_new: {result['CM_new']:.4f}, TP_new:{result['TP_new']:.4f}, TE_new: {result['TE_new']:.4f}")
    

if __name__ == "__main__":
    # Example: Test a single dataset
    main(dataset_name='circle', is_3d=False)
    # main(dataset_name='hepta', is_3d=True)