import os
import pandas as pd
import numpy as np
from minisom import MiniSom
import gsom  # Assuming GSOM is a custom module
from scipy.spatial.distance import euclidean, cdist, cityblock
import uuid
import zipfile
import matplotlib.pyplot as plt
from itertools import combinations

def zrehen_measure(weights, lattice_positions, neighbors, neurons):
    """
    Implements Zrehen Measure (ZM) as provided (Page 7).
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
    
    # Check all pairs of neighboring neurons
    for r, r_prime in combinations(range(n_neurons), 2):
        if is_neighbor(r, r_prime):
            w_r = weights[r]
            w_r_prime = weights[r_prime]
            dist_rr_prime = euclidean(w_r, w_r_prime) ** 2
            
            # Check for intruders
            for r_double_prime in range(n_neurons):
                if r_double_prime != r and r_double_prime != r_prime:
                    w_r_double_prime = weights[r_double_prime]
                    dist_r_r_double_prime = euclidean(w_r, w_r_double_prime) ** 2
                    dist_r_prime_r_double_prime = euclidean(w_r_prime, w_r_double_prime) ** 2
                    
                    # Intruder condition
                    if (dist_r_r_double_prime + dist_r_prime_r_double_prime) <= dist_rr_prime:
                        intruders += 1
    
    # Normalize by number of neurons
    zm = intruders / n_neurons
    return zm


# C-Measure (CM) with Minimal Wiring Scheme
def c_measure_new(weights, grid, p=1):
    """
    Compute C-Measure (CM) for topology preservation using minimal wiring scheme.
    Args:
        weights: (n_neurons, input_dim) array of weight vectors
        grid: (n_neurons, 2) array of grid coordinates
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
    # Compute cost for all pairs
    for i, j in combinations(range(n_neurons), 2):
        d_A = cityblock(grid[i], grid[j]) ** p  # Output space distance
        d_V = 1 if is_input_neighbor(i, j) else 0  # Binary connectivity in input space
        cost += d_V * d_A
    
    # Normalize by number of neurons
    cm = cost / n_neurons
    return cm

# def c_measure(weights, lattice_positions, neurons, p=1):
#     """
#     Implements C-Measure (CM) as provided (Page 8).
#     """
#     N = len(weights)
#     print(f"Number of neurons: {N}")
#     cost = 0.0
#     input_distances = cdist(weights, weights, metric='euclidean')
#     map_distances = cdist(lattice_positions, lattice_positions, metric='euclidean') ** p
#     for i in range(N):
#         for j in range(N):
#             if i != j:
#                 cost += input_distances[i, j] * map_distances[i, j]
#     cm = cost / neurons
#     return cm
# def c_measure(weights, lattice_positions, p=1, threshold=0.1):
    N = len(weights)
    print(f"Number of neurons: {N}")
    cost = 0.0
    input_distances = cdist(weights, weights, metric='euclidean')
    map_distances = cdist(lattice_positions, lattice_positions, metric='cityblock') ** p
    for i in range(N):
        for j in range(N):
            if i != j:
                d_V = 1 if input_distances[i, j] <= threshold else 0
                cost += d_V * map_distances[i, j]
    cm = cost / N
    return cm

# def topographic_product(weights, lattice_positions):
#     """
#     Implements Topographic Product (TP) as provided (Page 8–9).
#     """
#     N = len(weights)
#     total_log = 0.0
#     input_distances = cdist(weights, weights, metric='euclidean')
#     map_distances = cdist(lattice_positions, lattice_positions, metric='euclidean')
#     for j in range(N):
#         input_neighbors = np.argsort(input_distances[j])[1:]
#         map_neighbors = np.argsort(map_distances[j])[1:]
#         for k in range(1, N):
#             product = 1.0
#             for i in range(1, k+1):
#                 n_v = input_neighbors[i-1]
#                 n_a = map_neighbors[i-1]
#                 num = input_distances[j, n_v] * map_distances[j, n_a]
#                 denom = input_distances[j, n_a] * map_distances[j, n_v]
#                 if denom > 0:
#                     product *= num / denom
#             if product > 0:
#                 total_log += np.log(product ** (1 / (2 * k)))
#     tp = total_log / (N**2 - N)
#     return tp

def topographic_product(weights, grid):
    """
    Compute Topographic Product (TP) for topology preservation.
    Args:
        weights: (n_neurons, input_dim) array of weight vectors
        grid: (n_neurons, 2) array of grid coordinates
    Returns:
        tp: Topographic Product
    """
    n_neurons = weights.shape[0]
    print(f"Number of neurons: {n_neurons}")
    # Initialize total
    total = 0
    
    # Compute distances in input and output spaces
    d_V = np.zeros((n_neurons, n_neurons))
    d_A = np.zeros((n_neurons, n_neurons))
    for i, j in combinations(range(n_neurons), 2):
        d_V[i, j] = d_V[j, i] = euclidean(weights[i], weights[j])
        d_A[i, j] = d_A[j, i] = cityblock(grid[i], grid[j])
    
    # For each neuron, compute neighbor rankings
    for j in range(n_neurons):
        # Sort neighbors by distance
        V_neighbors = np.argsort(d_V[j])[1:]  # Exclude self
        A_neighbors = np.argsort(d_A[j])[1:]  # Exclude self
        
        for k in range(1, n_neurons):
            product = 1
            for i in range(k):
                n_V = V_neighbors[i]
                n_A = A_neighbors[i]
                # Compute product of distance ratios
                ratio1 = d_V[j, n_A] / d_V[j, n_V] ############ Error correction
                ratio2 = d_A[j, n_A] / d_A[j, n_V]
                # print(f"Ratio1: {ratio1}, Ratio2: {ratio2}") 

                product *= ratio1 * ratio2

            total += np.log(product ** 0.5)
    
    # Normalize
    tp = total / (n_neurons * (n_neurons - 1))
    return tp
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

def topographic_error(data, weights, lattice_positions, neighbors):
    """
    Implements Topographic Error (TE) as provided (Page 9).
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
    """
    df = pd.read_csv(file_path)
    cols = ['x', 'y', 'z'] if is_3d else ['x', 'y']
    data = df[cols[:2 if not is_3d else 3]].values
    # Normalize to [0,1]
    data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-10)
    return data

def is_valid_excel(file_path):
    """
    Check if the Excel file is a valid .xlsx file by verifying the presence of [Content_Types].xml.
    """
    try:
        with zipfile.ZipFile(file_path, 'r') as z:
            return '[Content_Types].xml' in z.namelist()
    except zipfile.BadZipFile:
        return False

def visualize_som(som):
    """
    Visualizes the SOM map using the distance map.
    """
    plt.figure(figsize=(8, 8))
    plt.pcolor(som.distance_map().T, cmap='gist_yarg')  # Plot the distance map
    plt.colorbar()  # Show the color bar
    plt.title("SOM Distance Map")
    plt.show()

def visualize_gsom(gsom_map, data, name_column, label_column):
    """
    Visualizes the GSOM map using the gsom.plot method.
    """
    map_points = gsom_map.predict(data, name_column,name_column)
    gsom.plot(map_points, name_column, gsom_map=gsom_map)
def main(dataset_name, is_3d=False):
    """
    Tests topology preservation measures (ZM, CM, TP, TE) on a single dataset
    using SOM (minisom) and GSOM (gsom library), saving results and parameters to Excel.
    """
    # SOM and GSOM parameters
    grid_size = 16  # Size of the grid for SOM 
    grid_shape = (grid_size, grid_size)
    som_iterations = 60
    som_learning_rate = 0.01
    gsom_spread_factor = 0.5
    gsom_learning_rate = 0.01
    gsom_iterations = 60
    gsom_smoothing = 30
    gsom_max_radius = 4

    # Define parameters
    parameters = {
        'Run ID': str(uuid.uuid4()),  # Unique ID for each run
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

    # Visualize SOM map
    # visualize_som(som)
    
    # Visualize GSOM map
    # visualize_gsom(gsom_map, data, "Name", "label")  # Replace "Name" and "label" with appropriate column names

    # Compute topology preservation measures
    results = []
    # zm_som = zrehen_measure(weights_som, lattice_som, neighbors_som, grid_size**2)
    # cm_som = c_measure(weights_som, lattice_som, p=1)
    # tp_som = topographic_product(weights_som, lattice_som)
    te_som = topographic_error(data, weights_som, lattice_som, neighbors_som)
    te_som_new = topographic_error_new(data, weights_som, lattice_som)
    # cm_som_new = c_measure_new(weights_som, lattice_som, p=1)
    # zm_som_new = zrehen_measure_new(weights_som, lattice_som)
    # tp_som_new = topographic_product_new(weights_som, lattice_som)

    # zm_gsom = zrehen_measure(weights_gsom, lattice_gsom, neighbors_gsom, neurons)
    # cm_gsom = c_measure(weights_gsom, lattice_gsom, p=1)
    # tp_gsom = topographic_product(weights_gsom, lattice_gsom)
    te_gsom = topographic_error(data, weights_gsom, lattice_gsom, neighbors_gsom)
    te_gsom_new = topographic_error_new(data, weights_gsom, lattice_gsom)
    # cm_gsom_new = c_measure_new(weights_gsom, lattice_gsom, p=1)
    # zm_gsom_new = zrehen_measure_new(weights_gsom, lattice_gsom)
    # tp_gsom_new = topographic_product_new(weights_gsom, lattice_gsom)

    # Append results to the list
    results.append({
        'Run ID': parameters['Run ID'],
        'Dataset': dataset_name,
        'Model': 'SOM',
        'Grid Size': grid_size,
        # 'ZM': zm_som,
        # 'CM': cm_som,
        # 'TP': tp_som,
        'TE': te_som,
        'TE_new': te_som_new,
        # 'CM_new': cm_som_new,
        # 'ZM_new': zm_som_new,
        # 'TP_new': tp_som_new
    })
    results.append({
        'Run ID': parameters['Run ID'],
        'Dataset': dataset_name,
        'Model': 'GSOM',
        'Neurons': neurons,
        # 'ZM': zm_gsom,
        # 'CM': cm_gsom,
        # 'TP': tp_gsom,
        'TE': te_gsom,
        'TE_new': te_gsom_new,
        # 'CM_new': cm_gsom_new,
        # 'ZM_new': zm_gsom_new,
        # 'TP_new': tp_gsom_new

    })

    # Print results
    # print(f"\nDataset: {dataset_name}")
    # print("SOM Results:")
    # print(f"  CM: {cm_som:.4f} (CM_new: {cm_som_new:.4f})")
    # # print(f"  ZM: {zm_som:.4f}, CM: {cm_som:.4f}, TP: {tp_som:.4f}, TE: {te_som:.4f}")
    # print("GSOM Results:")
    # print(f"  CM: {cm_gsom:.4f} (CM_new: {cm_gsom_new:.4f})")
    # # print(f"  ZM: {zm_gsom:.4f}, CM: {cm_gsom:.4f}, TP: {tp_gsom:.4f}, TE: {te_gsom:.4f}")

    # Print parameters
    print("\nParameters:")
    for key, value in parameters.items():
        print(f"  {key}: {value}")

    print("\nResults:")
    for result in results:
        print(f"  {result['Model']} - Dataset: {result['Dataset']}, Run ID: {result['Run ID']}")
        print(f"   TE: {result['TE']:.4f}, TE_new: {result['TE_new']:.4f}")
    

if __name__ == "__main__":
    # Example: Test a single dataset
    main(dataset_name='Lsun', is_3d=False)
    # main(dataset_name='hepta', is_3d=True)