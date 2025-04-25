import numpy as np
from scipy.spatial.distance import euclidean, cdist

def zrehen_measure(weights, lattice_positions, neighbors):
    """
    Implements the Zrehen Measure (ZM) as described in the paper (Page 7).
    Purpose: Quantifies local consistency by counting 'intruders' that disrupt the expected
    geometric arrangement of neighboring neurons' weight vectors in the input space.
    A lower ZM indicates better topology preservation (ZM = 0 for perfect preservation).

    Parameters:
    - weights: numpy array of shape (N, d), where N is the number of neurons and d is the
      dimensionality of the input space (weight vectors w_i).
    - lattice_positions: numpy array of shape (N, 2), the 2D coordinates of neurons in the map space.
    - neighbors: list of lists, where neighbors[i] contains indices of neurons adjacent to neuron i
      in the map space (based on the rectangular lattice).

    Returns:
    - zm: Normalized Zrehen Measure (sum of intruders divided by number of neurons).
    """
    N = len(weights)
    intruder_count = 0

    # Iterate over all pairs of neighboring neurons in the map space
    for i in range(N):
        for j in neighbors[i]:
            if j > i:  # Avoid double-counting pairs (i,j) and (j,i)
                w_i = weights[i]
                w_j = weights[j]
                # Compute squared distance between weight vectors w_i and w_j
                dist_ij_sq = euclidean(w_i, w_j) ** 2
                # Check for intruders (other neurons k that violate the condition)
                for k in range(N):
                    if k != i and k != j:
                        w_k = weights[k]
                        # Condition for intruder (Eq. 12 in paper):
                        # ||w_r - w_r''||^2 + ||w_r'' - w_r'||^2 <= ||w_r - w_r'||^2
                        dist_ik_sq = euclidean(w_i, w_k) ** 2
                        dist_jk_sq = euclidean(w_j, w_k) ** 2
                        if dist_ik_sq + dist_jk_sq <= dist_ij_sq:
                            intruder_count += 1

    # Normalize by number of neurons to account for different map sizes (Page 7)
    zm = intruder_count / N
    return zm

def c_measure(weights, lattice_positions, p=2):
    """
    Implements the C-Measure (CM) as described in the paper (Page 8).
    Purpose: Quantifies overall topological alignment by computing a cost function that sums
    the product of distances in the input space (V) and map space (A). The minimal wiring
    scheme is used, modeling the cost of axonal wiring in cortical mappings.
    A lower CM indicates better topology preservation.

    Parameters:
    - weights: numpy array of shape (N, d), where N is the number of neurons and d is the
      dimensionality of the input space (weight vectors w_i).
    - lattice_positions: numpy array of shape (N, 2), the 2D coordinates of neurons in the map space.
    - p: exponent for the map space distance (d_A(i,j) = ||i-j||^p), default p=2 (Page 8).

    Returns:
    - cm: Normalized C-Measure (sum of distance products divided by number of neurons).

    Assumptions:
    - The paper's minimal wiring scheme uses a binary connectivity function in the input space
      (1 if neighbors, 0 otherwise). Since the paper does not specify how to determine neighbors
      in the input space, we assume Euclidean distances for d_V (common in SOM literature).
    - The map space distance uses the power-law form d_A(i,j) = ||i-j||^p, with p=2 as a reasonable default.
    """
    N = len(weights)
    cost = 0.0

    # Compute pairwise Euclidean distances in the input space (d_V(w_i, w_j))
    input_distances = cdist(weights, weights, metric='euclidean')

    # Compute pairwise distances in the map space (d_A(i,j) = ||i-j||^p)
    map_distances = cdist(lattice_positions, lattice_positions, metric='euclidean') ** p

    # Sum the product of distances for all pairs (i,j), excluding i=j
    for i in range(N):
        for j in range(N):
            if i != j:
                cost += input_distances[i, j] * map_distances[i, j]

    # Normalize by number of neurons to account for different map sizes (Page 8)
    cm = cost / N
    return cm

def topographic_product(weights, lattice_positions):
    """
    Implements the Topographic Product (TP) as described in the paper (Page 8-9).
    Purpose: Measures the average mismatch between neighborhood structures in the input and
    map spaces to assess if the map's dimensionality matches the input manifold.
    TP ≈ 0 indicates a good match; TP < 0 suggests the map is too low-dimensional, TP > 0 too high.

    Parameters:
    - weights: numpy array of shape (N, d), where N is the number of neurons and d is the
      dimensionality of the input space (weight vectors w_i).
    - lattice_positions: numpy array of shape (N, 2), the 2D coordinates of neurons in the map space.

    Returns:
    - tp: Topographic Product value.

    Assumptions:
    - The paper does not specify the maximum number of neighbors (k) to consider in the summation.
      We assume k ranges from 1 to N-1 (all possible neighbors), as implied by the formula.
    - Euclidean distances are used for both input and map spaces, as stated in the paper (Page 8).
    """
    N = len(weights)
    total_log = 0.0

    # Compute pairwise Euclidean distances in input and map spaces
    input_distances = cdist(weights, weights, metric='euclidean')
    map_distances = cdist(lattice_positions, lattice_positions, metric='euclidean')

    for j in range(N):
        # Get indices of neighbors sorted by distance in input and map spaces
        input_neighbors = np.argsort(input_distances[j])[1:]  # Exclude self (j)
        map_neighbors = np.argsort(map_distances[j])[1:]  # Exclude self (j)

        for k in range(1, N):  # k from 1 to N-1
            product = 1.0
            for i in range(1, k+1):
                n_v = input_neighbors[i-1]  # i-th closest in input space
                n_a = map_neighbors[i-1]  # i-th closest in map space
                # Compute ratio: (d_V(j, n_i^V) * d_A(j, n_i^A)) / (d_V(j, n_i^A) * d_A(j, n_i^V))
                num = input_distances[j, n_v] * map_distances[j, n_a]
                denom = input_distances[j, n_a] * map_distances[j, n_v]
                if denom > 0:  # Avoid division by zero
                    product *= num / denom
            # Take the k-th root and logarithm
            if product > 0:  # Avoid log of zero or negative
                total_log += np.log(product ** (1 / (2 * k)))

    # Normalize by (N^2 - N) as per the formula (Page 9)
    tp = total_log / (N**2 - N)
    return tp

def topographic_error(data, weights, lattice_positions, neighbors):
    """
    Implements the Topographic Error (TE) as described in the paper (Page 9).
    Purpose: Measures the proportion of input data points whose first and second best-matching
    units (BMUs) in the map space are not adjacent, evaluating the continuity of the mapping.
    A lower TE (TE = 0 for perfect preservation) indicates better topology preservation.

    Parameters:
    - data: numpy array of shape (K, d), where K is the number of data points and d is the
      dimensionality of the input space.
    - weights: numpy array of shape (N, d), where N is the number of neurons (weight vectors w_i).
    - lattice_positions: numpy array of shape (N, 2), the 2D coordinates of neurons in the map space.
    - neighbors: list of lists, where neighbors[i] contains indices of neurons adjacent to neuron i
      in the map space (based on the rectangular lattice).

    Returns:
    - te: Topographic Error value.

    Assumptions:
    - The paper assumes a rectangular lattice for the map space (Page 4), so neighbors are determined
      based on adjacent positions in a 2D grid.
    """
    K = len(data)
    error_count = 0

    # Compute distances from each data point to all weight vectors
    distances = cdist(data, weights, metric='euclidean')

    for i in range(K):
        # Find indices of the first and second BMUs (closest and second-closest neurons)
        sorted_indices = np.argsort(distances[i])
        bmu1 = sorted_indices[0]  # First BMU
        bmu2 = sorted_indices[1]  # Second BMU

        # Check if bmu2 is in the neighbor list of bmu1
        if bmu2 not in neighbors[bmu1]:
            error_count += 1

    # Compute TE as the proportion of non-adjacent BMUs (Eq. 17, Page 9)
    te = error_count / K
    return te

def get_neighbors(lattice_positions, grid_shape):
    """
    Helper function to compute neighbors for each neuron in a 2D rectangular lattice.
    Parameters:
    - lattice_positions: numpy array of shape (N, 2), the 2D coordinates of neurons.
    - grid_shape: tuple (rows, cols), the shape of the rectangular lattice.
    Returns:
    - neighbors: list of lists, where neighbors[i] contains indices of adjacent neurons.
    """
    rows, cols = grid_shape
    N = len(lattice_positions)
    neighbors = [[] for _ in range(N)]

    for i in range(N):
        x, y = lattice_positions[i]
        # Check four directions: up, down, left, right
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols:
                # Find index of the neighbor in lattice_positions
                neighbor_idx = np.where((lattice_positions[:, 0] == nx) & 
                                       (lattice_positions[:, 1] == ny))[0]
                if len(neighbor_idx) > 0:
                    neighbors[i].append(neighbor_idx[0])
    
    return neighbors

# Example usage
if __name__ == "__main__":
    # Example data (assumed for demonstration)
    # Assume a 3x3 grid (9 neurons), 2D input space (d=2)
    np.random.seed(42)
    weights = np.random.rand(9, 2)  # 9 neurons, 2D weight vectors
    lattice_positions = np.array([[i, j] for i in range(3) for j in range(3)])  # 3x3 grid
    grid_shape = (3, 3)
    data = np.random.rand(100, 2)  # 100 data points in 2D input space

    # Get neighbors for the lattice
    neighbors = get_neighbors(lattice_positions, grid_shape)

    # Compute each topology preservation measure
    zm = zrehen_measure(weights, lattice_positions, neighbors)
    cm = c_measure(weights, lattice_positions, p=2)
    tp = topographic_product(weights, lattice_positions)
    te = topographic_error(data, weights, lattice_positions, neighbors)

    print(f"Zrehen Measure (ZM): {zm:.4f}")
    print(f"C-Measure (CM): {cm:.4f}")
    print(f"Topographic Product (TP): {tp:.4f}")
    print(f"Topographic Error (TE): {te:.4f}")