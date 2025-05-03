import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, euclidean
import os

def load_csv(path):
    return pd.read_csv(path).values

def get_neighbors(lattice_positions):
    """Find neighbors based on 4-connected grid logic."""
    neighbors = [[] for _ in range(len(lattice_positions))]
    pos_dict = {tuple(pos): idx for idx, pos in enumerate(lattice_positions)}
    for idx, (x, y) in enumerate(lattice_positions):
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor = (x + dx, y + dy)
            if neighbor in pos_dict:
                neighbors[idx].append(pos_dict[neighbor])
    return neighbors

def zrehen_measure(weights, positions, neighbors):
    intruders = 0
    N = len(weights)
    for i in range(N):
        for j in neighbors[i]:
            if j > i:
                dij = euclidean(weights[i], weights[j]) ** 2
                for k in range(N):
                    if k != i and k != j:
                        dik = euclidean(weights[i], weights[k]) ** 2
                        djk = euclidean(weights[j], weights[k]) ** 2
                        if dik + djk <= dij:
                            intruders += 1
    return intruders / N

def c_measure(weights, positions, p=2):
    N = len(weights)
    input_dists = cdist(weights, weights, 'euclidean')
    map_dists = cdist(positions, positions, 'euclidean') ** p
    cost = np.sum(input_dists * map_dists) / (N * N)
    return cost

def topographic_product(weights, positions, K=10):
    """Limited neighbor version for large maps."""
    N = len(weights)
    input_dists = cdist(weights, weights, 'euclidean')
    map_dists = cdist(positions, positions, 'euclidean')
    tp_total = 0
    for j in range(N):
        i_neighbors = np.argsort(input_dists[j])[1:K+1]
        m_neighbors = np.argsort(map_dists[j])[1:K+1]
        for k in range(K):
            nv = i_neighbors[k]
            na = m_neighbors[k]
            num = input_dists[j, nv] * map_dists[j, na]
            denom = input_dists[j, na] * map_dists[j, nv]
            if denom > 0:
                tp_total += np.log(num / denom)
    return tp_total / (N * K)

def topographic_error(data, weights, positions, neighbors):
    errors = 0
    dists = cdist(data, weights, 'euclidean')
    for i in range(len(data)):
        bmu1, bmu2 = np.argsort(dists[i])[:2]
        if bmu2 not in neighbors[bmu1]:
            errors += 1
    return errors / len(data)

# === File paths ===
som_weights = load_csv("outputs/som_circle_node_weights.csv")
som_positions = load_csv("outputs/som_circle_node_coords.csv")
gsom_weights = load_csv("outputs/gsom_circle_node_weights.csv")
gsom_positions = load_csv("outputs/gsom_circle_node_coordinates.csv")
input_data = load_csv("data/shapes/circle_radius100_652_points.csv")[:, :2]  # Change as needed

# === Preprocessing ===
neighbors_som = get_neighbors(som_positions.astype(int))
neighbors_gsom = get_neighbors(gsom_positions.astype(int))

# === SOM Measures ===
zm_som = zrehen_measure(som_weights, som_positions, neighbors_som)
cm_som = c_measure(som_weights, som_positions)
tp_som = topographic_product(som_weights, som_positions)
te_som = topographic_error(input_data, som_weights, som_positions, neighbors_som)

# === GSOM Measures ===
zm_gsom = zrehen_measure(gsom_weights, gsom_positions, neighbors_gsom)
cm_gsom = c_measure(gsom_weights, gsom_positions)
tp_gsom = topographic_product(gsom_weights, gsom_positions)
te_gsom = topographic_error(input_data, gsom_weights, gsom_positions, neighbors_gsom)

# === Display Results ===
print("\n--- SOM ---")
print(f"ZM: {zm_som:.4f}, CM: {cm_som:.4f}, TP: {tp_som:.4f}, TE: {te_som:.4f}")
print("\n--- GSOM ---")
print(f"ZM: {zm_gsom:.4f}, CM: {cm_gsom:.4f}, TP: {tp_gsom:.4f}, TE: {te_gsom:.4f}")
