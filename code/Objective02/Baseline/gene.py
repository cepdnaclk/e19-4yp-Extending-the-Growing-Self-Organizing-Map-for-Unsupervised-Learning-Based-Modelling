import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from minisom import MiniSom
from collections import Counter
import ast

# === Load Dataset ===
data_filename = "../example/data/GSE/GSE5281_merged_expression_metadata.csv"
df = pd.read_csv(data_filename)
print("Dataset shape:", df.shape)

# === Extract Expression Data ===
data = df.iloc[:, 3:]
print("Expression data shape:", data.shape)

output_folder = "output_baselines"
os.makedirs(output_folder, exist_ok=True)

# ================================
# === 1. PCA + Hierarchical  ===
# ================================

# --- Apply PCA ---
pca = PCA(n_components=10)
data_pca = pca.fit_transform(data)

# --- Hierarchical clustering on PCA-reduced data ---
Z_pca = linkage(data_pca, method='average')
ccc_pca, _ = cophenet(Z_pca, pdist(data_pca))
print(f"\n PCA+Hierarchical - Cophenetic Correlation Coefficient (CCC): {ccc_pca:.4f}")

# --- Loop through k values for silhouette score ---
for k in range(2, 11):
    cluster_labels = fcluster(Z_pca, t=k, criterion='maxclust')
    score = silhouette_score(data_pca, cluster_labels)
    print(f" PCA+Hierarchical - Silhouette Score (k={k}): {score:.4f}")

# --- Save dendrogram ---
plt.figure(figsize=(12, 6))
dendrogram(Z_pca, leaf_rotation=90)
plt.title("PCA + Hierarchical Clustering - Dendrogram")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "dendrogram_pca_hierarchical.png"))
plt.close()


# ================================
# === 2. SOM + Hierarchical    ===
# ================================

# --- Train SOM ---
som_dim = 5
som = MiniSom(som_dim, som_dim, data.shape[1], sigma=1.0, learning_rate=0.5)
som.random_weights_init(data.values)
som.train(data.values, 100)

# --- Collect SOM node weights ---
som_weights = np.array([som.get_weights()[i][j] for i in range(som_dim) for j in range(som_dim)])

# --- Hierarchical clustering on SOM node weights ---
Z_som = linkage(som_weights, method='average')
ccc_som, _ = cophenet(Z_som, pdist(som_weights))
print(f"\n SOM+Hierarchical - Cophenetic Correlation Coefficient (CCC): {ccc_som:.4f}")

# --- Silhouette scores ---
for k in range(2, 11):
    cluster_labels = fcluster(Z_som, t=k, criterion='maxclust')
    # Assign original data points to SOM nodes
    bmu_indices = np.array([som.winner(x) for x in data.values])
    # Flatten BMU to single integer for clustering comparison
    bmu_ids = np.array([i * som_dim + j for i, j in bmu_indices])
    # Map each unique BMU to a cluster label
    bmu_to_cluster = {}
    for idx, (i, j) in enumerate([(i, j) for i in range(som_dim) for j in range(som_dim)]):
        bmu_to_cluster[i * som_dim + j] = cluster_labels[idx]
    sample_cluster_labels = [bmu_to_cluster[bmu_id] for bmu_id in bmu_ids]
    score = silhouette_score(data.values, sample_cluster_labels)
    print(f" SOM+Hierarchical - Silhouette Score (k={k}): {score:.4f}")

# --- Save dendrogram ---
plt.figure(figsize=(12, 6))
dendrogram(Z_som, leaf_rotation=90)
plt.title("SOM + Hierarchical Clustering - Dendrogram")
plt.xlabel("SOM Node Index")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "dendrogram_som_hierarchical.png"))
plt.close()
