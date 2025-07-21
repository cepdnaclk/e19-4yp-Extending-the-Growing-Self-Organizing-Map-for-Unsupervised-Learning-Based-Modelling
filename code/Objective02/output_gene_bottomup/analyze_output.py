import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

# === Helper: Clean and count diseases ===
def count_disease_labels(label_list):
    clean = [x.replace("\xa0", "").strip().lower() for x in label_list]
    count_ad = sum('alzheimer' in x for x in clean)
    count_normal = sum('normal' in x for x in clean)
    return count_ad, count_normal

# === Load output.csv ===
output = pd.read_csv("output.csv")
output["output"] = output["output"].astype(int)

# === Safe eval for disease lists ===
def safe_parse_disease_state(val):
    if isinstance(val, str):
        try:
            return eval(val)
        except:
            return []
    return val

output["Disease_State"] = output["Disease_State"].apply(safe_parse_disease_state)

# === Get unique node list and build embedding matrix ===
unique_nodes = sorted(output["output"].unique())
node_embeddings = []

# You must load your GSOM node embeddings here
# This is dummy for demo; replace with actual node list
# Example: load from numpy if saved from training
node_embeddings = np.load("output_gene_bottomup/node_embeddings.npy")  # Must be shape (17, dim)

# === Hierarchical clustering ===
Z = linkage(node_embeddings, method='average')

# === Cluster for k=2 to k=17 and count disease types ===
print("==== Disease Composition for Each k ====\n")
for k in range(2, len(unique_nodes)+1):
    cluster_labels = fcluster(Z, t=k, criterion='maxclust')
    
    cluster_stats = {}
    for idx, cluster_id in enumerate(cluster_labels):
        node_id = idx  # Assuming embeddings are in order of node_id
        cluster_stats.setdefault(cluster_id, {"AD": 0, "Normal": 0})

        # Find matching rows in output
        node_rows = output[output["output"] == node_id]
        for _, row in node_rows.iterrows():
            ad_count, normal_count = count_disease_labels(row["Disease_State"])
            cluster_stats[cluster_id]["AD"] += ad_count
            cluster_stats[cluster_id]["Normal"] += normal_count

    print(f"\n--- k = {k} ---")
    for cluster_id, counts in cluster_stats.items():
        ad, normal = counts["AD"], counts["Normal"]
        print(f"Cluster {cluster_id}: AD = {ad}, Normal = {normal}, Total = {ad + normal}")
