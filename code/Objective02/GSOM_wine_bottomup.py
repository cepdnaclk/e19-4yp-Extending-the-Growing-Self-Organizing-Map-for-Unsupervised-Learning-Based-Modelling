import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from sklearn.metrics import silhouette_score
from collections import Counter
import ast

from GSOM import GSOM  # Ensure GSOM.py is in the same directory

if __name__ == '__main__':
    np.random.seed(1)

    # === Output Folder ===
    output_folder = "../output_wine_gsom"
    os.makedirs(output_folder, exist_ok=True)

    # === Load Wine Dataset ===
    df = pd.read_csv("example/data/wine.data", header=None)
    df.columns = ['Class'] + [f'Feature_{i}' for i in range(1, 14)]
    df['Sample_ID'] = [f"Wine_{i}" for i in range(len(df))]
    df = df[['Sample_ID', 'Class'] + [f'Feature_{i}' for i in range(1, 14)]]

    # === Extract Features ===
    data_training = df.iloc[:, 2:]
    print("Training data shape:", data_training.shape)

    # === Train GSOM ===
    gsom = GSOM(spred_factor=0.82, dimensions=data_training.shape[1], max_radius=4, initial_node_size=1000)
    gsom.fit(data_training.to_numpy(), training_iterations=100, smooth_iterations=50)

    # === Predict Mapping and Save ===
    output = gsom.predict(df, index_col="Sample_ID", label_col="Class")
    output.to_csv(os.path.join(output_folder, "output.csv"), index=False)
    print("GSOM training completed.")
    print("Output shape:", output.shape)
    print("Node Count:", gsom.node_count)

    # === GSOM Node Clustering ===
    node_embeddings = gsom.node_list[:gsom.node_count]
    node_coords = gsom.node_coordinate[:gsom.node_count]
    Z = linkage(node_embeddings, method='average')

    # === CCC Calculation ===
    dist_matrix = pdist(node_embeddings)
    coph_corr, _ = cophenet(Z, dist_matrix)
    print(f"\n✅ Cophenetic Correlation Coefficient (CCC): {coph_corr:.4f}")

    # === Save GSOM Node Dendrogram ===
    plt.figure(figsize=(12, 6))
    dendrogram(Z, labels=[str(i) for i in range(gsom.node_count)], leaf_rotation=90)
    plt.title("Dendrogram: Hierarchical Clustering of GSOM Nodes")
    plt.xlabel("Node Index")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "dendrogram_gsom_nodes.png"))
    plt.show()

    # === Load Mapped Output ===
    output_df = pd.read_csv(os.path.join(output_folder, "output.csv"))
    output_df["output"] = output_df["output"].astype(int)

    # === Loop over Cluster Numbers ===
    for k in range(2, 11):
        print(f"\n Generating summary for k = {k} clusters...")
        cluster_labels = fcluster(Z, t=k, criterion='maxclust')

        if len(set(cluster_labels)) > 1:
            sil_score = silhouette_score(node_embeddings, cluster_labels)
            print(f"✅ Silhouette Score for k={k}: {sil_score:.4f}")
        else:
            print(f"⚠️ Cannot compute silhouette score for k={k}: only one cluster")

        output_df["Cluster"] = output_df["output"].map(
            lambda x: cluster_labels[int(x)] if int(x) < len(cluster_labels) else -1
        )

        cluster_summary = []
        for cluster_id in sorted(output_df["Cluster"].unique()):
            cluster_nodes = output_df[output_df["Cluster"] == cluster_id]
            labels = cluster_nodes['Class'].tolist()
            label_counts = Counter(labels)
            if label_counts:
                dominant_label, dominant_count = label_counts.most_common(1)[0]
                purity = dominant_count / len(labels) * 100
            else:
                dominant_label, purity = "N/A", 0.0
            cluster_summary.append({
                "Cluster": cluster_id,
                "Node IDs": list(cluster_nodes['output'].unique()),
                "Dominant Label": dominant_label,
                "% Purity": f"{purity:.1f}%",
                "Total Count": len(labels)
            })

        summary_df = pd.DataFrame(cluster_summary)
        summary_file = f"gsom_cluster_summary_k{k}.csv"
        summary_df.to_csv(os.path.join(output_folder, summary_file), index=False)
        print(f" Saved: {summary_file}")

        # === GSOM Map Visualization (Optional, k=4) ===
        if k == 4:
            plt.figure(figsize=(10, 10))
            for i in range(1, k + 1):
                idxs = np.where(cluster_labels == i)[0]
                plt.scatter(node_coords[idxs, 0], node_coords[idxs, 1], label=f"Cluster {i}", s=30)
            plt.title("GSOM Map: Nodes Colored by Hierarchical Clusters (k=4)")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_folder, "gsom_hierarchical_clusters_k4.png"))
            plt.show()

    # === Annotated Dendrogram with ≥90% Purity Highlights ===
    print("\n Generating annotated dendrogram with high-purity clusters marked...")
    summary_df = pd.read_csv(os.path.join(output_folder, "gsom_cluster_summary_k10.csv"))

    node_label_map = {}
    for _, row in summary_df.iterrows():
        node_ids = row["Node IDs"].strip("[]").split(",")
        purity_val = float(row["% Purity"].replace('%', ''))
        for node_id in node_ids:
            node_id = int(node_id.strip())
            if purity_val >= 90.0:
                label = f"\u2b50 Node {node_id}\n{row['Dominant Label']}\n{row['% Purity']}"
            else:
                label = f"Node {node_id}\n{row['Dominant Label']}\n{row['% Purity']}"
            node_label_map[node_id] = label

    custom_labels = [node_label_map.get(i, f"Node {i}\n(no data)") for i in range(gsom.node_count)]

    plt.figure(figsize=(14, 7))
    dendrogram(Z, labels=custom_labels, leaf_rotation=90)
    plt.title("Annotated Dendrogram: GSOM Node Clustering (\u2b50 = ≥90% Purity)")
    plt.xlabel("GSOM Nodes with Cluster Info")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "dendrogram_gsom_nodes_annotated_highlighted.png"))
    plt.show()
