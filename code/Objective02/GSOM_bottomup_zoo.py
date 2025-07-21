import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from sklearn.metrics import silhouette_score
from collections import Counter
import ast

from GSOM import GSOM

if __name__ == '__main__':
    np.random.seed(1)

    # === Output Folder ===
    output_folder = "output_zoo_bottomup"
    os.makedirs(output_folder, exist_ok=True)

    # === Load Dataset ===
    data_filename = "example/data/zoo.txt"
    df = pd.read_csv(data_filename)
    print("Dataset shape:", df.shape)

    # === Extract Training Data ===
    data_training = df.iloc[:, 1:-1]  # All columns except Name and label
    print("Training data shape:", data_training.shape)

    # === Train GSOM ===
    gsom = GSOM(spred_factor=0.83, dimensions=data_training.shape[1], max_radius=4, initial_node_size=1000)
    gsom.fit(data_training.to_numpy(), training_iterations=100, smooth_iterations=50)

    # === Predict & Save Mappings ===
    predict_input = df[['Name', 'label'] + list(data_training.columns)]
    output = gsom.predict(predict_input, index_col="Name", label_col="label")
    output.to_csv(os.path.join(output_folder, "output.csv"), index=False)

    print("✅ GSOM training completed.")
    print("Output shape:", output.shape)
    print("Node Count:", gsom.node_count)

    # === Load Output CSV ===
    output_df = pd.read_csv(os.path.join(output_folder, "output.csv"))
    output_df["output"] = output_df["output"].astype(int)

    # === Filter Only Nodes with Mappings ===
    mapped_node_ids = sorted(output_df["output"].unique())
    print(f"Number of mapped GSOM nodes: {len(mapped_node_ids)}")

    filtered_node_embeddings = [gsom.node_list[i] for i in mapped_node_ids]
    filtered_node_coords = np.array([gsom.node_coordinate[i] for i in mapped_node_ids])

    # === Hierarchical Clustering on Filtered Nodes ===
    Z = linkage(filtered_node_embeddings, method='average')
    dist_matrix = pdist(filtered_node_embeddings)
    coph_corr, _ = cophenet(Z, dist_matrix)
    print(f"\n✅ Cophenetic Correlation Coefficient (CCC): {coph_corr:.4f}")

    # === Save Dendrogram for Filtered Nodes ===
    plt.figure(figsize=(12, 6))
    dendrogram(Z, labels=[str(i) for i in mapped_node_ids], leaf_rotation=90)
    plt.title("Dendrogram: Hierarchical Clustering of Mapped GSOM Nodes")
    plt.xlabel("GSOM Node Index")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "dendrogram_gsom_nodes_mapped_only.png"))
    plt.show()

    # === Loop over k (2 to 10) for clustering ===
    for k in range(2, 11):
        print(f"\nGenerating summary for k = {k} clusters...")
        cluster_labels = fcluster(Z, t=k, criterion='maxclust')

        if len(set(cluster_labels)) > 1:
            sil_score = silhouette_score(filtered_node_embeddings, cluster_labels)
            print(f"✅ Silhouette Score for k={k}: {sil_score:.4f}")
        else:
            print(f"⚠️ Only one cluster for k={k}, skipping silhouette score.")

        node_id_to_cluster = dict(zip(mapped_node_ids, cluster_labels))
        output_df["Cluster"] = output_df["output"].map(lambda x: node_id_to_cluster.get(x, -1))

        cluster_summary = []
        for cluster_id in sorted(output_df["Cluster"].unique()):
            cluster_nodes = output_df[output_df["Cluster"] == cluster_id]
            labels = cluster_nodes['label'].astype(str).tolist()
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
        print(f"✅ Saved: {summary_file}")

        # === Optional: Save GSOM Node Map for k=4 ===
        if k == 4:
            plt.figure(figsize=(10, 10))
            for i in range(1, k + 1):
                idxs = [mapped_node_ids[j] for j in range(len(cluster_labels)) if cluster_labels[j] == i]
                coords = np.array([gsom.node_coordinate[i] for i in idxs])
                plt.scatter(coords[:, 0], coords[:, 1], label=f"Cluster {i}", s=30)
            plt.title(f"GSOM Map: Nodes Colored by Clusters (k={k})")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_folder, f"gsom_hierarchical_clusters_k{k}.png"))
            plt.show()

    # === Annotated Dendrogram for k=10 ===
    print("\nGenerating annotated dendrogram with high-purity clusters marked...")
    summary_df = pd.read_csv(os.path.join(output_folder, "gsom_cluster_summary_k10.csv"))

    node_label_map = {}
    for _, row in summary_df.iterrows():
        node_ids = ast.literal_eval(row["Node IDs"])
        purity_str = row["% Purity"].replace('%', '')
        try:
            purity_val = float(purity_str)
        except:
            purity_val = 0.0

        for node_id in node_ids:
            if purity_val >= 90.0:
                label = f"\u2b50 Node {node_id}\nLabel: {row['Dominant Label']}\n{row['% Purity']}"
            else:
                label = f"Node {node_id}\nLabel: {row['Dominant Label']}\n{row['% Purity']}"
            node_label_map[int(node_id)] = label

    custom_labels = [node_label_map.get(i, f"Node {i}\n(no data)") for i in mapped_node_ids]

    plt.figure(figsize=(14, 7))
    dendrogram(Z, labels=custom_labels, leaf_rotation=90)
    plt.title("Annotated Dendrogram (⭐ = ≥90% Purity)")
    plt.xlabel("Mapped GSOM Nodes")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "dendrogram_gsom_nodes_annotated_highlighted.png"))
    plt.show()
