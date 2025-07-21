import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from sklearn.metrics import silhouette_score
from collections import Counter
import ast

from GSOM import GSOM

if __name__ == '__main__':
    np.random.seed(1)

    # === Output Folder ===
    output_folder = "output_gene_bottomup"
    os.makedirs(output_folder, exist_ok=True)

    # === Load Dataset ===
    data_filename = "example/data/GSE/GSE5281_merged_expression_metadata.csv"
    df = pd.read_csv(data_filename)
    print("Dataset shape:", df.shape)

    # === Extract Expression Data ===
    data_training = df.iloc[:, 3:]
    print("Training data shape:", data_training.shape)

    # === Train GSOM ===
    gsom = GSOM(spred_factor=0.83, dimensions=data_training.shape[1], max_radius=4, initial_node_size=1000)
    gsom.fit(data_training.to_numpy(), training_iterations=100, smooth_iterations=50)

    # === Predict and Save Node Mappings ===
    predict_input = df.loc[:, ['Sample_ID', 'Disease_State'] + list(data_training.columns)]
    output = gsom.predict(predict_input, index_col="Sample_ID", label_col="Disease_State")
    output.to_csv(os.path.join(output_folder, "output.csv"), index=False)

    print("GSOM training completed.")
    print("Output shape:", output.shape)
    print("Node Count:", gsom.node_count)

    # === Hierarchical Clustering ===
    node_embeddings = gsom.node_list[:gsom.node_count]
    node_coords = gsom.node_coordinate[:gsom.node_count]
    Z = linkage(node_embeddings, method='average')

    # === CCC ===
    dist_matrix = pdist(node_embeddings)
    coph_corr, _ = cophenet(Z, dist_matrix)
    print(f"\n✅ Cophenetic Correlation Coefficient (CCC): {coph_corr:.4f}")

    # === Save Original Dendrogram ===
    plt.figure(figsize=(12, 6))
    dendrogram(Z, labels=[str(i) for i in range(gsom.node_count)], leaf_rotation=90)
    plt.title("Dendrogram: Hierarchical Clustering of GSOM Nodes")
    plt.xlabel("Node Index")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "dendrogram_gsom_nodes.png"))
    plt.show()

    # === Load Output CSV ===
    output_df = pd.read_csv(os.path.join(output_folder, "output.csv"))
    output_df["output"] = output_df["output"].astype(int)

    # === Loop over multiple k values ===
    for k in range(2, 11):
        print(f"\n Generating summary for k = {k} clusters...")
        cluster_labels = fcluster(Z, t=k, criterion='maxclust')

        # === Silhouette Score ===
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
            labels = [
                label.strip()
                for label_list in cluster_nodes['Disease_State']
                for label in ast.literal_eval(label_list)
            ]
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

        # === Optional: save GSOM map for one value (k=4) ===
        if k == 4:
            plt.figure(figsize=(10, 10))
            for i in range(1, k + 1):
                idxs = np.where(cluster_labels == i)[0]
                plt.scatter(node_coords[idxs, 0], node_coords[idxs, 1], label=f"Cluster {i}", s=30)
            plt.title(f"GSOM Map: Nodes Colored by Hierarchical Clusters (k={k})")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_folder, f"gsom_hierarchical_clusters_k{k}.png"))
            plt.show()

    # === Annotated Dendrogram for k=10 with for >90% Purity ===
    print("\n Generating annotated dendrogram with high-purity clusters marked...")
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
                label = f"\u2b50 Node {node_id}\n{row['Dominant Label']}\n{row['% Purity']}"
            else:
                label = f"Node {node_id}\n{row['Dominant Label']}\n{row['% Purity']}"
            node_label_map[int(node_id)] = label

    custom_labels = []
    for i in range(gsom.node_count):
        if i in node_label_map:
            custom_labels.append(node_label_map[i])
        else:
            custom_labels.append(f"Node {i}\n(no data)")

    plt.figure(figsize=(14, 7))
    dendrogram(Z, labels=custom_labels, leaf_rotation=90)
    plt.title("Annotated Dendrogram: GSOM Node Clustering (\u2b50 = ≥90% Purity)")
    plt.xlabel("GSOM Nodes with Cluster Info")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "dendrogram_gsom_nodes_annotated_highlighted.png"))
    plt.show()


