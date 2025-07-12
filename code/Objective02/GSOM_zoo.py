if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
    from scipy.spatial.distance import pdist
    from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
    import pandas as pd
    import numpy as np
    from collections import Counter
    import os
    import ast
    from GSOM import GSOM

    np.random.seed(1)

    # Setup
    os.makedirs("output", exist_ok=True)
    os.makedirs("GSOM_maps", exist_ok=True)
    os.makedirs("node_clusters", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("dendrograms", exist_ok=True)
    os.makedirs("cluster_mapping", exist_ok=True)

    # Load Zoo data
    df = pd.read_csv("example/data/zoo.txt")
    df["Id"] = df.index.astype(str)
    df.rename(columns={"label": "Species"}, inplace=True)
    features = df.drop(columns=["Name", "Id", "Species"])
    full_input = pd.concat([features, df[["Id", "Species"]]], axis=1)

    # GSOM training
    gsom = GSOM(spred_factor=0.84, dimensions=features.shape[1], distance='euclidean', max_radius=4)
    gsom.fit(features.to_numpy(), training_iterations=100, smooth_iterations=50)
    output = gsom.predict(full_input, index_col="Id", label_col="Species")
    output.to_csv("output/output_zoo.csv", index=False)

    # Plot GSOM
    plt.figure(figsize=(10, 8))
    plt.scatter(output["x"], output["y"], c=output["hit_count"], cmap='viridis', s=100, edgecolors='k')
    plt.colorbar(label='Hit Count')
    plt.title("GSOM Node Map (Zoo Dataset)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("GSOM_maps/gsom_node_map_zoo.png")
    plt.close()
    print("🗺️ GSOM Node Map saved as 'GSOM_maps/gsom_node_map_zoo.png'")

    # Active nodes
    df_out = pd.read_csv("output/output_zoo.csv")
    active_nodes = df_out[df_out["hit_count"] > 0].copy()
    active_nodes["Species"] = active_nodes["Species"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else x
    )
    active_nodes = active_nodes[active_nodes["hit_count"] >= 2].copy()

    # Get node weights
    try:
        node_indices = active_nodes["output"].to_numpy()
        X = gsom.node_list[node_indices]  # node weights
    except Exception as e:
        print("❌ Failed to retrieve weights from node_list:", e)
        exit(1)

    # Hierarchical clustering
    best_coph, best_method, best_Z = 0, None, None
    for method in ['single', 'complete', 'average', 'ward']:
        Z_temp = linkage(X, method=method)
        coph_corr, _ = cophenet(Z_temp, pdist(X))
        print(f"{method.capitalize():<10} CCC: {coph_corr:.4f}")
        if coph_corr > best_coph:
            best_coph, best_method, best_Z = coph_corr, method, Z_temp
    Z = best_Z
    print(f"\n✅ Best linkage method: {best_method} with CCC = {best_coph:.4f}")

    # Cluster nodes
    active_nodes["cluster"] = fcluster(Z, 7, criterion='maxclust')  # 7 classes in zoo
    labels_nodes = active_nodes["cluster"].to_numpy()
    sil_node_score = silhouette_score(X, labels_nodes) if len(set(labels_nodes)) > 1 else None

    # Merge clusters to samples
    node_clusters_path = "node_clusters/gsom_node_clusters_zoo.csv"
    active_nodes.to_csv(node_clusters_path, index=False)
    merged = pd.merge(df_out, active_nodes[["output", "cluster"]], on="output", how="left")
    merged_valid = merged.dropna(subset=["cluster"]).copy()
    X_samples = merged_valid[["x", "y"]].to_numpy()
    labels_samples = merged_valid["cluster"].astype(int).to_numpy()
    true_labels = df.loc[df["Id"].isin(merged_valid["Id"])]["Species"].to_numpy()

    # Sample-level metrics
    sil_sample_score = ari_score = nmi_score = None
    if len(set(labels_samples)) > 1 and len(true_labels) == len(labels_samples):
        sil_sample_score = silhouette_score(X_samples, labels_samples)
        ari_score = adjusted_rand_score(true_labels, labels_samples)
        nmi_score = normalized_mutual_info_score(true_labels, labels_samples)
        print(f"📐 Silhouette Score (Samples): {sil_sample_score:.4f}")
        print(f"🧮 Adjusted Rand Index (ARI): {ari_score:.4f}")
        print(f"🧠 Normalized Mutual Information (NMI): {nmi_score:.4f}")
    else:
        print("⚠️ Not enough sample clusters to compute clustering metrics.")

    # Save metrics
    with open("results/silhouette_scores_zoo.txt", "w") as f:
        f.write("GSOM + Hierarchical Clustering Evaluation (Zoo)\n")
        f.write("=========================================\n")
        f.write(f"Best Linkage Method: {best_method}\n")
        f.write(f"Cophenetic Correlation Coefficient (CCC): {best_coph:.4f}\n\n")
        f.write(f"Silhouette Score (GSOM Nodes): {sil_node_score:.4f}\n" if sil_node_score else "Silhouette Score (GSOM Nodes): Not enough clusters.\n")
        f.write("\nSample-Level Evaluation:\n")
        if sil_sample_score:
            f.write(f"Silhouette Score (Samples): {sil_sample_score:.4f}\n")
            f.write(f"Adjusted Rand Index (ARI): {ari_score:.4f}\n")
            f.write(f"Normalized Mutual Information (NMI): {nmi_score:.4f}\n")
        else:
            f.write("Silhouette / ARI / NMI (Samples): Not enough clusters.\n")

    print("✅ All files saved in respective folders for Zoo dataset.")
