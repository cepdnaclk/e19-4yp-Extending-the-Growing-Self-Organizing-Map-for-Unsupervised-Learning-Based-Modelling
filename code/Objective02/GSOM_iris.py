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

    # Create necessary directories early
    os.makedirs("output", exist_ok=True)
    os.makedirs("GSOM_maps", exist_ok=True)
    os.makedirs("node_clusters", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("dendrograms", exist_ok=True)
    os.makedirs("cluster_mapping", exist_ok=True)

    # Step 1: Load Iris dataset from CSV
    df = pd.read_csv("example/data/iris.csv")
    df["Id"] = df.index.astype(str)
    if "species" in df.columns:
        df.rename(columns={"species": "Species"}, inplace=True)
    features = df.drop(columns=["Id", "Species"])
    full_input = pd.concat([features, df[["Id", "Species"]]], axis=1)

    # Step 2: Train GSOM
    gsom = GSOM(spred_factor=0.84, dimensions=features.shape[1], distance='euclidean', max_radius=4)
    gsom.fit(features.to_numpy(), training_iterations=100, smooth_iterations=50)

    # Step 3: Predict and save
    output = gsom.predict(full_input, index_col="Id", label_col="Species")
    output.to_csv("output/output_iris.csv", index=False)

    # Step 4: Plot GSOM node map
    plt.figure(figsize=(10, 8))
    plt.scatter(output["x"], output["y"], c=output["hit_count"], cmap='viridis', s=100, edgecolors='k')
    plt.colorbar(label='Hit Count')
    plt.title("GSOM Node Map (Hit Count Intensity)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("GSOM_maps/gsom_node_map_iris.png")
    plt.close()

    print("🗺️ GSOM Node Map saved as 'GSOM_maps/gsom_node_map_iris.png'")

    # Step 5: Active nodes and filter
    df_out = pd.read_csv("output/output_iris.csv")
    active_nodes = df_out[df_out["hit_count"] > 0].copy()
    active_nodes["Species"] = active_nodes["Species"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else x
    )
    active_nodes = active_nodes[active_nodes["hit_count"] >= 2].copy()

    # Step 6: Hierarchical clustering using GSOM node weights
    try:
        node_indices = active_nodes["output"].to_numpy()
        X = gsom.node_list[node_indices]  # Use weights
    except Exception as e:
        print("❌ Failed to retrieve weights from node_list:", e)
        exit(1)

    best_coph, best_method, best_Z = 0, None, None
    for method in ['single', 'complete', 'average', 'ward']:
        Z_temp = linkage(X, method=method)
        coph_corr, _ = cophenet(Z_temp, pdist(X))
        print(f"{method.capitalize():<10} CCC: {coph_corr:.4f}")
        if coph_corr > best_coph:
            best_coph, best_method, best_Z = coph_corr, method, Z_temp
    Z = best_Z
    print(f"\n✅ Best linkage method: {best_method} with CCC = {best_coph:.4f}")

    # Step 7: Cluster GSOM nodes
    active_nodes["cluster"] = fcluster(Z, 3, criterion='maxclust')
    labels_nodes = active_nodes["cluster"].to_numpy()
    sil_node_score = silhouette_score(X, labels_nodes) if len(set(labels_nodes)) > 1 else None

    # Step 8: Sample metrics
    node_clusters_path = "node_clusters/gsom_node_clusters_iris.csv"
    active_nodes.to_csv(node_clusters_path, index=False)

    data_points = pd.read_csv("output/output_iris.csv")
    node_clusters = pd.read_csv(node_clusters_path)
    merged = pd.merge(data_points, node_clusters[["output", "cluster"]], on="output", how="left")

    merged_valid = merged.dropna(subset=["cluster"]).copy()
    X_samples = merged_valid[["x", "y"]].to_numpy()
    labels_samples = merged_valid["cluster"].astype(int).to_numpy()

    species_mapping = {name: idx for idx, name in enumerate(sorted(df["Species"].unique()))}
    df["label_numeric"] = df["Species"].map(species_mapping)
    clean_ids = merged_valid["Id"].apply(lambda x: ast.literal_eval(x)[0] if isinstance(x, str) and x.startswith("[") else str(x))
    true_labels = df[df["Id"].isin(clean_ids)]["label_numeric"].to_numpy()

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

    # Step 9: Save metrics
    with open("results/silhouette_scores_iris.txt", "w") as f:
        f.write("GSOM + Hierarchical Clustering Evaluation (Iris)\n")
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

    # Step 10: Label summary and plots
    def formatted_label(row):
        counter = Counter(row["Species"])
        dom_label, dom_count = counter.most_common(1)[0]
        total = sum(counter.values())
        percent = (dom_count / total) * 100 if total > 0 else 0
        return f"Cluster {row['cluster']} | {dom_label.capitalize()} ({percent:.1f}%) | N={total}"

    active_nodes["label_summary"] = active_nodes.apply(formatted_label, axis=1)
    active_nodes.to_csv(node_clusters_path, index=False)

    # Label-based GSOM map
    plt.figure(figsize=(10, 8))
    color_map = {"setosa": "red", "versicolor": "green", "virginica": "blue"}
    dominant_labels = active_nodes["label_summary"].apply(
        lambda x: "setosa" if "setosa" in x else ("versicolor" if "versicolor" in x else "virginica")
    )
    colors = dominant_labels.map(color_map)
    plt.scatter(active_nodes["x"], active_nodes["y"], c=colors, s=120, edgecolors='black')
    for _, row in active_nodes.iterrows():
        plt.text(row["x"], row["y"], str(row["cluster"]), fontsize=8, ha='center', va='center', color='white')
    plt.title("GSOM Node Label Map (Dominant Class per Cluster)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("GSOM_maps/gsom_node_label_map_iris.png")
    plt.close()

    # Dendrogram
    plt.figure(figsize=(18, 8))
    dendrogram(
        Z,
        labels=active_nodes["label_summary"].values,
        leaf_rotation=90,
        leaf_font_size=10
    )
    plt.title(f"Hierarchical Clustering on GSOM Nodes (Linkage: {best_method})")
    plt.xlabel("Clustered GSOM Nodes (Iris Classes)")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig("dendrograms/hierarchical_clustering_iris_annotated.png")
    plt.close()

    # Save sample-cluster mapping
    merged.to_csv("cluster_mapping/iris_sample_cluster_mapping.csv", index=False)
    print("✅ All files saved in respective folders for Iris dataset.")
