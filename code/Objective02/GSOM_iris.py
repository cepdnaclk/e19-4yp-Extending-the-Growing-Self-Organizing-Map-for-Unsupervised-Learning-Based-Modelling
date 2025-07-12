if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
    from scipy.spatial.distance import pdist
    from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
    import pandas as pd
    import numpy as np
    from collections import Counter
    import ast
    import os

    from GSOM import GSOM  # Assumes your GSOM implementation is in gsom.py

    np.random.seed(1)

    # Step 1: Load the Iris dataset
    df = pd.read_csv("example/data/Iris.csv")
    print("Dataset shape:", df.shape)

    # Step 2: Assign ID and rename Species for simplicity
    df["Id"] = df["Id"].astype(str)
    df["Species"] = df["Species"].astype(str)

    # Step 3: Prepare inputs
    features = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
    df["label_numeric"] = df["Species"].astype("category").cat.codes
    full_input = pd.concat([features, df[["Id", "Species"]]], axis=1)

    # Step 4: Train GSOM
    gsom = GSOM(spread_factor=0.83, dimensions=features.shape[1], distance='euclidean', max_radius=4)
    gsom.fit(features.to_numpy(), training_iterations=100, smooth_iterations=50)

    # Step 5: Predict & save
    output = gsom.predict(full_input, index_col="Id", label_col="Species")
    output.to_csv("output_iris.csv", index=False)

    # Step 6: Active GSOM nodes
    df_out = pd.read_csv("output_iris.csv")
    active_nodes = df_out[df_out["hit_count"] > 0].copy()
    active_nodes["Species"] = active_nodes["Species"].apply(ast.literal_eval)

    active_nodes = active_nodes[active_nodes["hit_count"] >= 2].copy()

    # Step 7: Hierarchical clustering
    X = active_nodes[["x", "y"]].to_numpy()
    print("\n📊 Cophenetic Correlation Coefficients for Linkage Methods:")
    best_coph, best_method, best_Z = 0, None, None
    for method in ['single', 'complete', 'average', 'ward']:
        Z_temp = linkage(X, method=method)
        coph_corr, _ = cophenet(Z_temp, pdist(X))
        print(f"{method.capitalize():<10} CCC: {coph_corr:.4f}")
        if coph_corr > best_coph:
            best_coph, best_method, best_Z = coph_corr, method, Z_temp

    Z = best_Z
    print(f"\n✅ Best linkage method: {best_method} with CCC = {best_coph:.4f}")

    # Step 8: Assign clusters
    active_nodes["cluster"] = fcluster(Z, 3, criterion='maxclust')

    # Step 9: Silhouette Scores (Nodes & Samples)
    os.makedirs("results", exist_ok=True)

    X_nodes = active_nodes[["x", "y"]].to_numpy()
    labels_nodes = active_nodes["cluster"].to_numpy()

    sil_node_score = silhouette_score(X_nodes, labels_nodes) if len(set(labels_nodes)) > 1 else None
    if sil_node_score:
        print(f"📐 Silhouette Score (GSOM Nodes): {sil_node_score:.4f}")

    # Step 9.2: Sample metrics
    data_points = pd.read_csv("output_iris.csv")
    node_clusters = pd.read_csv("gsom_node_clusters_iris.csv") if os.path.exists("gsom_node_clusters_iris.csv") else active_nodes
    merged = pd.merge(data_points, node_clusters[["output", "cluster"]], on="output", how="left")

    merged_valid = merged.dropna(subset=["cluster"]).copy()
    X_samples = merged_valid[["x", "y"]].to_numpy()
    labels_samples = merged_valid["cluster"].astype(int).to_numpy()

    clean_ids = merged_valid["Id"].astype(str)
    true_labels = df.set_index("Id").loc[clean_ids, "label_numeric"].to_numpy()

    if len(set(labels_samples)) > 1:
        sil_sample_score = silhouette_score(X_samples, labels_samples)
        ari_score = adjusted_rand_score(true_labels, labels_samples)
        nmi_score = normalized_mutual_info_score(true_labels, labels_samples)
        print(f"📐 Silhouette Score (Samples): {sil_sample_score:.4f}")
        print(f"🧮 Adjusted Rand Index (ARI): {ari_score:.4f}")
        print(f"🧠 Normalized Mutual Information (NMI): {nmi_score:.4f}")
    else:
        sil_sample_score = ari_score = nmi_score = None
        print("⚠️ Not enough sample clusters to compute metrics.")

    # Step 10: Save results
    with open("results/silhouette_scores_iris.txt", "w") as f:
        f.write("GSOM + Hierarchical Clustering Evaluation (Iris)\n")
        f.write("==============================================\n")
        f.write(f"Best Linkage Method: {best_method}\n")
        f.write(f"Cophenetic Correlation Coefficient (CCC): {best_coph:.4f}\n\n")
        if sil_node_score:
            f.write(f"Silhouette Score (GSOM Nodes): {sil_node_score:.4f}\n")
        if sil_sample_score:
            f.write(f"Silhouette Score (Samples): {sil_sample_score:.4f}\n")
            f.write(f"Adjusted Rand Index (ARI): {ari_score:.4f}\n")
            f.write(f"Normalized Mutual Information (NMI): {nmi_score:.4f}\n")

    # Step 11: Save node labels
    def formatted_label(row):
        counter = Counter(row["Species"])
        major = counter.most_common(1)[0][0]
        percent = counter[major] / sum(counter.values()) * 100
        return f"Cluster {row['cluster']} | {major} ({percent:.1f}%) | N={sum(counter.values())}"

    active_nodes["label_summary"] = active_nodes.apply(formatted_label, axis=1)
    active_nodes.to_csv("gsom_node_clusters_iris.csv", index=False)

    # Step 12: Plot dendrogram
    plt.figure(figsize=(18, 8))
    dendrogram(Z, labels=active_nodes["label_summary"].values, leaf_rotation=90, leaf_font_size=10)
    plt.title(f"Hierarchical Clustering on GSOM Nodes (Linkage: {best_method})")
    plt.xlabel("GSOM Nodes (Iris Species)")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig("hierarchical_clustering_iris_annotated.png")
    plt.close()

    # Step 13: Save final sample-cluster mapping
    merged.to_csv("iris_sample_cluster_mapping.csv", index=False)

    print("✅ Dendrogram saved as 'hierarchical_clustering_iris_annotated.png'")
    print("✅ Node clusters saved as 'gsom_node_clusters_iris.csv'")
    print("✅ Sample-cluster mapping saved as 'iris_sample_cluster_mapping.csv'")
