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
    from GSOM import GSOM

    np.random.seed(1)

    # Step 1: Load dataset
    df = pd.read_csv("example/data/GSE/GSE5281_normalized_gene_expression.csv", index_col=0)
    df = df.T
    df.index.name = "Sample_ID"

    # Step 2–3: Label and input prep
    df["Id"] = df.index
    df["Species"] = ["Alzheimer's Disease" if i < 87 else "Control" for i in range(161)]
    features = df.drop(columns=["Id", "Species"])
    full_input = pd.concat([features, df[["Id", "Species"]]], axis=1)

    # Step 4: Train GSOM
    gsom = GSOM(spred_factor=0.82, dimensions=features.shape[1], distance='euclidean', max_radius=4)
    gsom.fit(features.to_numpy(), training_iterations=100, smooth_iterations=50)

    # Step 5: Predict & save
    os.makedirs("output", exist_ok=True)  
    output = gsom.predict(full_input, index_col="Id", label_col="Species")
    output.to_csv("output/output_gse5281.csv", index=False)  

    # Step 5.1: Plot GSOM node map
    os.makedirs("GSOM_maps", exist_ok=True)  
    plt.figure(figsize=(10, 8))
    plt.scatter(output["x"], output["y"], c=output["hit_count"], cmap='viridis', s=100, edgecolors='k')
    plt.colorbar(label='Hit Count')
    plt.title("GSOM Node Map (Hit Count Intensity)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("GSOM_maps/gsom_node_map_gse5281.png")  
    plt.close()
    print("🗺️ GSOM Node Map saved as 'GSOM_maps/gsom_node_map_gse5281.png'")

    # Step 6: Filter active nodes
    df_out = pd.read_csv("output/output_gse5281.csv")  
    active_nodes = df_out[df_out["hit_count"] > 0].copy()
    active_nodes["Species"] = active_nodes["Species"].apply(ast.literal_eval)
    active_nodes = active_nodes[active_nodes["hit_count"] >= 2].copy()

    # Step 7: Hierarchical clustering
    X = active_nodes[["x", "y"]].to_numpy()
    best_coph, best_method, best_Z = 0, None, None
    for method in ['single', 'complete', 'average', 'ward']:
        Z_temp = linkage(X, method=method)
        coph_corr, _ = cophenet(Z_temp, pdist(X))
        print(f"{method.capitalize():<10} CCC: {coph_corr:.4f}")
        if coph_corr > best_coph:
            best_coph, best_method, best_Z = coph_corr, method, Z_temp
    Z = best_Z
    print(f"\n✅ Best linkage method: {best_method} with CCC = {best_coph:.4f}")

    # Step 8–9: Clustering
    active_nodes["cluster"] = fcluster(Z, 2, criterion='maxclust')
    X_nodes = active_nodes[["x", "y"]].to_numpy()
    labels_nodes = active_nodes["cluster"].to_numpy()
    sil_node_score = silhouette_score(X_nodes, labels_nodes) if len(set(labels_nodes)) > 1 else None

    node_clusters_path = "node_clusters/gsom_node_clusters_gse5281.csv"  
    os.makedirs("node_clusters", exist_ok=True)  

    # Step 9.2: Sample metrics
    data_points = pd.read_csv("output/output_gse5281.csv")  
    node_clusters = pd.read_csv(node_clusters_path) if os.path.exists(node_clusters_path) else active_nodes
    merged = pd.merge(data_points, node_clusters[["output", "cluster"]], on="output", how="left")

    merged_valid = merged.dropna(subset=["cluster"]).copy()
    X_samples = merged_valid[["x", "y"]].to_numpy()
    labels_samples = merged_valid["cluster"].astype(int).to_numpy()
    df["label_numeric"] = df["Species"].replace({"Alzheimer's Disease": 1, "Control": 0})
    clean_ids = merged_valid["Id"].apply(lambda x: ast.literal_eval(x)[0] if isinstance(x, str) and x.startswith("[") else x).astype(str)
    true_labels = df.loc[clean_ids, "label_numeric"].to_numpy()

    sil_sample_score = ari_score = nmi_score = None
    if len(set(labels_samples)) > 1:
        sil_sample_score = silhouette_score(X_samples, labels_samples)
        ari_score = adjusted_rand_score(true_labels, labels_samples)
        nmi_score = normalized_mutual_info_score(true_labels, labels_samples)
        print(f"📐 Silhouette Score (Samples): {sil_sample_score:.4f}")
        print(f"🧮 Adjusted Rand Index (ARI): {ari_score:.4f}")
        print(f"🧠 Normalized Mutual Information (NMI): {nmi_score:.4f}")
    else:
        print("⚠️ Not enough sample clusters to compute clustering metrics.")

    # Step 10: Save metrics
    os.makedirs("results", exist_ok=True)  
    with open("results/silhouette_scores_gene.txt", "w") as f:  
        f.write("GSOM + Hierarchical Clustering Evaluation\n")
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

    # Step 11: Label summary
    def formatted_label(row):
        counter = Counter(row["Species"])
        ad = counter.get("Alzheimer's Disease", 0)
        ctrl = counter.get("Control", 0)
        total = ad + ctrl
        dominant = "AD" if ad > ctrl else "Control"
        percent = (max(ad, ctrl) / total) * 100 if total > 0 else 0
        return f"Cluster {row['cluster']} | {dominant} ({percent:.1f}%) | N={total}"

    active_nodes["label_summary"] = active_nodes.apply(formatted_label, axis=1)
    active_nodes.to_csv(node_clusters_path, index=False)  

    # Step 11.1: Plot label-based GSOM map
    plt.figure(figsize=(10, 8))
    color_map = {"AD": "red", "Control": "blue"}
    dominant_labels = active_nodes["label_summary"].apply(lambda x: "AD" if "AD" in x else "Control")
    colors = dominant_labels.map(color_map)
    plt.scatter(active_nodes["x"], active_nodes["y"], c=colors, s=120, edgecolors='black')
    for i, row in active_nodes.iterrows():
        plt.text(row["x"], row["y"], str(row["cluster"]), fontsize=8, ha='center', va='center', color='white')
    plt.title("GSOM Node Label Map (Dominant Class per Cluster)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("GSOM_maps/gsom_node_label_map_gse5281.png")  
    plt.close()

    # Step 12: Dendrogram
    os.makedirs("dendrograms", exist_ok=True)  
    plt.figure(figsize=(18, 8))
    dendrogram(
        Z,
        labels=active_nodes["label_summary"].values,
        leaf_rotation=90,
        leaf_font_size=10
    )
    plt.title(f"Hierarchical Clustering on GSOM Nodes (Linkage: {best_method})")
    plt.xlabel("Clustered GSOM Nodes (AD vs Control)")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig("dendrograms/hierarchical_clustering_gse5281_annotated.png")  
    plt.close()

    # Step 13: Save sample-cluster mapping
    os.makedirs("cluster_mapping", exist_ok=True)  
    merged.to_csv("cluster_mapping/gse5281_sample_cluster_mapping.csv", index=False)  

    print(" All files saved in respective folders.")