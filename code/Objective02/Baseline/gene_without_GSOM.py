import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score
from collections import Counter
import matplotlib.pyplot as plt
import os

# === Load your dataset ===
df = pd.read_csv("../example/data/GSE/GSE5281_merged_expression_metadata.csv")

# === Extract expression data (all columns after Disease_State) ===
expression_data = df.iloc[:, 3:].to_numpy()
sample_ids = df['Sample_ID'].tolist()  # For dendrogram labels

# === Perform hierarchical clustering on raw data ===
Z_raw = linkage(expression_data, method='average')

# === Calculate CCC ===
dist_matrix = pdist(expression_data)
ccc, _ = cophenet(Z_raw, dist_matrix)
print(f"\n✅ Baseline Cophenetic Correlation Coefficient (CCC): {ccc:.4f}")

# === Evaluate k=2 to 10 clusters ===
for k in range(2, 11):
    print(f"\n[Baseline] k = {k}")
    cluster_labels = fcluster(Z_raw, t=k, criterion='maxclust')

    # Silhouette score
    if len(set(cluster_labels)) > 1:
        sil_score = silhouette_score(expression_data, cluster_labels)
        print(f"✅ Silhouette Score (Raw) for k={k}: {sil_score:.4f}")
    else:
        print(f"⚠️ Only one cluster formed at k={k}")

    # Evaluate cluster purity
    df_temp = df[['Sample_ID', 'Disease_State']].copy()
    df_temp["Cluster"] = cluster_labels

    cluster_summary = []
    for cluster_id in sorted(df_temp["Cluster"].unique()):
        cluster_samples = df_temp[df_temp["Cluster"] == cluster_id]

        # === Cluster purity computation with cleaned Disease_State strings ===
        labels = [
            label.strip()
            for label_list in cluster_samples['Disease_State']
            for label in str(label_list).replace('\xa0', '').replace('Â', '').split(',')
        ]

        label_counts = Counter(labels)
        if label_counts:
            dominant_label, dominant_count = label_counts.most_common(1)[0]
            purity = dominant_count / len(labels) * 100
        else:
            dominant_label, purity = "N/A", 0.0

        cluster_summary.append({
            "Cluster": cluster_id,
            "Dominant Label": dominant_label,
            "% Purity": f"{purity:.1f}%",
            "Total Count": len(labels)
        })

    # === Save cluster summary ===
    summary_df = pd.DataFrame(cluster_summary)
    os.makedirs("output_gene_bottomup", exist_ok=True)
    summary_df.to_csv(f"output_gene_bottomup/baseline_raw_cluster_summary_k{k}.csv", index=False)
    print("Cluster summary saved.")

# === Plot dendrogram with correct sample labels ===
plt.figure(figsize=(14, 6))
dendrogram(Z_raw, labels=sample_ids, leaf_rotation=90, leaf_font_size=6)
plt.title("Dendrogram: Raw Gene Expression (No GSOM)")
plt.xlabel("Sample ID")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig("output_gene_bottomup/baseline_dendrogram_raw_data.png", dpi=300)
plt.show()
