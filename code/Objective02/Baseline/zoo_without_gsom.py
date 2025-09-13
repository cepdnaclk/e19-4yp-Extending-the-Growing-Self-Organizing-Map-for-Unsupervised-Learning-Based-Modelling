import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score
from collections import Counter
import matplotlib.pyplot as plt
import os

# === Load Zoo dataset ===
df = pd.read_csv("../example/data/zoo.txt")

# === Extract features and labels ===
features = df.iloc[:, 1:-1].to_numpy()  # from w1 to w15
sample_ids = df['Name'].tolist()       # for dendrogram labels
true_labels = df['label']              # integer species class

# === Perform hierarchical clustering on raw features ===
Z_raw = linkage(features, method='average')

# === Calculate Cophenetic Correlation Coefficient ===
dist_matrix = pdist(features)
ccc, _ = cophenet(Z_raw, dist_matrix)
print(f"\n✅ Baseline Cophenetic Correlation Coefficient (CCC): {ccc:.4f}")

# === Evaluate k=2 to 10 clusters ===
for k in range(2, 11):
    print(f"\n[Baseline] k = {k}")
    cluster_labels = fcluster(Z_raw, t=k, criterion='maxclust')

    # Silhouette Score
    if len(set(cluster_labels)) > 1:
        sil_score = silhouette_score(features, cluster_labels)
        print(f"✅ Silhouette Score (Raw) for k={k}: {sil_score:.4f}")
    else:
        print(f"⚠️ Only one cluster formed at k={k}")

    # Evaluate cluster purity
    df_temp = df[['Name', 'label']].copy()
    df_temp['Cluster'] = cluster_labels

    cluster_summary = []
    for cluster_id in sorted(df_temp['Cluster'].unique()):
        cluster_samples = df_temp[df_temp['Cluster'] == cluster_id]
        label_counts = Counter(cluster_samples['label'])

        if label_counts:
            dominant_label, dominant_count = label_counts.most_common(1)[0]
            purity = dominant_count / len(cluster_samples) * 100
        else:
            dominant_label, purity = "N/A", 0.0

        cluster_summary.append({
            "Cluster": cluster_id,
            "Dominant Label": dominant_label,
            "% Purity": f"{purity:.1f}%",
            "Total Count": len(cluster_samples)
        })

    summary_df = pd.DataFrame(cluster_summary)
    os.makedirs("output_zoo_bottomup", exist_ok=True)
    summary_df.to_csv(f"output_zoo_bottomup/baseline_raw_cluster_summary_k{k}.csv", index=False)
    print("Cluster summary saved.")

# === Plot dendrogram ===
plt.figure(figsize=(16, 6))
dendrogram(Z_raw, labels=sample_ids, leaf_rotation=90, leaf_font_size=6)
plt.title("Dendrogram: Zoo Dataset (Raw Features)")
plt.xlabel("Animal Name")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig("output_zoo_bottomup/baseline_dendrogram_zoo_data.png", dpi=300)
plt.show()
