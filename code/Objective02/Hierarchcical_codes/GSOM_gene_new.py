import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, cdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
from sklearn.manifold import trustworthiness
from scipy.stats import hypergeom

from GSOM import GSOM

# ====================== CONFIG ======================
INPUT_CSV          = "../example/data/GSE/transposed_with_meta.csv"
OUT_DIR            = "../output_gene_clustering"

# GSOM & clustering
SPREAD             = 0.50
MAX_RADIUS         = 3
INITIAL_NODE_CAP   = 50_000
TRAIN_ITERS        = 100
SMOOTH_ITERS       = 50
LINKAGE_METHOD     = "average"

# Gene filtering for training
TOP_N_VAR_GENES    = 5000

# AD biomarker list
AD_BIOMARKERS = [
    "AC004951.6","MAFF","SLC39A12","PCYOX1L","CTD-3092A11.2","RP11-271C24.3","PRO1804","PRR34-AS1",
    "SST","CHGB","MT1M","JPX","APLNR","PPEF1"
]

# -------------------- Helpers --------------------
def clean_text(s: str) -> str:
    return str(s).replace("\xa0", " ").replace("Â", "").strip()

# ========================== MAIN ==========================
if __name__ == "__main__":
    np.random.seed(1)
    os.makedirs(OUT_DIR, exist_ok=True)

    # === Load & clean ===
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    df.columns = [clean_text(c) for c in df.columns]
    df.iloc[:, 0] = df.iloc[:, 0].apply(clean_text)
    if df.columns[0] != "Gene":
        raise ValueError(f"Expected first column 'Gene', got {df.columns[0]}")
    df = df.set_index("Gene")

    # Expect metadata rows present; drop to get genes×samples
    if not {"Brain_Region", "Disease_State"}.issubset(df.index):
        raise ValueError("Missing 'Brain_Region' and/or 'Disease_State' rows in the file.")
    expr = df.drop(index=["Brain_Region", "Disease_State"])
    expr = expr.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    expr.index = pd.Index([clean_text(i) for i in expr.index], name="Gene")

    # Filter to top-N variable genes (always keep any biomarkers present)
    biomarkers_present = sorted(set(AD_BIOMARKERS) & set(expr.index))
    if TOP_N_VAR_GENES and TOP_N_VAR_GENES < expr.shape[0]:
        var = expr.var(axis=1)
        topN = set(var.nlargest(TOP_N_VAR_GENES).index)
        keep = list(topN | set(biomarkers_present))
        expr = expr.loc[keep]

    # === GSOM training on genes×samples ===
    try:
        gsom = GSOM(spread_factor=SPREAD, dimensions=expr.shape[1],
                    max_radius=MAX_RADIUS, initial_node_size=INITIAL_NODE_CAP)
    except TypeError:
        # fallback for classes that use 'spred_factor'
        gsom = GSOM(spred_factor=SPREAD, dimensions=expr.shape[1],
                    max_radius=MAX_RADIUS, initial_node_size=INITIAL_NODE_CAP)

    gsom.fit(expr.to_numpy(), training_iterations=TRAIN_ITERS, smooth_iterations=SMOOTH_ITERS)
    print("\nGSOM training complete.")
    print("Node count:", gsom.node_count)

    # Codebook vectors (node embeddings)
    node_embeddings = np.asarray(gsom.node_list[:gsom.node_count])  # (n_nodes, dim)

    # Map each gene to its BMU node
    X = expr.to_numpy(dtype=float)
    D = cdist(X, node_embeddings, metric="euclidean")
    bmu_idx = D.argmin(axis=1).astype(int)
    out = pd.DataFrame({"Gene_ID": expr.index.tolist(), "Node_ID": bmu_idx})

    # ---------------- Dendrograms (ALL) ----------------

    # 1) Node-level dendrogram
    Z_nodes = linkage(node_embeddings, method=LINKAGE_METHOD)
    ccc_nodes, _ = cophenet(Z_nodes, pdist(node_embeddings, metric="euclidean"))
    print(f"CCC (nodes, euclidean/average): {ccc_nodes:.4f}")

    plt.figure(figsize=(12, 6))
    dendrogram(Z_nodes, labels=[str(i) for i in range(len(node_embeddings))], leaf_rotation=90)
    plt.title(f"GSOM Node Dendrogram (Euclidean, average) — CCC={ccc_nodes:.3f}")
    plt.xlabel("Nodes")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "dendrogram_nodes.png"), dpi=300)
    plt.close()

    # 2) Full gene-level dendrogram (all genes via BMU embeddings)
    gene_embeddings = node_embeddings[bmu_idx, :]
    Z_genes_full = linkage(gene_embeddings, method=LINKAGE_METHOD)
    ccc_genes_full, _ = cophenet(Z_genes_full, pdist(gene_embeddings, metric="euclidean"))
    print(f"CCC (genes — full set, euclidean/average): {ccc_genes_full:.4f}")

    labels_full = out["Gene_ID"].tolist()
    plt.figure(figsize=(14, max(12, len(labels_full) * 0.02)))
    dendrogram(Z_genes_full, labels=labels_full, leaf_rotation=90, leaf_font_size=6)
    plt.title(f"Full Genes Dendrogram (Euclidean, average) — CCC={ccc_genes_full:.3f}")
    plt.xlabel("Genes")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "dendrogram_genes_full.png"), dpi=300)
    plt.close()

    # 3) Top-100 most-variable genes dendrogram (COMPACT)
    var_all = expr.var(axis=1)
    top100_genes = var_all.nlargest(100).index.tolist()
    top100_df = out[out["Gene_ID"].isin(top100_genes)].copy()
    if len(top100_df) >= 2:
        top100_embed = node_embeddings[top100_df["Node_ID"].astype(int).values, :]
        Z_top100 = linkage(top100_embed, method=LINKAGE_METHOD)
        ccc_top100, _ = cophenet(Z_top100, pdist(top100_embed, metric="euclidean"))
        print(f"CCC (top-100 genes, euclidean/average): {ccc_top100:.4f}")

        # <<< COMPACT FIGURE SETTINGS >>>
        plt.figure(figsize=(14, 6))  # compact height (was very tall before)
        dendrogram(
            Z_top100,
            labels=top100_df["Gene_ID"].tolist(),
            leaf_rotation=90,
            leaf_font_size=6  # slightly smaller labels
        )
        plt.title(f"Top 100 Variable Genes Dendrogram (Euclidean, average) — CCC={ccc_top100:.3f}")
        plt.xlabel("Genes")
        plt.ylabel("Distance")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "dendrogram_genes_top100.png"), dpi=300)
        plt.close()
    else:
        print("Not enough genes to build the top-100 dendrogram.")

    # 4) Biomarkers vs. background dendrogram
    AD_SET = set(AD_BIOMARKERS)
    bio_genes = sorted(AD_SET & set(out["Gene_ID"]))
    if len(bio_genes) >= 2:
        non_bio_candidates = [g for g in out["Gene_ID"] if g not in AD_SET and g in var_all.index]
        bg_ranked = (var_all.loc[non_bio_candidates]
                     .sort_values(ascending=False)
                     .head(15)
                     .index
                     .tolist())

        subset_genes = bio_genes + bg_ranked
        sub_df = out[out["Gene_ID"].isin(subset_genes)].copy()
        sub_embed = node_embeddings[sub_df["Node_ID"].astype(int).values, :]
        sub_labels = [("★ " + g if g in AD_SET else g) for g in sub_df["Gene_ID"].tolist()]

        Z_sub = linkage(sub_embed, method=LINKAGE_METHOD)
        ccc_sub, _ = cophenet(Z_sub, pdist(sub_embed, metric="euclidean"))
        print(f"CCC (biomarkers vs background, euclidean/average): {ccc_sub:.4f}")

        plt.figure(figsize=(12, max(6, len(sub_labels) * 0.35)))
        dendrogram(Z_sub, labels=sub_labels, leaf_rotation=90)
        plt.title(f"Biomarkers vs Background Genes Dendrogram (Euclidean, average) — CCC={ccc_sub:.3f}")
        plt.xlabel("Genes")
        plt.ylabel("Distance")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "dendrogram_biomarkers_vs_background.png"), dpi=300)
        plt.close()
    else:
        print("Not enough biomarkers present for the biomarker vs background dendrogram.")

    print("\nDone. Figures written to:", OUT_DIR)
