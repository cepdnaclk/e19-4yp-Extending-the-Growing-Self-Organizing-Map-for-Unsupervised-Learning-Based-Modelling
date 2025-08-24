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

# Mapping file to translate dataset gene IDs to symbols (set to a path or keep None)
MAPPING_CSV        = None  # e.g., "../example/data/GSE/gene_id_to_symbol.csv"
MAPPING_SRC_COLS   = ["ensembl_id", "gene_id", "id", "GeneID"]
MAPPING_DST_COLS   = ["symbol", "gene_symbol", "hgnc_symbol"]

# GSOM & clustering
SPREAD             = 0.50        # lower => less growth
MAX_RADIUS         = 3
INITIAL_NODE_CAP   = 50_000
TRAIN_ITERS        = 100
SMOOTH_ITERS       = 50
LINKAGE_METHOD     = "average"

# Gene filtering
TOP_N_VAR_GENES    = 5000        # set None to use all genes

# Extra ks to summarize (in addition to best silhouette k)
EXTRA_KS           = [4, 6, 8, 10]

# AD biomarker list (exact symbols)
AD_BIOMARKERS = [
    "AC004951.6","MAFF","SLC39A12","PCYOX1L","CTD-3092A11.2","RP11-271C24.3","PRO1804","PRR34-AS1",
    "SST","CHGB","MT1M","JPX","APLNR","PPEF1"
]
# ====================================================


def clean_text(s: str) -> str:
    return str(s).replace("\xa0", " ").replace("Â", "").strip()


def pick_first_present(df_cols, candidates):
    """Pick the first column name in candidates that exists in df_cols (case-insensitive)."""
    lower = {c.lower(): c for c in df_cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def try_apply_mapping(expr: pd.DataFrame, mapping_path: str) -> pd.DataFrame:
    """
    If a mapping CSV is provided, map expr.index -> gene symbols using the provided file.
    Keeps duplicate targets by collapsing (averaging) rows that map to the same symbol.
    """
    print(f"\nTrying alias mapping from: {mapping_path}")
    mapdf = pd.read_csv(mapping_path)
    src_col = pick_first_present(mapdf.columns, MAPPING_SRC_COLS)
    dst_col = pick_first_present(mapdf.columns, MAPPING_DST_COLS)
    if not src_col or not dst_col:
        print(f"  ! Could not find mapping columns. Looked for src in {MAPPING_SRC_COLS}, dst in {MAPPING_DST_COLS}. Skipping mapping.")
        return expr

    # Clean mapping text
    mapdf[src_col] = mapdf[src_col].astype(str).map(clean_text)
    mapdf[dst_col] = mapdf[dst_col].astype(str).map(clean_text)

    # Clean expr index
    expr.index = pd.Index([clean_text(i) for i in expr.index], name="id")

    present = mapdf[mapdf[src_col].isin(expr.index)][[src_col, dst_col]].dropna()
    if present.empty:
        print("  ! None of your dataset IDs were found in the mapping file. Skipping mapping.")
        return expr

    # Map index -> symbol; keep rows that get a symbol; average duplicates
    expr_mapped = expr.copy()
    expr_mapped["__symbol__"] = expr_mapped.index.map(dict(zip(present[src_col], present[dst_col])))
    before = expr_mapped.shape[0]
    expr_mapped = expr_mapped.dropna(subset=["__symbol__"]).copy()

    # Collapse duplicates (same symbol) by averaging
    expr_mapped = (expr_mapped.groupby("__symbol__", as_index=True).mean(numeric_only=True))
    expr_mapped.index.name = "Gene"
    after = expr_mapped.shape[0]

    print(f"  Mapped {before} → {after} rows (averaged duplicates by symbol).")
    return expr_mapped


# -------------------- Extra Metric Helpers --------------------
def dunn_index(X, labels):
    """Higher is better. Uses Euclidean distances."""
    from numpy.linalg import norm
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return np.nan
    # precompute pairwise distances (memory-safe-ish for a few thousand points)
    D = cdist(X, X, metric="euclidean")
    idxs = [np.where(labels == c)[0] for c in uniq]
    # min inter-cluster distance
    inter = np.inf
    for i in range(len(idxs)):
        for j in range(i+1, len(idxs)):
            d = D[np.ix_(idxs[i], idxs[j])].min()
            if d < inter:
                inter = d
    # max intra-cluster diameter
    intra = 0.0
    for idc in idxs:
        if len(idc) > 1:
            diam = D[np.ix_(idc, idc)].max()
        else:
            diam = 0.0
        intra = max(intra, diam)
    return inter / (intra + 1e-12)


def s_dbw_index(X, labels):
    """
    S_Dbw (simplified): lower is better.
    Combines average intra-cluster scatter with inter-cluster density near midpoints.
    Based on Halkidi et al. (2001) idea; light-weight implementation sufficient for comparison.
    """
    uniq = np.unique(labels)
    k = len(uniq)
    if k < 2:
        return np.nan
    # centroids & scatter
    centroids, scatters = [], []
    for c in uniq:
        Xi = X[labels == c]
        mu = Xi.mean(axis=0)
        centroids.append(mu)
        scatters.append(np.sqrt(((Xi - mu) ** 2).sum(axis=1)).mean())
    centroids = np.vstack(centroids)
    scatters = np.array(scatters)
    # overall scatter
    stdev_overall = np.sqrt(((X - X.mean(axis=0)) ** 2).sum(axis=1)).mean()
    scatter_term = (scatters.mean() / (stdev_overall + 1e-12))
    # density term: density at midpoints vs. near centroids
    def density_count(Xc, ref, sigma):
        d = np.linalg.norm(Xc - ref, axis=1)
        return (d <= sigma).sum()
    sigma = scatters.mean() if np.isfinite(scatters.mean()) else 1.0
    dens_ratio = 0.0
    for i in range(k):
        # nearest centroid to i
        dists = np.linalg.norm(centroids - centroids[i], axis=1)
        dists[i] = np.inf
        j = dists.argmin()
        mid = (centroids[i] + centroids[j]) / 2.0
        dens_mid = density_count(X, mid, sigma)
        dens_i = density_count(X, centroids[i], sigma)
        dens_j = density_count(X, centroids[j], sigma)
        denom = (dens_i + dens_j) if (dens_i + dens_j) > 0 else 1.0
        dens_ratio += dens_mid / denom
    dens_term = dens_ratio / k
    return scatter_term + dens_term


def biomarker_enrichment(labels, gene_ids, biomarker_set):
    """
    Hypergeometric test per cluster: are biomarkers over-represented?
    Returns dict: cluster -> (count_bio, size, pvalue)
    """
    N = len(gene_ids)
    K = sum([g in biomarker_set for g in gene_ids])  # total biomarkers in dataset
    res = {}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        n = len(idx)
        k = sum([gene_ids[i] in biomarker_set for i in idx])
        pval = hypergeom.sf(k - 1, N, K, n)  # P(X >= k)
        res[int(c)] = (int(k), int(n), float(pval))
    return res


def bootstrap_stability(X, base_labels, n_boot=15, frac=0.8, random_state=1):
    """
    Resample genes, recluster with same linkage, compare ARI against base on the subset.
    Returns mean, std of ARI across bootstraps.
    """
    rng = np.random.RandomState(random_state)
    k = len(np.unique(base_labels))
    aris = []
    n = X.shape[0]
    for _ in range(n_boot):
        idx = rng.choice(n, size=int(frac * n), replace=False)
        Xb = X[idx]
        Zb = linkage(Xb, method=LINKAGE_METHOD)
        lb = fcluster(Zb, t=k, criterion="maxclust")
        base_sub = base_labels[idx]
        aris.append(adjusted_rand_score(base_sub, lb))
    return float(np.mean(aris)), float(np.std(aris))


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

    # Sanity for metadata rows
    if not {"Brain_Region", "Disease_State"}.issubset(df.index):
        raise ValueError("Missing 'Brain_Region' and/or 'Disease_State' rows in the file.")

    # === Expression matrix: rows=GENES, cols=SAMPLES ===
    expr = df.drop(index=["Brain_Region", "Disease_State"])
    expr = expr.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # Optional alias mapping
    if MAPPING_CSV is not None and os.path.exists(MAPPING_CSV):
        expr = try_apply_mapping(expr, MAPPING_CSV)
    else:
        expr.index = pd.Index([clean_text(i) for i in expr.index], name="Gene")

    # Biomarkers present/absent BEFORE filtering
    biomarkers_present_before = sorted(set(AD_BIOMARKERS) & set(expr.index))
    biomarkers_missing_before = sorted(set(AD_BIOMARKERS) - set(expr.index))
    print("\nBiomarkers present (before filtering):", biomarkers_present_before)
    print("Biomarkers missing (before filtering):", biomarkers_missing_before)

    # === Top-N most variable genes (force-include present biomarkers) ===
    if TOP_N_VAR_GENES is not None and TOP_N_VAR_GENES < expr.shape[0]:
        var = expr.var(axis=1)
        topN = set(var.nlargest(TOP_N_VAR_GENES).index)
        must = set(biomarkers_present_before)
        keep = list(topN | must)
        expr = expr.loc[keep]

    # Biomarkers AFTER filtering (diagnostic)
    biomarkers_present_after = sorted(set(AD_BIOMARKERS) & set(expr.index))
    biomarkers_missing_after = sorted(set(AD_BIOMARKERS) - set(expr.index))
    print("\nBiomarkers present (after filtering):", biomarkers_present_after)
    print("Biomarkers missing (after filtering):", biomarkers_missing_after)

    gene_ids   = expr.index.tolist()
    sample_ids = expr.columns.tolist()
    print(f"\nGenes used: {len(gene_ids)} | Samples: {len(sample_ids)}")

    # === GSOM on genes × samples ===
    gsom = GSOM(
        spred_factor=SPREAD,           # rename to spread_factor if your GSOM uses that arg name
        dimensions=expr.shape[1],
        max_radius=MAX_RADIUS,
        initial_node_size=INITIAL_NODE_CAP
    )
    gsom.fit(expr.to_numpy(), training_iterations=TRAIN_ITERS, smooth_iterations=SMOOTH_ITERS)

    # ==== 1) Map every gene to its BMU (vector) ====
    print("\nGSOM training complete.")
    print("Node count:", gsom.node_count)

    node_embeddings = np.asarray(gsom.node_list[:gsom.node_count])  # (n_nodes, dim)
    X = expr.to_numpy(dtype=float)                                   # (n_genes, dim)
    gene_ids_clean = pd.Series(gene_ids).map(clean_text).tolist()

    # BMU via vectorized distances
    D = cdist(X, node_embeddings, metric="euclidean")
    bmu_idx = D.argmin(axis=1).astype(int)

    out = pd.DataFrame({"Gene_ID": gene_ids_clean, "Node_ID": bmu_idx})
    AD_SET = set(AD_BIOMARKERS)
    out["Is_AD_Biomarker"] = out["Gene_ID"].isin(AD_SET).astype(int)

    out[["Gene_ID","Node_ID","Is_AD_Biomarker"]].to_csv(
        os.path.join(OUT_DIR, "genes_to_nodes.csv"), index=False
    )

    print("\nDiagnostics:")
    print("  Genes mapped to BMUs:", len(out), "(expected:", X.shape[0], ")")
    print("  Unique nodes used:", out["Node_ID"].nunique(), "/ total nodes:", len(node_embeddings))
    print("  Biomarkers matched in mapping:", out["Is_AD_Biomarker"].sum(), f"/ {len(AD_BIOMARKERS)}")

    # ==== 2) HClust on GSOM nodes + quality (CCC / Silhouette) ====
    Z_nodes = linkage(node_embeddings, method=LINKAGE_METHOD)

    # CCC (nodes)
    ccc_nodes, _ = cophenet(Z_nodes, pdist(node_embeddings))
    print(f"\nCCC (nodes): {ccc_nodes:.4f}")

    # Silhouette across k
    best_k, best_sil = None, -1.0
    node_cluster_labels_by_k = {}
    upper_k = min(11, max(2, len(node_embeddings)))
    for k in range(2, upper_k):
        clabs = fcluster(Z_nodes, t=k, criterion="maxclust")
        node_cluster_labels_by_k[k] = clabs
        if len(set(clabs)) > 1:
            sil = silhouette_score(node_embeddings, clabs)
            if sil > best_sil:
                best_sil, best_k = sil, k
            print(f"Silhouette (nodes) @k={k}: {sil:.4f}")
        else:
            print(f"Silhouette undefined @k={k} (one cluster)")

    # Save node dendrogram
    plt.figure(figsize=(12, 6))
    dendrogram(Z_nodes, labels=[str(i) for i in range(len(node_embeddings))], leaf_rotation=90)
    plt.title("GSOM Nodes: Hierarchical Clustering")
    plt.xlabel("Node ID"); plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "dendrogram_nodes.png"), dpi=300)
    plt.close()

    # ==== 3) Map genes -> node clusters (best_k + EXTRA_KS) + METRICS ====
    if best_k is None:
        best_k = 4
    ks_to_do = sorted(set([best_k] + [k for k in EXTRA_KS if 2 <= k <= max(2, len(node_embeddings))]))

    # Prepare shared embeddings once
    node_id_array  = out["Node_ID"].astype(int).values
    gene_embeddings = node_embeddings[node_id_array, :]  # (n_genes, dim)

    metrics_rows = []
    for k in ks_to_do:
        clabs = node_cluster_labels_by_k.get(k)
        if clabs is None:
            clabs = fcluster(Z_nodes, t=k, criterion="maxclust")
            node_cluster_labels_by_k[k] = clabs

        node_to_cluster = {i: clabs[i] for i in range(len(clabs))}
        out[f"Cluster_k{k}"] = out["Node_ID"].map(node_to_cluster).astype(int)
        labels_k = out[f"Cluster_k{k}"].to_numpy()

        # Save per-k assignments
        out[['Gene_ID','Node_ID','Is_AD_Biomarker',f'Cluster_k{k}']].to_csv(
            os.path.join(OUT_DIR, f"genes_nodes_clusters_k{k}.csv"), index=False
        )

        # Summary per cluster
        summ = (out.groupby(f"Cluster_k{k}", dropna=False)
                  .agg(Total_Genes=("Gene_ID","count"),
                       Biomarkers=("Is_AD_Biomarker","sum"))
                  .reset_index()
                  .sort_values(f"Cluster_k{k}"))
        summ.to_csv(os.path.join(OUT_DIR, f"cluster_summary_k{k}.csv"), index=False)
        print(f"\nCluster summary (k={k}):")
        print(summ)

        # Export per-cluster gene lists
        for cid, grp in out.groupby(f"Cluster_k{k}"):
            path = os.path.join(OUT_DIR, f"genes_in_cluster_k{k}_{cid}.txt")
            grp["Gene_ID"].to_csv(path, index=False, header=False)

        # -------- Extra metrics for this k --------
        if len(np.unique(labels_k)) > 1:
            # Internal indices on gene embeddings
            dbi  = davies_bouldin_score(gene_embeddings, labels_k)
            ch   = calinski_harabasz_score(gene_embeddings, labels_k)
            dunn = dunn_index(gene_embeddings, labels_k)
            sdbw = s_dbw_index(gene_embeddings, labels_k)
            print(f"[Metrics @k={k}] DBI: {dbi:.4f} (↓), CH: {ch:.2f} (↑), Dunn: {dunn:.4f} (↑), S_Dbw: {sdbw:.4f} (↓)")
        else:
            dbi = ch = dunn = sdbw = np.nan
            print(f"[Metrics @k={k}] Only one cluster; internal indices undefined.")

        # Trustworthiness: raw space (expr) → GSOM gene embeddings
        try:
            tw = trustworthiness(expr.to_numpy(dtype=float), gene_embeddings, n_neighbors=10)
            print(f"Trustworthiness (raw → GSOM embeddings): {tw:.4f} (↑)")
        except Exception as e:
            tw = np.nan
            print("Trustworthiness not computed:", e)

        # Biomarker enrichment per cluster
        bio_res = biomarker_enrichment(labels_k, out["Gene_ID"].tolist(), AD_SET)
        for c,(k_bio, n_c, p) in sorted(bio_res.items()):
            print(f"  Cluster {c}: biomarkers={k_bio}/{n_c}, hypergeom p={p:.3e}")

        # Bootstrap stability (ARI)
        mean_ari, std_ari = bootstrap_stability(gene_embeddings, labels_k, n_boot=15, frac=0.8, random_state=1)
        print(f"Bootstrap stability (ARI) @k={k}: mean={mean_ari:.3f} ± {std_ari:.3f}")

        # Row for CSV
        metrics_rows.append({
            "k": k,
            "Silhouette_nodes": silhouette_score(node_embeddings, node_cluster_labels_by_k[k]) if len(set(node_cluster_labels_by_k[k])) > 1 else np.nan,
            "CCC_nodes": ccc_nodes,
            "DBI_genes": dbi,
            "CH_genes": ch,
            "Dunn_genes": dunn,
            "S_Dbw_genes": sdbw,
            "Trustworthiness_raw_to_GSOM": tw,
            "Bootstrap_ARI_mean": mean_ari,
            "Bootstrap_ARI_std": std_ari
        })

    # Save metrics table
    metrics_df = pd.DataFrame(metrics_rows).sort_values("k")
    metrics_df.to_csv(os.path.join(OUT_DIR, "metrics_k_summary.csv"), index=False)

    # ==== 4) FULL gene-level dendrogram via GSOM BMU embeddings ====
    gene_labels = out["Gene_ID"].tolist()
    Z_genes = linkage(gene_embeddings, method=LINKAGE_METHOD)
    ccc_genes, _ = cophenet(Z_genes, pdist(gene_embeddings))
    print(f"\nCCC (genes via GSOM embeddings): {ccc_genes:.4f}")

    label_marked = [("★ " + g if g in AD_BIOMARKERS else g) for g in gene_labels]
    plt.figure(figsize=(14, max(12, len(gene_labels) * 0.02)))
    dendrogram(Z_genes, labels=label_marked, leaf_rotation=90, leaf_font_size=6)
    plt.title("Full Genes Dendrogram via GSOM Embeddings (★ = AD biomarker)")
    plt.xlabel("Genes"); plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "dendrogram_genes_gsom_embeddings.png"), dpi=300)
    plt.close()

    # Save dendrogram order (handy for reports)
    ddata = dendrogram(Z_genes, no_plot=True)
    order = ddata["leaves"]
    pd.Series([gene_labels[i] for i in order], name="Gene") \
      .to_csv(os.path.join(OUT_DIR, "genes_dendrogram_order.csv"), index=False)

    # ==== 5) Biomarkers (+ 15 high-variance non-biomarkers) dendrogram ====
    bio_genes = sorted(AD_SET & set(out["Gene_ID"]))
    if len(bio_genes) >= 2:
        # pick 15 high-variance non-biomarkers from CURRENT expr matrix
        var_series = expr.var(axis=1)
        non_bio_candidates = [g for g in out["Gene_ID"] if g not in AD_SET and g in var_series.index]

        n_bg = min(15, len(non_bio_candidates))
        if n_bg > 0:
            bg_ranked = (var_series.loc[non_bio_candidates]
                         .sort_values(ascending=False)
                         .head(n_bg)
                         .index
                         .tolist())
        else:
            bg_ranked = []

        subset_genes = bio_genes + bg_ranked
        subset = out[out["Gene_ID"].isin(subset_genes)].copy()

        subset_node_ids = subset["Node_ID"].astype(int).values
        subset_embed    = node_embeddings[subset_node_ids, :]
        subset_labels   = [("★ " + g if g in AD_SET else g) for g in subset["Gene_ID"].tolist()]

        Z_sub = linkage(subset_embed, method=LINKAGE_METHOD)
        plt.figure(figsize=(12, max(6, len(subset_labels) * 0.4)))
        dendrogram(Z_sub, labels=subset_labels, leaf_rotation=90)
        plt.title("Biomarkers vs High-Variance Background Genes (★ = AD biomarker)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "dendrogram_biomarkers_vs_background.png"), dpi=300)
        plt.close()

        print(f"\nBiomarker+background dendrogram saved. Biomarkers = {len(bio_genes)}, Background = {len(bg_ranked)}")
    else:
        print("\nNot enough biomarkers available to build comparison dendrogram.")

    print("\nDone. All outputs in:", OUT_DIR)
