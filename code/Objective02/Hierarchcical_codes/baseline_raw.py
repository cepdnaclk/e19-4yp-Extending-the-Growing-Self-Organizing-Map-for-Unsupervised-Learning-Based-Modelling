"""
Fast baseline comparison for gene expression clustering:
- RAW(z-scored) + HClust
- PCA(90% variance) + HClust
- SOM (20x20 BMU) + HClust

Config:
- Uses top 1000 most variable genes
- Evaluates only k = 2, 4
- Outputs a single merged CSV with all results
"""

import os, numpy as np, pandas as pd
from scipy.spatial.distance import pdist, cdist
from scipy.cluster.hierarchy import linkage, cophenet, fcluster
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import warnings

# ---------- CONFIG ----------
INPUT_CSV = "../example/data/GSE/transposed_with_meta.csv"
OUT_DIR   = "../output_gene_clustering_baselines"
OUT_CSV   = os.path.join(OUT_DIR, "baseline_fast_all_metrics.csv")

TOP_N_GENES    = 1000
K_LIST         = [2,4]
PCA_VAR        = 0.90
SOM_SIDE       = 20
LINKAGE_METHOD = "average"
DIST           = "euclidean"
# ----------------------------

def clean_text(s): 
    return str(s).replace("\xa0"," ").replace("Â","").strip()

def zscore_by_gene(M):
    m = M.mean(axis=1, keepdims=True)
    s = M.std(axis=1, keepdims=True) + 1e-8
    return (M - m) / s

def dunn_index(X, lab):
    u = np.unique(lab)
    if len(u) < 2: 
        return np.nan
    D = cdist(X, X)
    idx = [np.where(lab == c)[0] for c in u]
    inter = min(D[np.ix_(idx[i], idx[j])].min() 
                for i in range(len(idx)) for j in range(i+1, len(idx)))
    intra = max(0 if len(idc) <= 1 else D[np.ix_(idc, idc)].max() 
                for idc in idx)
    return inter / (intra + 1e-12)

if __name__=="__main__":
    # ---- Load dataset ----
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    df.columns = [clean_text(c) for c in df.columns]
    df.iloc[:,0] = df.iloc[:,0].apply(clean_text)

    if df.columns[0] != "Gene":
        raise ValueError("First column must be 'Gene'")
    df = df.set_index("Gene")

    expr = df.drop(index=["Brain_Region","Disease_State"])
    expr = expr.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # ---- Keep top-N variable genes ----
    var = expr.var(axis=1)
    expr = expr.loc[var.nlargest(TOP_N_GENES).index]

    X_raw = expr.to_numpy(dtype=float)
    Xz = zscore_by_gene(X_raw)
    genes = expr.index.tolist()

    # ---- Embeddings ----
    embeddings = {}
    embeddings["RAWz"] = Xz

    # PCA baseline
    pca = PCA(n_components=PCA_VAR, random_state=1)
    embeddings[f"PCA{int(PCA_VAR*100)}"] = pca.fit_transform(Xz)

    # SOM baseline
    try:
        from minisom import MiniSom
        som = MiniSom(SOM_SIDE, SOM_SIDE, Xz.shape[1], 
                      sigma=1.0, learning_rate=0.5, 
                      neighborhood_function='gaussian', random_seed=1)
        som.random_weights_init(Xz)
        som.train_random(Xz, 500)  # fewer iterations for speed
        codebook = np.array([w for row in som.get_weights() for w in row])
        bmu_idx = cdist(Xz, codebook, metric=DIST).argmin(axis=1)
        Xsom = codebook[bmu_idx,:]
        embeddings[f"SOM{SOM_SIDE}x{SOM_SIDE}"] = Xsom
    except Exception as e:
        warnings.warn(f"SOM baseline skipped: {e}")

    # ---- Evaluate each embedding ----
    rows = []
    for name, Xrep in embeddings.items():
        Z = linkage(Xrep, method=LINKAGE_METHOD, metric=DIST)
        ccc, _ = cophenet(Z, pdist(Xrep, metric=DIST))

        for k in K_LIST:
            labs = fcluster(Z, t=k, criterion="maxclust")
            if len(np.unique(labs)) > 1:
                sil  = silhouette_score(Xrep, labs)
                dbi  = davies_bouldin_score(Xrep, labs)
                ch   = calinski_harabasz_score(Xrep, labs)
                dnn  = dunn_index(Xrep, labs)
            else:
                sil = dbi = ch = dnn = np.nan

            rows.append({
                "Method": name,
                "k": k,
                "CCC_tree": ccc,
                "Silhouette": sil,
                "DBI": dbi,
                "CH": ch,
                "Dunn": dnn
            })

    # --- Ensure output folder & save CSV ---
    os.makedirs(OUT_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print("✅ Done. Results saved to:", OUT_CSV)
