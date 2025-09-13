"""
Baseline comparison for gene expression clustering:
  - Raw data + HClust
  - PCA-reduced data + HClust
  - SOM codebook embedding + HClust
Outputs metrics for comparison against GSOM+HClust.
"""

import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, cdist
from scipy.cluster.hierarchy import linkage, cophenet, fcluster
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score, pairwise_distances

# ---------- CONFIG ----------
INPUT_CSV = "../example/data/GSE/transposed_with_meta.csv"
OUT_DIR   = "../output_gene_clustering_baselines_all"
K_LIST    = [2, 4, 6, 8, 10]
PCA_VARIANCE = 0.9
LINKAGE_METHOD = "average"
DIST_FOR_LINKAGE = "euclidean"
DIST_FOR_METRICS = "euclidean"

AD_BIOMARKERS = [
    "AC004951.6","MAFF","SLC39A12","PCYOX1L","CTD-3092A11.2","RP11-271C24.3",
    "PRO1804","PRR34-AS1","SST","CHGB","MT1M","JPX","APLNR","PPEF1"
]

# ---------- Helpers ----------
def clean_text(s): return str(s).replace("\xa0", " ").replace("Â", "").strip()

def zscore_by_gene(M):
    m = M.mean(axis=1, keepdims=True)
    s = M.std(axis=1, keepdims=True) + 1e-8
    return (M - m) / s

def dunn_index(X, labels):
    uniq = np.unique(labels)
    if len(uniq) < 2: return np.nan
    D = cdist(X, X)
    idxs = [np.where(labels == c)[0] for c in uniq]
    inter = np.inf
    for i in range(len(idxs)):
        for j in range(i+1, len(idxs)):
            d = D[np.ix_(idxs[i], idxs[j])].min()
            inter = min(inter, d)
    intra = 0
    for idc in idxs:
        if len(idc) > 1: intra = max(intra, D[np.ix_(idc, idc)].max())
    return inter / (intra + 1e-12)

def s_dbw_index(X, labels):
    uniq = np.unique(labels)
    if len(uniq) < 2: return np.nan
    centroids, scatters = [], []
    for c in uniq:
        Xi = X[labels == c]; mu = Xi.mean(axis=0)
        centroids.append(mu); scatters.append(np.linalg.norm(Xi-mu,axis=1).mean())
    centroids = np.vstack(centroids); scatters = np.array(scatters)
    stdev_all = np.linalg.norm(X - X.mean(axis=0), axis=1).mean()
    scatter_term = scatters.mean()/(stdev_all+1e-12)
    sigma = scatters.mean() if np.isfinite(scatters.mean()) else 1.0
    dens_ratio=0
    for i in range(len(centroids)):
        dists = np.linalg.norm(centroids-centroids[i], axis=1); dists[i]=np.inf
        j=dists.argmin(); mid=(centroids[i]+centroids[j])/2
        dens_mid=((np.linalg.norm(X-mid,axis=1)<=sigma).sum())
        dens_i=((np.linalg.norm(X-centroids[i],axis=1)<=sigma).sum())
        dens_j=((np.linalg.norm(X-centroids[j],axis=1)<=sigma).sum())
        denom=(dens_i+dens_j) if (dens_i+dens_j)>0 else 1
        dens_ratio+=dens_mid/denom
    return scatter_term+dens_ratio/len(centroids)

def bootstrap_stability(X, base_labels, n_boot=10, frac=0.8, random_state=1):
    rng = np.random.RandomState(random_state)
    k = len(np.unique(base_labels)); aris=[]; n = X.shape[0]
    for _ in range(n_boot):
        idx = rng.choice(n, size=int(frac*n), replace=False)
        Zb = linkage(X[idx], method=LINKAGE_METHOD)
        lb = fcluster(Zb, t=k, criterion="maxclust")
        aris.append(adjusted_rand_score(base_labels[idx], lb))
    return float(np.mean(aris)), float(np.std(aris))

def biomarker_tests(Xrep, labs, gene_ids, biomarker_set):
    G = gene_ids; ADset=set(biomarker_set)
    bio_idx=[i for i,g in enumerate(G) if g in ADset]
    # co-clustering rate
    same=0; total=0
    for i in range(len(bio_idx)):
        for j in range(i+1,len(bio_idx)):
            total+=1; same+=int(labs[bio_idx[i]]==labs[bio_idx[j]])
    cocluster = same/total if total>0 else np.nan
    # compactness
    Dm = pairwise_distances(Xrep, metric=DIST_FOR_METRICS)
    if len(bio_idx)>=2 and len(G)-len(bio_idx)>=2:
        wb=[Dm[i,j] for i in bio_idx for j in bio_idx if j>i]
        nb_idx=[i for i,g in enumerate(G) if g not in ADset]
        rng=np.random.RandomState(1)
        nb_sample=rng.choice(nb_idx,size=min(5*len(bio_idx),len(nb_idx)),replace=False)
        bb=[Dm[i,j] for i in bio_idx for j in nb_sample]
        mean_wb=np.mean(wb); mean_bb=np.mean(bb)
        compact_gain=mean_bb-mean_wb
    else:
        compact_gain=np.nan
    return cocluster, compact_gain

# ---------- MAIN ----------
if __name__=="__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    df.columns = [clean_text(c) for c in df.columns]
    df.iloc[:,0] = df.iloc[:,0].apply(clean_text)
    if df.columns[0]!="Gene": raise ValueError("First column must be 'Gene'")
    df=df.set_index("Gene")
    expr=df.drop(index=["Brain_Region","Disease_State"])
    expr=expr.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    X_raw=expr.to_numpy(); genes=expr.index.tolist()
    Xz=zscore_by_gene(X_raw)

    embeddings={}
    embeddings["RAW(z)"]=Xz
    pca=PCA(n_components=PCA_VARIANCE, random_state=1)
    embeddings["PCA"]=pca.fit_transform(Xz)

    # Optional SOM baseline
    try:
        from minisom import MiniSom
        som_side=20
        som=MiniSom(som_side,som_side,input_len=Xz.shape[1],sigma=1.0,learning_rate=0.5,random_seed=1)
        som.random_weights_init(Xz); som.train_random(Xz,1000)
        codebook=np.array([w for row in som.get_weights() for w in row])
        bmu_idx=cdist(Xz, codebook).argmin(axis=1)
        embeddings["SOM"]=codebook[bmu_idx]
    except Exception as e:
        print("SOM skipped:",e)

    rows=[]
    for name,Xrep in embeddings.items():
        Z=linkage(Xrep, method=LINKAGE_METHOD, metric=DIST_FOR_LINKAGE)
        ccc,_=cophenet(Z,pdist(Xrep,metric=DIST_FOR_LINKAGE))
        for k in K_LIST:
            labs=fcluster(Z,t=k,criterion="maxclust")
            if len(np.unique(labs))>1:
                sil=silhouette_score(Xrep,labs)
                dbi=davies_bouldin_score(Xrep,labs)
                ch=calinski_harabasz_score(Xrep,labs)
                dnn=dunn_index(Xrep,labs); sdb=s_dbw_index(Xrep,labs)
            else:
                sil=dbi=ch=dnn=sdb=np.nan
            mean_ari,std_ari=bootstrap_stability(Xrep,labs)
            cocluster,compact_gain=biomarker_tests(Xrep,labs,genes,AD_BIOMARKERS)
            rows.append({
                "Method":name,"k":k,"CCC":ccc,"Silhouette":sil,"DBI":dbi,"CH":ch,
                "Dunn":dnn,"S_Dbw":sdb,"Stability_meanARI":mean_ari,"Stability_stdARI":std_ari,
                "Biomarker_coClust":cocluster,"Biomarker_compactGain":compact_gain
            })
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR,"baseline_metrics.csv"),index=False)
    print("Done. Results saved to",OUT_DIR)
