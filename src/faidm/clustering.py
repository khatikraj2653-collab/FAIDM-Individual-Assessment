import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

from .features import get_feature_groups, build_preprocessor
from .data import make_target_binary_merge12
from .utils import save_fig

def _prep_X(df: pd.DataFrame):
    X = df.drop(columns=["Diabetes_012"]).copy()
    numeric, ordinal, binary = get_feature_groups(df)
    prep = build_preprocessor(numeric, ordinal, binary)
    Xp = prep.fit_transform(X)
    return X, Xp

def kmeans_clustering(
    df: pd.DataFrame,
    figures_dir: Path,
    tables_dir: Path,
    random_state: int,
    kmin: int,
    kmax: int
) -> pd.DataFrame:
    _, Xp = _prep_X(df)

    ks = list(range(kmin, kmax + 1))
    sils, dbs = [], []

    for k in ks:
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = km.fit_predict(Xp)
        sils.append(silhouette_score(Xp, labels))
        dbs.append(davies_bouldin_score(Xp, labels))

    best_k = ks[int(np.argmax(sils))]

    pd.DataFrame({"k": ks, "silhouette": sils, "davies_bouldin": dbs}).to_csv(
        tables_dir / "kmeans_model_selection.csv", index=False
    )

    plt.figure()
    plt.plot(ks, sils, marker="o")
    plt.title("KMeans: Silhouette vs K")
    plt.xlabel("K")
    plt.ylabel("Silhouette")
    save_fig(figures_dir / "cluster_kmeans_silhouette_vs_k.png")

    plt.figure()
    plt.plot(ks, dbs, marker="o")
    plt.title("KMeans: Davies-Bouldin vs K (lower better)")
    plt.xlabel("K")
    plt.ylabel("Davies-Bouldin")
    save_fig(figures_dir / "cluster_kmeans_davies_bouldin_vs_k.png")

    km = KMeans(n_clusters=best_k, random_state=random_state, n_init="auto")
    labels = km.fit_predict(Xp)

    out = df.copy()
    out["cluster_kmeans"] = labels

    
    pca = PCA(n_components=2, random_state=random_state)
    X2 = pca.fit_transform(Xp)
    plt.figure(figsize=(7, 6))
    plt.scatter(X2[:, 0], X2[:, 1], s=4, alpha=0.6, c=labels)
    plt.title(f"KMeans via PCA (K={best_k})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    save_fig(figures_dir / "cluster_kmeans_pca_scatter.png")

    out.groupby("cluster_kmeans").mean(numeric_only=True).to_csv(tables_dir / "kmeans_cluster_profile_means.csv")

    ybin = make_target_binary_merge12(df)
    rate = pd.DataFrame({"cluster": labels, "diabetes": ybin}).groupby("cluster")["diabetes"].mean().sort_values(ascending=False)
    rate.to_csv(tables_dir / "kmeans_cluster_diabetes_rate_binary_merge12.csv")

    return out

def dbscan_clustering(
    df: pd.DataFrame,
    figures_dir: Path,
    tables_dir: Path,
    random_state: int,
    eps: float,
    min_samples: int
) -> pd.DataFrame:
    _, Xp = _prep_X(df)

    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = db.fit_predict(Xp)

    out = df.copy()
    out["cluster_dbscan"] = labels

    n_noise = int((labels == -1).sum())
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    pd.DataFrame([{
        "eps": eps,
        "min_samples": min_samples,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_pct": float(n_noise) / len(labels)
    }]).to_csv(tables_dir / "dbscan_summary.csv", index=False)

    
    pca = PCA(n_components=2, random_state=random_state)
    X2 = pca.fit_transform(Xp)
    plt.figure(figsize=(7, 6))
    plt.scatter(X2[:, 0], X2[:, 1], s=4, alpha=0.6, c=labels)
    plt.title(f"DBSCAN via PCA (eps={eps}, min_samples={min_samples}, -1=noise)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    save_fig(figures_dir / "cluster_dbscan_pca_scatter.png")

    out.groupby("cluster_dbscan").mean(numeric_only=True).to_csv(tables_dir / "dbscan_cluster_profile_means.csv")

    ybin = make_target_binary_merge12(df)
    rate = pd.DataFrame({"cluster": labels, "diabetes": ybin}).groupby("cluster")["diabetes"].mean().sort_values(ascending=False)
    rate.to_csv(tables_dir / "dbscan_cluster_diabetes_rate_binary_merge12.csv")

    return out
