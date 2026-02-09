from src.faidm.config import CFG
from src.faidm.utils import ensure_dir
from src.faidm.data import load_dataset
from src.faidm.eda import run_eda
from src.faidm.clustering import kmeans_clustering, dbscan_clustering
from src.faidm.modeling_binary import train_and_evaluate_binary_merge12


def main():
    
    for d in [CFG.outputs_dir, CFG.figures_dir, CFG.tables_dir, CFG.models_dir, CFG.predictions_dir]:
        ensure_dir(d)

    print("=== START ===")
    print("Project root:", CFG.project_root)
    print("Dataset path :", CFG.data_raw)
    print("Outputs dir  :", CFG.outputs_dir)
    print("run_dbscan   :", CFG.run_dbscan)
    print("cluster_n    :", CFG.clustering_sample_n)
    print("dbscan eps   :", CFG.dbscan_eps)
    print("dbscan min_s :", CFG.dbscan_min_samples)
    print("-------------")

   
    df = load_dataset(str(CFG.data_raw))
    print("Loaded dataset:", df.shape)

   
    print("\n[EDA] Running EDA...")
    run_eda(df, CFG.figures_dir, CFG.tables_dir)
    print("[EDA] Done. (Check outputs/figures and outputs/tables)")

    
    df_cluster = df.sample(
        n=min(CFG.clustering_sample_n, len(df)),
        random_state=CFG.random_state
    ).copy()
    print("\n[CLUSTER] Clustering sample size:", df_cluster.shape)

   
    print("[CLUSTER] Running KMeans...")
    df_km = kmeans_clustering(
        df_cluster,
        CFG.figures_dir,
        CFG.tables_dir,
        CFG.random_state,
        CFG.kmin,
        CFG.kmax
    )
    km_out = CFG.tables_dir / "data_with_kmeans_clusters_sample.csv"
    df_km.to_csv(km_out, index=False)
    print("[CLUSTER] KMeans done. Saved:", km_out)

   
    if CFG.run_dbscan:
        print("\n[CLUSTER] Running DBSCAN... (this can be slower than KMeans)")
        try:
            df_db = dbscan_clustering(
                df_cluster,
                CFG.figures_dir,
                CFG.tables_dir,
                CFG.random_state,
                CFG.dbscan_eps,
                CFG.dbscan_min_samples
            )
            db_out = CFG.tables_dir / "data_with_dbscan_clusters_sample.csv"
            df_db.to_csv(db_out, index=False)
            print("[CLUSTER] DBSCAN done. Saved:", db_out)

           
            expected = [
                CFG.figures_dir / "cluster_dbscan_pca_scatter.png",
                CFG.tables_dir / "dbscan_summary.csv",
                CFG.tables_dir / "dbscan_cluster_profile_means.csv",
                CFG.tables_dir / "dbscan_cluster_diabetes_rate_binary_merge12.csv",
            ]
            print("[CLUSTER] DBSCAN expected files check:")
            for p in expected:
                print("  -", p.name, "=>", "OK" if p.exists() else "MISSING")

        except Exception as e:
            print("[CLUSTER] DBSCAN ERROR:", repr(e))
            print("[CLUSTER] DBSCAN failed. Try these fixes:")
            print("  1) Reduce clustering_sample_n to 10000")
            print("  2) Increase dbscan_min_samples to 50 or 100")
            print("  3) Keep dbscan_eps ~ 1.0 to 1.5")
    else:
        print("\n[CLUSTER] DBSCAN skipped (run_dbscan=False).")

    
    print("\n[CLASSIFY] Training and evaluating binary_merge12 models...")
    summary = train_and_evaluate_binary_merge12(
        df=df,
        figures_dir=CFG.figures_dir,
        tables_dir=CFG.tables_dir,
        models_dir=CFG.models_dir,
        predictions_dir=CFG.predictions_dir,
        random_state=CFG.random_state,
        test_size=CFG.test_size,
        rf_estimators=CFG.rf_estimators,
        hgb_max_iter=CFG.hgb_max_iter,
        enable_grid_search=CFG.enable_grid_search
    )

 
    if "pr_auc" in summary.columns:
        cols = [c for c in ["model", "pr_auc", "roc_auc", "recall_pos", "f1_pos"] if c in summary.columns]
        print("\nTop models by PR-AUC:")
        print(summary.sort_values("pr_auc", ascending=False)[cols].head(10))
    else:
        print("\nModel summary:")
        print(summary.head())

    print("\n=== DONE ===")
    print("All outputs saved in:", CFG.outputs_dir)


if __name__ == "__main__":
    main()
