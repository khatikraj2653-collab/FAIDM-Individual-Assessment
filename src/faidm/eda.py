import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from .utils import save_fig
from .data import make_target_binary_merge12

def run_eda(df: pd.DataFrame, figures_dir: Path, tables_dir: Path) -> None:
    vc3 = df["Diabetes_012"].value_counts(dropna=False).sort_index()
    vc3.to_csv(tables_dir / "target_distribution_original_012.csv")

    plt.figure()
    vc3.plot(kind="bar")
    plt.title("Diabetes_012 distribution (0/1/2)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    save_fig(figures_dir / "eda_target_distribution_original.png")

  
    y = make_target_binary_merge12(df)
    vc2 = y.value_counts().sort_index()
    vc2.to_csv(tables_dir / "target_distribution_binary_merge12.csv")

    plt.figure()
    vc2.plot(kind="bar")
    plt.title("Binary target distribution (merge 0+1 vs 2)")
    plt.xlabel("Class (0=non-diabetic, 1=diabetic)")
    plt.ylabel("Count")
    save_fig(figures_dir / "eda_target_distribution_binary_merge12.png")

    
    for col in ["BMI", "MentHlth", "PhysHlth"]:
        plt.figure()
        df[col].hist(bins=40)
        plt.title(f"Distribution: {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        save_fig(figures_dir / f"eda_hist_{col}.png")
