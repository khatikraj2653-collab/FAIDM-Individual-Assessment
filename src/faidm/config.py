from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    project_root: Path = Path(__file__).resolve().parents[2]

    data_raw: Path = project_root / "data" / "raw" / "CDC Diabetes Dataset.csv"

    outputs_dir: Path = project_root / "outputs"
    figures_dir: Path = outputs_dir / "figures"
    tables_dir: Path = outputs_dir / "tables"
    models_dir: Path = outputs_dir / "models"
    predictions_dir: Path = outputs_dir / "predictions"

    random_state: int = 42
    test_size: float = 0.2

    
    clustering_sample_n: int = 10000        
    run_dbscan: bool = True                
    enable_grid_search: bool = False     

    
    kmin: int = 2
    kmax: int = 6


    dbscan_eps: float = 1.2
    dbscan_min_samples: int = 25

   
    rf_estimators: int = 200
    hgb_max_iter: int = 300

CFG = Config()
