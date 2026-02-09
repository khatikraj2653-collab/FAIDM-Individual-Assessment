from pathlib import Path
import matplotlib.pyplot as plt

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
