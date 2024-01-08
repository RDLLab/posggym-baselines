from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
REPO_DIR = PKG_DIR.parent
BASE_RESULTS_DIR = REPO_DIR / "results"

BASE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
