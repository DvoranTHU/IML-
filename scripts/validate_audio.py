import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.audio.validate import run_validation
from src.config import load_config, resolve_path

# 全量检查现有数据集是否有问题
def main():
    cfg = load_config()
    meta_dir = resolve_path(cfg["outputs"]["metadata"])
    meta_path = meta_dir / "dataset.csv"
    meta_dir.mkdir(parents=True, exist_ok=True)

    issues_df = run_validation(meta_path, cfg=cfg, project_root=PROJECT_ROOT)
    issues_path = meta_dir / "audio_issues.csv"
    issues_df.to_csv(issues_path, index=False)

if __name__ == "__main__":
    main()
