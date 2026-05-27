import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, resolve_path
from src.datasets import (
    build_crema_d_metadata,
    build_emodb_metadata,
    build_ravdess_metadata,
)

# debug用
def summarize(name, df, file, actor_col="actor_id"):
    file.write(f"{name}\n")
    file.write(f"样本数: {len(df)}\n")
    file.write(f"演员人数: {df[actor_col].nunique()}\n")
    file.write("情感分布:\n")
    file.write(df["emotion"].value_counts().sort_index().to_string())
    file.write("\n\n")


def main():
    # 加载配置
    cfg = load_config()
    out_dir = resolve_path(cfg["outputs"]["metadata"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # 构建三个数据集元数据
    ravdess = build_ravdess_metadata(resolve_path(cfg["data"]["ravdess"]), PROJECT_ROOT)
    crema_d = build_crema_d_metadata(resolve_path(cfg["data"]["crema_d"]), PROJECT_ROOT)
    emodb = build_emodb_metadata(resolve_path(cfg["data"]["emodb"]), PROJECT_ROOT)

    all_df = pd.concat([ravdess, crema_d, emodb], ignore_index=True)
    out_path = out_dir / "all.csv"
    all_df.to_csv(out_path, index=False)

    debug_path = out_dir / "debug.txt"
    with debug_path.open("w", encoding="utf-8") as f:
        summarize("RAVDESS", ravdess, f)
        summarize("CREMA-D", crema_d, f)
        summarize("EMODB", emodb, f)

if __name__ == "__main__":
    main()
