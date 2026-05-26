from pathlib import Path
import yaml

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 配置文件路径
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"

# 加载配置
def load_config(path=None):
    if path is not None:
        config_path = Path(path)
    else:
        config_path = DEFAULT_CONFIG_PATH
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)

# 解析路径
def resolve_path(relative, root=None):
    if root is not None:
        base = root
    else:
        base = PROJECT_ROOT
    return (base / relative).resolve()
