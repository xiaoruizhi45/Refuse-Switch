"""集中管理路径。所有脚本都从这里取配置。

默认全部放在项目目录里：模型 models/Qwen3-1.7B、数据 data/、输出 runs/。
想放别处，用环境变量覆盖：
  REFUSAL_MODEL_DIR   模型目录
  REFUSAL_DATA_DIR    数据集目录
  REFUSAL_OUTPUT_DIR  输出目录
"""
import os
from pathlib import Path

# tools/ 的上一级就是项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_NAME = "Qwen3-1.7B"


def _dir(env_name, default):
    return Path(os.environ.get(env_name) or default).expanduser()


MODEL_DIR = _dir("REFUSAL_MODEL_DIR", PROJECT_ROOT / "models" / MODEL_NAME)
DATA_DIR = _dir("REFUSAL_DATA_DIR", PROJECT_ROOT / "data")
OUTPUT_DIR = _dir("REFUSAL_OUTPUT_DIR", PROJECT_ROOT / "runs")

# 所有输出（direction.pt / *.json）都放这里。
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
