"""下载 Qwen3-1.7B 到 MODEL_DIR（约 3.5GB）。

先试 ModelScope（国内快），失败再试 Hugging Face（自动带上国内镜像）。
已经下载好的会直接跳过。
"""
import os
import subprocess
import sys

from config import MODEL_DIR

REPO = "Qwen/Qwen3-1.7B"

# 国内镜像优先；用户自己设过就不覆盖。必须在导入 huggingface_hub 之前设置。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def already_there():
    return (MODEL_DIR / "config.json").exists() and any(MODEL_DIR.glob("*.safetensors"))


def from_modelscope():
    try:
        import modelscope  # noqa: F401
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "modelscope",
             "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"],
            check=True,
        )
    from modelscope import snapshot_download

    snapshot_download(REPO, local_dir=str(MODEL_DIR))


def from_huggingface():
    from huggingface_hub import snapshot_download

    snapshot_download(REPO, local_dir=str(MODEL_DIR))


def main():
    if already_there():
        print(f"[skip] 模型已存在：{MODEL_DIR}")
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, fetch in (("ModelScope", from_modelscope), ("Hugging Face 镜像", from_huggingface)):
        print(f"[get ] 从 {name} 下载 {REPO} ...")
        try:
            fetch()
        except Exception as exc:
            print(f"[fail] {name}：{exc}")
            continue
        if already_there():
            print(f"[ok  ] 模型已就绪：{MODEL_DIR}")
            return

    sys.exit(
        "模型下载失败。请检查网络后重新运行 install.bat，\n"
        f"或手动下载 {REPO} 放进：{MODEL_DIR}"
    )


if __name__ == "__main__":
    main()
