"""一键部署：建虚拟环境 → 装依赖 → 下载模型和数据 → 计算拒绝方向。

用法：双击 install.bat，或  python tools/install.py
每一步都会检查是否已完成，中途失败后直接重跑即可接着来。
"""
import re
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv"
PY = VENV / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def step(text):
    print(f"\n{'=' * 60}\n{text}\n{'=' * 60}", flush=True)


def run(*args):
    subprocess.run([str(a) for a in args], cwd=ROOT, check=True)


PYPI_MIRRORS = [
    "https://pypi.tuna.tsinghua.edu.cn/simple",
    "https://mirrors.aliyun.com/pypi/simple",
    None,  # 官方源兜底
]


def pip_install(*args):
    """依次用清华、阿里云、官方源安装，哪个先成功用哪个。"""
    for mirror in PYPI_MIRRORS:
        cmd = [PY, "-m", "pip", "install", *args] + (["-i", mirror] if mirror else [])
        print(f"使用源：{mirror or 'PyPI 官方'}", flush=True)
        if subprocess.run([str(a) for a in cmd], cwd=ROOT).returncode == 0:
            return
    raise subprocess.CalledProcessError(1, ["pip", *map(str, args)])


def cuda_works():
    probe = subprocess.run(
        [str(PY), "-c", "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"]
    )
    return probe.returncode == 0


def install_torch(tag):
    """装 torch：有 N 卡时先试阿里云的 PyTorch 镜像，装完检查 GPU 可用，不行再换官方源。"""
    if tag is None:
        pip_install("torch")
        return
    mirror = f"https://mirrors.aliyun.com/pytorch-wheels/{tag}"
    official = f"https://download.pytorch.org/whl/{tag}"
    attempts = (
        ("阿里云 PyTorch 镜像", ["-f", mirror, "-i", PYPI_MIRRORS[0]]),
        ("PyTorch 官方源", ["--index-url", official]),
    )
    for name, extra in attempts:
        print(f"torch 下载源：{name}", flush=True)
        cmd = [PY, "-m", "pip", "install", "torch", "--force-reinstall", *extra]
        if subprocess.run([str(a) for a in cmd], cwd=ROOT).returncode == 0 and cuda_works():
            return
        print("这个源没装成 GPU 版，换下一个。", flush=True)
    print("[警告] 没能装上 GPU 版 torch，将使用 CPU（很慢）。", flush=True)
    pip_install("torch")


def torch_tag_note():
    """按显卡驱动支持的 CUDA 版本，挑 PyTorch 的下载源；没有 N 卡就装 CPU 版。"""
    try:
        out = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=20
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None, "未检测到 NVIDIA 显卡，安装 CPU 版（能跑，但会慢很多）"

    match = re.search(r"CUDA (?:\w+ )?Version:\s*(\d+)\.(\d+)", out)
    version = (int(match[1]), int(match[2])) if match else (0, 0)
    for need, tag in (((12, 8), "cu128"), ((12, 6), "cu126"), ((12, 4), "cu124")):
        if version >= need:
            return tag, f"检测到 NVIDIA 显卡，安装 GPU 版 torch（{tag}，约 2-3GB）"
    return None, "显卡驱动太旧，安装 CPU 版 torch（想用 GPU 请先更新显卡驱动）"


def torch_ready():
    probe = subprocess.run([str(PY), "-c", "import torch, transformers"], capture_output=True)
    return probe.returncode == 0


def main():
    if sys.version_info < (3, 10):
        sys.exit("需要 Python 3.10 或更新版本，请到 https://www.python.org/downloads/ 安装。")

    step("[1/5] 创建虚拟环境 .venv")
    if PY.exists():
        print("已存在，跳过。")
    else:
        venv.create(VENV, with_pip=True)

    step("[2/5] 安装依赖（torch 较大，请耐心等待）")
    if torch_ready():
        print("已安装，跳过。")
    else:
        tag, note = torch_tag_note()
        print(note)
        install_torch(tag)
        pip_install("-r", str(ROOT / "requirements.txt"))

    step("[3/5] 下载模型 Qwen3-1.7B（约 3.5GB）")
    run(PY, ROOT / "tools" / "download_model.py")

    step("[4/5] 下载数据（AdvBench + Alpaca）")
    run(PY, ROOT / "tools" / "prepare_data.py")

    step("[5/5] 计算拒绝方向（需要几分钟）")
    if (ROOT / "runs" / "direction.pt").exists():
        print("runs/direction.pt 已存在，跳过。")
    else:
        run(PY, ROOT / "tools" / "real_run.py", "direction")

    step("全部完成！双击 start.bat 开始和模型对话。")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(f"\n出错了（{exc.cmd[-1]}）。看上面的报错信息，网络问题的话重新运行一次即可。")
