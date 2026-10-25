"""下载实验用的两份数据，保存成 real_run.py 认识的格式。

  advbench.csv   AdvBench 有害目标（520 条，列名 goal）
  alpaca.json    Alpaca 普通指令（5 万多条，字段 instruction/input/output）

用法：
  python tools/prepare_data.py                # 下到项目里的 data/（或 REFUSAL_DATA_DIR）
  python tools/prepare_data.py --force        # 已存在也重新下载
  python tools/prepare_data.py --data-dir D:/my_data

如果联网下载失败（比如 GitHub 访问不了），可以手动下载这两个文件，
按上面的文件名放进数据目录即可。
"""
import argparse
import csv
import json
import time
import urllib.request
from pathlib import Path

from config import DATA_DIR

# GitHub 国内镜像优先，最后才连官方。
GITHUB_PROXIES = ["https://gh-proxy.com/", "https://ghfast.top/", ""]

ADVBENCH_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
    "main/data/advbench/harmful_behaviors.csv"
)
ALPACA_URL = (
    "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/"
    "main/alpaca_data.json"
)


def read_with_deadline(response, seconds):
    """读完整个响应；总耗时超过 seconds 就放弃，避免卡在龟速镜像上。"""
    deadline = time.monotonic() + seconds
    chunks = []
    while True:
        chunk = response.read(1 << 20)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        if time.monotonic() > deadline:
            raise TimeoutError(f"{seconds} 秒内没下载完")


def download(url, dest, force=False):
    """下载到 dest；已存在且没加 --force 就跳过。"""
    if dest.exists() and not force:
        print(f"[skip] 已存在：{dest}")
        return

    data, errors = None, []
    for proxy in GITHUB_PROXIES:
        full = proxy + url
        print(f"[get ] {full}")
        request = urllib.request.Request(
            full, headers={"User-Agent": "refusal-direction/1.0"}
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                data = read_with_deadline(response, 180)
            if data[:1] == b"<":  # 镜像偶尔返回网页而不是文件
                raise ValueError("返回的是网页，不是数据文件")
            break
        except Exception as exc:  # 网络问题、403、超时等
            data = None
            errors.append(f"{full}: {exc}")
            print(f"[fail] {exc}")
    if data is None:
        raise SystemExit(
            "下载失败：\n" + "\n".join(errors) + "\n\n"
            f"请手动下载后保存为：{dest}\n"
            "（AdvBench 需要 harmful_behaviors.csv，Alpaca 需要 alpaca_data.json）"
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    print(f"[ok  ] {dest} ({len(data) / 1024 / 1024:.1f} MB)")


def validate(data_dir):
    """确认两个文件的结构和 real_run.py 的预期一致。"""
    advbench = data_dir / "advbench.csv"
    alpaca = data_dir / "alpaca.json"

    with open(advbench, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or "goal" not in rows[0]:
        raise SystemExit(f"{advbench} 格式不对：需要带 goal 列的 CSV。")

    with open(alpaca, encoding="utf-8") as f:
        items = json.load(f)
    if not items or not {"instruction", "input"}.issubset(items[0]):
        raise SystemExit(
            f"{alpaca} 格式不对：需要包含 instruction/input 字段的 JSON 数组。"
        )

    usable = sum(1 for d in items if not d["input"].strip())
    print(f"[ok  ] AdvBench：{len(rows)} 条有害目标")
    print(f"[ok  ] Alpaca：{len(items)} 条，其中可直接用作无害问题 {usable} 条")
    if len(rows) < 128 or usable < 228:
        print("[warn] 数量偏少，real_run.py 至少需要 128 条有害、228 条无害。")


def main():
    parser = argparse.ArgumentParser(description="下载 AdvBench + Alpaca")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help=f"保存目录（默认 {DATA_DIR}）",
    )
    parser.add_argument("--force", action="store_true", help="已存在也重新下载")
    args = parser.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    download(ADVBENCH_URL, args.data_dir / "advbench.csv", args.force)
    download(ALPACA_URL, args.data_dir / "alpaca.json", args.force)
    validate(args.data_dir)
    print(f"\n数据已就绪：{args.data_dir}")


if __name__ == "__main__":
    main()
