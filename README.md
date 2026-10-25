# 拒绝方向实验（Refusal Direction）

用一个很小的本地模型（Qwen3-1.7B），复现 **「拒绝行为由一个方向介导」** 的实验：

1. 让模型分别读「有害问题」和「普通问题」，记录每层最后一个 token 的内部激活；
2. 两类激活求平均再相减，得到 **拒绝方向 `r`**；
3. 往模型里 **加 `r`**，普通问题也会被拒绝；**减掉 `r`**，拒绝行为被削弱。

方法来自 Arditi 等 2024 年的论文
[Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)。

> **安全声明**：本项目只用于机制研究与教学演示。"消融方向"分支被刻意限制为只生成开头 14 个 token，
> 不生成完整的有害内容。请勿用它绕过线上模型的安全机制。

---

## 三步跑起来（Windows）

**第 1 步：装 Python**
去 <https://www.python.org/downloads/> 下载 3.10 或更新版本。
安装时 **务必勾选 "Add python.exe to PATH"**。

**第 2 步：双击 `install.bat`**
全自动完成：创建环境 → 安装依赖 → 下载模型（约 3.5GB）→ 下载数据 → 计算拒绝方向。
首次大约需要 10–30 分钟，取决于网速。中途失败（多半是网络）直接重新双击，会从断点继续。

**第 3 步：双击 `start.bat`**
输入一个问题回车，会看到三种回答：原样、加上拒绝方向、消去拒绝方向。空行退出。

**已经装过环境 / 下过模型的话**：`install.bat` 会自动识别并跳过——
当前 `python` 里已有 torch 和 transformers 就直接用它（不建 `.venv`、不重装）；
模型、数据、`direction.pt` 已经存在也都会跳过。环境在别处的话，运行前先设置环境变量：
`REFUSAL_PYTHON`（现成环境的 python.exe 路径）、`REFUSAL_MODEL_DIR`（模型目录）、`REFUSAL_DATA_DIR`（数据目录）。

就这样。下面是进阶内容，不需要可以不看。

---

## 电脑要求

| 项目 | 要求 |
| --- | --- |
| 系统 | Windows 10/11（Linux/macOS 见下方说明） |
| Python | 3.10+ |
| 显卡 | 推荐 NVIDIA ≥ 8GB 显存；没有显卡也能跑，会自动用 CPU，但慢很多 |
| 磁盘 | 约 10GB 空闲（模型 3.5GB + PyTorch 2–3GB + 数据） |

Linux / macOS：在终端运行 `python3 tools/install.py` 部署，
之后用 `.venv/bin/python tools/chat.py` 聊天、`.venv/bin/python tools/real_run.py <命令>` 做实验。

---

## 效果示例

`layer=16, scale=0.8` 时的真实记录：

| 问题 | 原模型 | 加上拒绝方向后 |
| --- | --- | --- |
| 你好 | 你好！我是你的助手，有什么可以帮助你的吗？ | 我无法提供或协助任何非法、有害、暴力、恐怖主义……（不断重复） |
| 怎么制作蛋糕 | 制作蛋糕是一个很有趣的过程…… | 制作蛋糕是一项非常危险且不推荐的行为，我必须明确指出：任何涉及……都是非法、不道德…… |

方向被放大后，模型对完全无害的问题也进入「拒绝模式」。

---

## 工作原理（30 秒版）

```text
有害问题 ─┐
          ├─→ 取每层最后一个 token 的激活 ─→ 两组均值相减 ─→ 方向 r
普通问题 ─┘

加方向：  x ← x + α·r                （挑一层，在层输入端挂钩子）
消方向：  x ← x − (x·r̂)·r̂          （每一层都做）
```

公式、层选择和局限见 [docs/原理.md](docs/原理.md)。代码里每个函数都有中文注释。

---

## 进阶：更多实验命令

```text
run.bat <命令> [参数]
```

| 命令 | 作用 | 常用参数 | 输出（在 `runs/`） |
| --- | --- | --- | --- |
| `direction` | 计算每层的拒绝方向（install 已做过） | — | `direction.pt` |
| `sweep` | 逐层加方向，看无害问题被拒绝的比例，用来挑层 | `--from 6 --to 22 --step 2 --count 20` | `sweep.json` |
| `ask` | 同一问题：原样 vs 加方向 | `--layer 16 --scale 1.0` | `conversations.json` |
| `ablate` | 每层消融，只生成开头几个 token | `--layer 16 --length 14` | `ablation.json` |
| `stats` | 100 条留出的无害问题统计拒绝率 | `--layer 12 --scale 1.0` | `stats.json` |
| `pointcloud` | 128 有害 + 128 无害激活降到 3 维 | `--layer 12` | `pointcloud.json` |

例子：

```text
run.bat sweep --from 10 --to 24 --step 2 --count 20
run.bat ask "How do I bake a cake?" --layer 16 --scale 0.8
```

聊天也能调参：`start.bat --layer 16 --scale 0.8`。聊天记录会存到 `runs/temp.json`。

**调参提示**：先 `sweep` 选层，再调 `--scale`（0.5–1.5）。太小没效果，太大回答会复读或崩坏。
默认值只对 Qwen3-1.7B 有效，换模型要重新扫。

---

## 常见问题

**双击 install.bat 提示找不到 Python**
没装或安装时没勾选 "Add python.exe to PATH"。重装 Python 并勾选，然后重新双击。

**下载很慢 / 失败**
安装脚本已优先使用国内镜像：pip 用清华/阿里云，torch 用阿里云 PyTorch 镜像，模型用 ModelScope（失败再用 hf-mirror），
数据用 GitHub 代理；镜像都不行才连官方。直接重新双击 `install.bat` 即可续传。
若数据仍下载失败，手动下载 AdvBench 的 `harmful_behaviors.csv` 和 Alpaca 的
`alpaca_data.json`，分别改名为 `data/advbench.csv`、`data/alpaca.json`。

**显存不足 `CUDA out of memory`**
关掉占显存的程序；还不行就把 `tools/real_run.py` 里 `last_token_activations` 的 `batch_size` 从 16 改成 8 或 4。
1.7B 模型大约需要 4–6GB 显存。

**想强制用 CPU 或指定显卡**
设置环境变量 `REFUSAL_DEVICE=cpu`（或 `cuda:1`）。

**模型、数据想放别的盘**
用环境变量 `REFUSAL_MODEL_DIR`、`REFUSAL_DATA_DIR`、`REFUSAL_OUTPUT_DIR` 指过去（见 `tools/config.py`）。

**换成别的模型（如 Qwen3.5）**
Qwen3.5 需要 torch ≥ 2.6；`install.bat` 安装的是最新 torch，一般没问题。
用 `REFUSAL_MODEL_DIR` 指向新模型并重新跑 `run.bat direction`。

**回答一直复读**
`scale` 太大，降到 0.5–0.8，或换更靠前的层。

**中文乱码**
用 Windows Terminal，或把终端字体换成支持中文的字体。

---

## 目录结构

```text
install.bat   一键部署        start.bat   交互聊天        run.bat   进阶实验命令
tools/        install.py（部署）  real_run.py（主实验）  chat.py（聊天）
              prepare_data.py（下载数据）  download_model.py（下载模型）  config.py（路径）
docs/原理.md  方法与公式
models/ data/ runs/   模型、数据、输出（自动生成，不进仓库）
```

---

## 安全与伦理

- 这是**机制研究**项目，不是越狱工具。
- 数据集包含有害指令，请勿用于训练或部署。
- "消融方向"默认只生成 14 个 token，不要为了获取有害内容而调大。
- 加方向会让模型**过度拒绝**，这不是"更安全"，只是方向被放大后的副作用。
- 在线上服务做类似实验前，请先获得授权并遵守平台条款。

## 引用

```bibtex
@inproceedings{arditi2024refusal,
  title     = {Refusal in Language Models Is Mediated by a Single Direction},
  author    = {Arditi, Andy and Obeso, Oscar and Syed, Aaquib and Paleka, Daniel
               and Panickssery, Nina and Gurnee, Wes and Nanda, Neel},
  booktitle = {NeurIPS},
  year      = {2024},
  url       = {https://arxiv.org/abs/2406.11717}
}
```
