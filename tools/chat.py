# tools/chat.py
"""
【这个脚本在干嘛？】
和 real_run.py 的 `ask` 命令类似：问一个问题，分别看
  1) 不加方向（baseline）
  2) 加上方向（steered）
  3) 消去方向（ablated，用 RemoveDirection 把 r 方向的分量投影掉）
三版回答，并打印这个问题在该层激活沿 r̂ 的投影数值。

问题用 input() 敲进去，结果追加到 runs/temp.json。
双击 start.bat 即可运行；下面是命令行用法。

【怎么用（项目根目录）】
  python tools/chat.py [--layer 16] [--scale 0.8]
                       [--ablate-length 14] [--no-projection]
  或双击 start.bat（参数同样可以跟在后面）

  --layer           加方向/消去方向用哪一层的 r（默认 16）
  --scale           加方向时乘的倍数（默认 0.8）；对消去方向无效，因为内部会归一化
  --ablate-length   消去方向那版只生成多少 token（默认 14，安全起见别太大）
  --no-projection   不打印投影数值（省一次前向，稍微快一点）
"""
import json

import torch

from real_run import (
    AddDirection,
    RemoveDirection,
    DEVICE,
    generate,
    get_arg,
    last_token_activations,
    load_model,
    MODEL_DIR,
    OUTPUT_DIR,
)


def main():
    layer = get_arg("--layer", 16)
    scale = get_arg("--scale", 0.8)
    ablate_length = get_arg("--ablate-length", 14)
    show_projection = "--no-projection" not in __import__("sys").argv

    direction_path = OUTPUT_DIR / "direction.pt"
    if not direction_path.exists():
        raise SystemExit(
            f"找不到 {direction_path}\n请先双击 install.bat，或运行：run.bat direction"
        )

    tokenizer, model, layers = load_model()

    # 只读一次文件，省一次 IO，也省一次警告。
    r_raw = torch.load(direction_path, map_location="cpu", weights_only=True)["r"][layer]

    # 加方向用：原始 r × scale，放到当前设备。
    direction = r_raw.to(DEVICE) * scale

    # 消去方向用：原始 r（不乘 scale）。RemoveDirection 内部会归一化。
    direction_raw = r_raw.to(DEVICE)

    # 算投影用：单位向量，放 CPU。
    # 因为 last_token_activations 返回的激活在 CPU，两边设备必须一致。
    r_hat = (r_raw / r_raw.norm()).float().cpu()

    log_path = OUTPUT_DIR / "temp.json"

    while True:
        question = input("\nQuestion (empty line to exit): ").strip()
        if not question:
            break

        # 1) 原样回答
        baseline = generate(tokenizer, model, question)

        # 2) 加方向回答
        with AddDirection(layers, layer, direction):
            steered = generate(tokenizer, model, question)

        # 3) 消去方向回答：只生成开头几个 token，避免真的生成有害内容。
        with RemoveDirection(model, layers, direction_raw):
            ablated = generate(tokenizer, model, question, ablate_length)

        # 4) 投影数值（可选）：抓这个问题在第 layer 层的激活，
        #    算它沿 r̂ 的分量有多长，以及占总激活范数的比例。
        projection = None
        if show_projection:
            act = last_token_activations(tokenizer, model, [question])[0, layer]  # [H], float32
            proj_len = (act @ r_hat).item()          # 沿 r̂ 的投影长度（带符号）
            act_norm = act.norm().item()             # 激活总长度
            projection = {
                "projection_length": round(proj_len, 4),          # 沿 r̂ 有多长
                "activation_norm": round(act_norm, 4),            # 总长度
                "projection_ratio": round(proj_len / act_norm, 4) if act_norm else 0.0,  # 占比
            }

        # ---- 打印 ----
        print(f"== baseline\n{baseline}\n")
        print(f"== + direction (layer {layer}, scale {scale})\n{steered}\n")
        print(f"== − direction / ablated (layer {layer}, {ablate_length} tokens)\n{ablated}\n")
        if projection:
            print(
                f"== projection along r̂ at layer {layer}: "
                f"len={projection['projection_length']}, "
                f"‖act‖={projection['activation_norm']}, "
                f"ratio={projection['projection_ratio']}"
            )

        # ---- 记录 ----
        history = json.load(open(log_path, encoding="utf-8")) if log_path.exists() else []
        record = {
            "question": question,
            "layer": layer,
            "scale": scale,
            "baseline": baseline,
            "steered": steered,
            "ablated": ablated,
            "ablate_length": ablate_length,
            "model": MODEL_DIR.name,
        }
        if projection:
            record["projection"] = projection
        history.append(record)
        json.dump(history, open(log_path, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()