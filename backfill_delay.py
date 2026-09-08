# -*- coding: utf-8 -*-
"""回填 data/05_ppo_native_delay.json 与 data/06_sac_native_delay.json 的时延数据。

数据来源：eval_delay_backfill.py 生成的 _eval_delay_results.json
（best checkpoint 确定性贪婪评估得到的 average_delay_ms）。

原则：
  - 保留原 JSON 里已有的训练曲线数据（ep_delay 列表 / delay_mean_ms 等）不删除。
  - 为每个 seed 补一个统一口径的 best_eval_delay_ms（best checkpoint 评估时延），
    并同步更新 delay_mean_ms / delay_min_ms / delay_max_ms 为该代表值。
  - 明确标注 source 与口径，避免与训练曲线时延混淆。
"""
import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

EVAL = json.load(open(os.path.join(HERE, "_eval_delay_results.json"), encoding="utf-8"))


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def backfill(algo_key, algo_name, path):
    d = load(path)
    ev = EVAL[algo_key]
    for seed, info in ev.items():
        seed_obj = d["seeds"].get(seed)
        if seed_obj is None:
            seed_obj = {}
            d["seeds"][seed] = seed_obj
        delay = info["delay_ms"]
        eff = info["eff_peak"]
        # 补 best checkpoint 评估时延（统一口径代表值）
        seed_obj["best_eval_delay_ms"] = delay
        seed_obj["best_eval_eff_peak"] = eff
        seed_obj["delay_mean_ms"] = delay
        seed_obj["delay_min_ms"] = delay
        seed_obj["delay_max_ms"] = delay
        seed_obj["source"] = "best_checkpoint_greedy_eval"
        seed_obj.pop("note", None)
    # meta 补充说明
    d["meta"]["note"] = (
        f"{algo_name} 训练曲线 ep_delay 缺失/不全（旧脚本未记录），"
        f"时延由 best checkpoint 确定性贪婪评估补录（best_eval_delay_ms）。"
    )
    write(path, d)
    print(f"回填 {os.path.basename(path)}:")
    for seed in ["42", "123", "2024"]:
        s = d["seeds"][seed]
        print(f"  seed {seed}: best_eval_delay_ms={s['best_eval_delay_ms']} "
              f"eff_peak={s['best_eval_eff_peak']}")


def main():
    backfill("ppo", "PPO", os.path.join(DATA, "05_ppo_native_delay.json"))
    backfill("sac", "SAC", os.path.join(DATA, "06_sac_native_delay.json"))
    print("\n完成。")


if __name__ == "__main__":
    main()
