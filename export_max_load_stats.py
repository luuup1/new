# -*- coding: utf-8 -*-
"""从 checkpoint 的 training_curves.json 重新导出最大负载汇总（三种口径）。

口径（对齐 EXPERIMENT_PROTOCOL.md + 用户对"最大负载"的诉求）：
  - 峰值负载 = max(所有 cell 归一化负载)，单次调度内部取 max（代码已实现）。
  - 跨多次评估，对每个 seed 的 eval_eff_peak 序列分别统计三种口径：
      mean  = 训练全程 eval_eff_peak 的均值 —— 典型水平（主口径）
      best  = min(eval_eff_peak)          —— 能力上限（最不堵的一次，偏乐观）
      worst = max(eval_eff_peak)          —— 最坏情况（最堵的一次，偏保守）
  - 跨 3 seed：每个 seed 一个代表性数值，再求 mean ± std（ddof=1）。

输出：
  - data/04_native_max_load.json  (PPO / SAC / TD3, native)
  - data/11_dimred_max_load.json  (PPO / SAC / TD3, dimred)

DDQN 已搁置，不纳入。

只读源文件，不修改任何 checkpoint。
"""
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")

SEEDS = ["42", "123", "2024"]

# 每个算法 × action_mode 对应的 checkpoint 目录（与 seed 顺序一一对应）
SOURCE_DIRS = {
    ("PPO", "native"): [
        "checkpoints_ppo_cold_seed42",
        "checkpoints_ppo_cold_seed123",
        "checkpoints_ppo_cold_seed2024",
    ],
    ("SAC", "native"): [
        "checkpoints_sac_seed42_0716",
        "checkpoints_sac_seed123_0716",
        "checkpoints_sac_seed2024_0716",
    ],
    ("TD3", "native"): [
        "checkpoints_td3_native_s42",
        "checkpoints_td3_native_s123",
        "checkpoints_td3_native_s2024",
    ],
    ("PPO", "dimred"): [
        "checkpoints_ppo_cold_dimred_seed42",
        "checkpoints_ppo_cold_dimred_seed123",
        "checkpoints_ppo_cold_dimred_seed2024",
    ],
    ("SAC", "dimred"): [
        "checkpoints_sac_dimred_fixed_seed42",
        "checkpoints_sac_dimred_fixed_seed123",
        "checkpoints_sac_dimred_fixed_seed2024",
    ],
    ("TD3", "dimred"): [
        "checkpoints_td3_dimred_seed42",
        "checkpoints_td3_dimred_seed123",
        "checkpoints_td3_dimred_seed2024",
    ],
}

ALGORITHMS = ["PPO", "SAC", "TD3"]


def load_curves(d):
    p = os.path.join(HERE, d, "training_curves.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def seed_stats(curves_dir, seed):
    """返回该 seed 的 {mean, best, worst, n_eval}，缺数据则 None。"""
    c = load_curves(curves_dir)
    if c is None or "eval_eff_peak" not in c:
        return None
    ee = [x for x in c["eval_eff_peak"] if x is not None]
    if not ee:
        return None
    return {
        "mean": round(statistics.mean(ee), 4),
        "best": round(min(ee), 4),
        "worst": round(max(ee), 4),
        "n_eval": len(ee),
    }


def build(action_mode):
    algos = {}
    for algo in ALGORITHMS:
        dirs = SOURCE_DIRS[(algo, action_mode)]
        mean_per_seed, best_per_seed, worst_per_seed = {}, {}, {}
        notes = []
        for seed, d in zip(SEEDS, dirs):
            s = seed_stats(d, seed)
            if s is None:
                mean_per_seed[seed] = None
                best_per_seed[seed] = None
                worst_per_seed[seed] = None
                notes.append(f"{algo}/seed{seed}: 无 eval_eff_peak 数据")
                continue
            mean_per_seed[seed] = s["mean"]
            best_per_seed[seed] = s["best"]
            worst_per_seed[seed] = s["worst"]
        algos[algo] = {
            "eval_mean_per_seed": mean_per_seed,
            "eval_best_per_seed": best_per_seed,
            "eval_worst_per_seed": worst_per_seed,
            "note": "; ".join(notes) if notes else None,
        }
    meta = {
        "dim_reduction": action_mode == "dimred",
        "action_mode": action_mode,
        "period_mode": "simple",
        "order_mode": "random",
        "flow_count": 50,
        "reward_mode": "load_balance",
        "metric": (
            "峰值负载 = max(cell 归一化负载)。跨多次评估对每 seed 的 eval_eff_peak 序列统计三种口径："
            "mean=全程均值(典型水平,主口径)；best=min(能力上限,偏乐观)；worst=max(最坏情况,偏保守)。"
            "跨 3 seed 求 mean±std(ddof=1)。"
        ),
        "note": "DDQN 已搁置。TD3 native 源=checkpoints_td3_native_s*; TD3 dimred 源=checkpoints_td3_dimred_seed*。",
    }
    return {"meta": meta, "algorithms": algos}


def main():
    os.makedirs(OUT, exist_ok=True)
    for action_mode, fname in [
        ("native", "04_native_max_load.json"),
        ("dimred", "11_dimred_max_load.json"),
    ]:
        obj = build(action_mode)
        path = os.path.join(OUT, fname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        print(f"wrote {fname}")

    # 打印汇总（mean 口径，主口径）
    print("\n=== 主口径 mean ± std（跨 3 seed，越低越好）===")
    for action_mode in ("native", "dimred"):
        obj = build(action_mode)
        print(f"\n[{action_mode}]")
        for algo in ALGORITHMS:
            m = obj["algorithms"][algo]["eval_mean_per_seed"]
            vals = [v for v in m.values() if v is not None]
            if vals:
                mean = statistics.mean(vals)
                std = statistics.stdev(vals) if len(vals) > 1 else 0.0
                print(f"  {algo}: mean={mean:.4f}±{std:.4f}  "
                      f"per_seed={[round(v,4) for v in vals]}")
    print("\n=== 参考：best(能力上限) / worst(最坏) ===")
    for action_mode in ("native", "dimred"):
        obj = build(action_mode)
        print(f"\n[{action_mode}]")
        for algo in ALGORITHMS:
            b = obj["algorithms"][algo]["eval_best_per_seed"]
            w = obj["algorithms"][algo]["eval_worst_per_seed"]
            bv = [v for v in b.values() if v is not None]
            wv = [v for v in w.values() if v is not None]
            bmean = statistics.mean(bv) if bv else 0
            wmean = statistics.mean(wv) if wv else 0
            print(f"  {algo}: best_mean={bmean:.4f}  worst_mean={wmean:.4f}")


if __name__ == "__main__":
    main()
