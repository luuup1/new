# -*- coding: utf-8 -*-
"""把降维(dimred) cold 三算法 × 3 seed 训练结果导出为 data/ 下 7 个 JSON。

对齐 export_data.py 的 native 口径，替换 8~14 的 pending 占位：

  8  ppo  dimred reward   9  sac  dimred reward   10 ddqn dimred reward
  11 dimred max_load（三算法汇总 eval_best = min(eval_eff_peak)）
  12 ppo  dimred delay    13 sac  dimred delay    14 ddqn dimred delay

数据来源（各目录 training_curves.json）：
  - reward: ep_reward（= -effective_peak_load）
  - max_load: min(eval_eff_peak)（训练全程最佳贪婪评估, lower=better）
  - delay: ep_delay（= average_delay_ms），dimred 三算法训练曲线均有 ep_delay 字段，
    无需像 native PPO/SAC 那样用 best checkpoint 贪婪评估补录。

只读源文件，不修改任何 checkpoint / json。
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")

SEEDS = ["42", "123", "2024"]

# dimred 目录（对齐 native 命名 + _dimred 后缀）
ALGO_DIRS = {
    "ppo": [
        "checkpoints_ppo_cold_dimred_seed42",
        "checkpoints_ppo_cold_dimred_seed123",
        "checkpoints_ppo_cold_dimred_seed2024",
    ],
    "sac": [
        "checkpoints_sac_dimred_seed42",
        "checkpoints_sac_dimred_seed123",
        "checkpoints_sac_dimred_seed2024",
    ],
    "ddqn": [
        "checkpoints_ddqn_dimred_seed42",
        "checkpoints_ddqn_dimred_seed123",
        "checkpoints_ddqn_dimred_seed2024",
    ],
}

ALGO_CN = {"ppo": "PPO", "sac": "SAC", "ddqn": "DDQN"}


def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        with open(path, encoding="gbk") as f:
            return json.load(f)


def get_curve(curves_dir):
    p = os.path.join(HERE, curves_dir, "training_curves.json")
    if not os.path.exists(p):
        return None
    return load_json(p)


def write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"  wrote {os.path.basename(path)}")


def base_meta(algo=None, dimred=True):
    m = {
        "dim_reduction": True,
        "action_mode": "dimred",
        "period_mode": "simple",
        "order_mode": "random",
        "flow_count": 50,
        "reward_mode": "load_balance",
    }
    if algo is not None:
        m["algorithm"] = ALGO_CN.get(algo, algo)
    return m


def export_reward(algo):
    seeds = {}
    for i, d in enumerate(ALGO_DIRS[algo]):
        seed = SEEDS[i]
        c = get_curve(d)
        if c is None or "ep_reward" not in c:
            seeds[seed] = {"episode": [], "ep_reward": [], "missing": True}
            continue
        seeds[seed] = {
            "episode": c.get("episode", []),
            "ep_reward": c.get("ep_reward", []),
        }
    meta = base_meta(algo)
    meta["metric"] = "ep_reward (= -effective_peak_load)"
    return {"meta": meta, "seeds": seeds}


def export_max_load():
    algos = {}
    for algo in ALGO_DIRS:
        per_seed = {}
        notes = []
        for i, d in enumerate(ALGO_DIRS[algo]):
            seed = SEEDS[i]
            c = get_curve(d)
            if c is None or "eval_eff_peak" not in c:
                per_seed[seed] = None
                notes.append(f"{algo}/seed{seed}: 无 eval_eff_peak")
                continue
            ee = [x for x in c["eval_eff_peak"] if x is not None]
            best = round(float(min(ee)), 4) if ee else None
            per_seed[seed] = best
        algos[ALGO_CN[algo]] = {"eval_best_per_seed": per_seed, "note": notes or None}
    meta = base_meta()
    meta["metric"] = "eval_best = min(eval_eff_peak)  (训练全程最佳贪婪评估, lower=better)"
    return {"meta": meta, "algorithms": algos}


def export_delay(algo):
    seeds = {}
    for i, d in enumerate(ALGO_DIRS[algo]):
        seed = SEEDS[i]
        c = get_curve(d)
        if c is None:
            seeds[seed] = {"episode": [], "ep_delay": None, "missing": True}
            continue
        if "ep_delay" not in c:
            seeds[seed] = {"episode": c.get("episode", []), "ep_delay": None,
                           "note": "无 ep_delay 字段"}
            continue
        delay = c["ep_delay"]
        valid = [x for x in delay if x is not None]
        if not valid:
            seeds[seed] = {"episode": c.get("episode", []), "ep_delay": None,
                           "note": "ep_delay 全为 null"}
            continue
        seeds[seed] = {"episode": c.get("episode", []), "ep_delay": delay,
                       "delay_mean_ms": round(sum(valid) / len(valid), 4),
                       "delay_min_ms": round(min(valid), 4),
                       "delay_max_ms": round(max(valid), 4)}
    meta = base_meta(algo)
    meta["metric"] = "ep_delay = average_delay_ms (每 episode 调度平均时延, 单位 ms)"
    return {"meta": meta, "seeds": seeds}


def main():
    os.makedirs(OUT, exist_ok=True)
    print("=== 导出降维(dimred)数据 ===")

    # 8/9/10 奖励曲线
    for i, algo in enumerate(["ppo", "sac", "ddqn"], start=8):
        write(os.path.join(OUT, f"{i}_{algo}_dimred_reward.json"), export_reward(algo))

    # 11 最大负载汇总
    write(os.path.join(OUT, "11_dimred_max_load.json"), export_max_load())

    # 12/13/14 时延
    for i, algo in enumerate(["ppo", "sac", "ddqn"], start=12):
        write(os.path.join(OUT, f"{i}_{algo}_dimred_delay.json"), export_delay(algo))

    print("\n完成。dimred 7 个 JSON 已写入 data/ 目录。")


if __name__ == "__main__":
    main()
