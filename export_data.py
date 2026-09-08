# -*- coding: utf-8 -*-
"""把现有实验数据整理成 data/ 下 14 个独立 JSON。

口径（与 EXPERIMENT_PROTOCOL.md 一致）：
  - period_mode = simple, order_mode = random, flow_count = 50
  - reward_mode = load_balance, action_mode = native（未降维）
  - 奖励 = ep_reward（即 -effective_peak_load）
  - 最大负载 = min(eval_eff_peak)（训练全程最佳贪婪评估）

文件清单：
  1  ppo  native reward   2  sac  native reward   3  ddqn  native reward
  4  native max_load（三算法汇总）
  5  ppo  native delay    6  sac  native delay    7  ddqn  native delay
  8~14  dimred（降维）对应 8 个占位（尚未运行，data=null）

只读源文件，不修改任何原有 checkpoint / json。
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")

SEEDS = ["42", "123", "2024"]

ALGO_DIRS = {
    "ppo": [
        "checkpoints_ppo_cold_seed42",
        "checkpoints_ppo_cold_seed123",
        "checkpoints_ppo_cold_seed2024",
    ],
    "sac": [
        "checkpoints_sac_seed42_0716",
        "checkpoints_sac_seed123_0716",
        "checkpoints_sac_seed2024_0716",
    ],
    "ddqn": [
        "checkpoints_ddqn_seed42_0716",
        "checkpoints_ddqn_seed123_0716",
        "checkpoints_ddqn_seed2024_0716",
    ],
}

ALGO_CN = {"ppo": "PPO", "sac": "SAC", "ddqn": "DDQN"}


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_curves(path):
    try:
        return load_json(path)
    except UnicodeDecodeError:
        with open(path, encoding="gbk") as f:
            return json.load(f)


def get_curve(curves_dir, seed):
    """返回该种子目录的 training_curves.json，缺失则 None。"""
    p = os.path.join(HERE, curves_dir, "training_curves.json")
    if not os.path.exists(p):
        return None
    return load_curves(p)


def write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"  wrote {os.path.basename(path)}")


def base_meta(algo, dimred=False):
    m = {
        "algorithm": ALGO_CN.get(algo, algo),
        "dim_reduction": dimred,
        "action_mode": "dimred" if dimred else "native",
        "period_mode": "simple",
        "order_mode": "random",
        "flow_count": 50,
        "reward_mode": "load_balance",
    }
    return m


# =====================================================================
# 奖励曲线 (1/2/3)
# =====================================================================
def export_reward(algo):
    seeds = {}
    for i, d in enumerate(ALGO_DIRS[algo]):
        seed = SEEDS[i]
        c = get_curve(d, seed)
        if c is None or "ep_reward" not in c:
            seeds[seed] = {"episode": [], "ep_reward": [], "missing": True}
            continue
        seeds[seed] = {
            "episode": c.get("episode", []),
            "ep_reward": c.get("ep_reward", []),
        }
    meta = base_meta(algo, dimred=False)
    meta["metric"] = "ep_reward (= -effective_peak_load)"
    return {"meta": meta, "seeds": seeds}


# =====================================================================
# 最大负载 (4)
# =====================================================================
def export_max_load():
    algos = {}
    for algo in ALGO_DIRS:
        per_seed = {}
        notes = []
        for i, d in enumerate(ALGO_DIRS[algo]):
            seed = SEEDS[i]
            c = get_curve(d, seed)
            if c is None or "eval_eff_peak" not in c:
                per_seed[seed] = None
                notes.append(f"{algo}/seed{seed}: 无 eval_eff_peak")
                continue
            ee = [x for x in c["eval_eff_peak"] if x is not None]
            best = round(float(min(ee)), 4) if ee else None
            per_seed[seed] = best
        algos[ALGO_CN[algo]] = {"eval_best_per_seed": per_seed, "note": notes or None}
    meta = {
        "dim_reduction": False,
        "action_mode": "native",
        "period_mode": "simple",
        "order_mode": "random",
        "flow_count": 50,
        "metric": "eval_best = min(eval_eff_peak)  (训练全程最佳贪婪评估, lower=better)",
    }
    return {"meta": meta, "algorithms": algos}


# =====================================================================
# 时延 (5/6/7)
# =====================================================================
def export_delay(algo):
    seeds = {}
    for i, d in enumerate(ALGO_DIRS[algo]):
        seed = SEEDS[i]
        c = get_curve(d, seed)
        if c is None:
            seeds[seed] = {"episode": [], "ep_delay": None, "missing": True}
            continue
        if "ep_delay" not in c:
            seeds[seed] = {"episode": c.get("episode", []), "ep_delay": None,
                           "note": "无 ep_delay 字段（训练时未记录）"}
            continue
        delay = c["ep_delay"]
        valid = [x for x in delay if x is not None]
        if not valid:
            seeds[seed] = {"episode": c.get("episode", []), "ep_delay": None,
                           "note": "ep_delay 全为 null（训练时未记录）"}
            continue
        seeds[seed] = {"episode": c.get("episode", []), "ep_delay": delay,
                       "delay_mean_ms": round(sum(valid) / len(valid), 4),
                       "delay_min_ms": round(min(valid), 4),
                       "delay_max_ms": round(max(valid), 4)}
    meta = base_meta(algo, dimred=False)
    meta["metric"] = "ep_delay = average_delay_ms (每 episode 调度平均时延, 单位 ms)"
    return {"meta": meta, "seeds": seeds}


# =====================================================================
# 降维占位 (8~14)
# =====================================================================
def export_dimred_placeholder(algo=None, kind="reward"):
    meta = base_meta(algo, dimred=True) if algo else {
        "dim_reduction": True, "action_mode": "dimred",
        "period_mode": "simple", "order_mode": "random", "flow_count": 50,
        "reward_mode": "load_balance",
    }
    meta["metric"] = {"reward": "ep_reward", "max_load": "eval_best = min(eval_eff_peak)",
                      "delay": "ep_delay (average_delay_ms)"}[kind]
    return {
        "meta": meta,
        "status": "pending",
        "note": "降维(dimred)实验尚未运行，数据待补充。",
        "data": None,
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    print("=== 导出未降维数据 ===")

    # 1/2/3 奖励曲线
    for i, algo in enumerate(["ppo", "sac", "ddqn"], start=1):
        write(os.path.join(OUT, f"0{i}_{algo}_native_reward.json"), export_reward(algo))

    # 4 最大负载汇总
    write(os.path.join(OUT, "04_native_max_load.json"), export_max_load())

    # 5/6/7 时延
    for i, algo in enumerate(["ppo", "sac", "ddqn"], start=5):
        write(os.path.join(OUT, f"0{i}_{algo}_native_delay.json"), export_delay(algo))

    print("=== 导出降维占位（空） ===")

    # 8/9/10 奖励曲线
    for i, algo in enumerate(["ppo", "sac", "ddqn"], start=8):
        write(os.path.join(OUT, f"{i}_{algo}_dimred_reward.json"),
              export_dimred_placeholder(algo, "reward"))

    # 11 最大负载汇总
    write(os.path.join(OUT, "11_dimred_max_load.json"),
          export_dimred_placeholder(None, "max_load"))

    # 12/13/14 时延
    for i, algo in enumerate(["ppo", "sac", "ddqn"], start=12):
        write(os.path.join(OUT, f"{i}_{algo}_dimred_delay.json"),
              export_dimred_placeholder(algo, "delay"))

    print("\n完成。14 个 JSON 已写入 data/ 目录。")


if __name__ == "__main__":
    main()
