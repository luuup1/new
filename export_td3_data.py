# -*- coding: utf-8 -*-
"""把 TD3 native + dimred 各 3 seed 训练结果导出为 data/ 下的标准 JSON。

对齐 export_data.py / export_dimred_data.py 的既有口径（period=simple,
order=random, flow=50, reward=load_balance, metric=ep_reward / ep_delay /
min(eval_eff_peak)）。

文件编号（TD3 为第 4 个算法，接续在既有 01~14 之后）：

  native:
    15_td3_native_reward.json    (ep_reward 曲线, 3 seed)
    16_td3_native_delay.json     (ep_delay 曲线, 3 seed)
  dimred:
    17_td3_dimred_reward.json    (ep_reward 曲线, 3 seed)
    18_td3_dimred_delay.json     (ep_delay 曲线, 3 seed)
  汇总（native + dimred 各自 max_load，或合并到一份对比文件）：
    19_td3_max_load.json         (native & dimred 的 eval_best per seed)

checkpoint 目录命名约定（与训练命令的 --save-dir 对应）：
  native: checkpoints_td3_native_seed{42,123,2024}
  dimred: checkpoints_td3_dimred_seed{42,123,2024}

只读源文件，不修改任何 checkpoint / json。
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")

SEEDS = ["42", "123", "2024"]

DIRS = {
    "native": ["checkpoints_td3_native_s42",
               "checkpoints_td3_native_s123",
               "checkpoints_td3_native_s2024"],
    "dimred": ["checkpoints_td3_dimred_seed42",
               "checkpoints_td3_dimred_seed123",
               "checkpoints_td3_dimred_seed2024"],
}


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


def base_meta(action_mode):
    return {
        "algorithm": "TD3",
        "dim_reduction": action_mode == "dimred",
        "action_mode": action_mode,
        "period_mode": "simple",
        "order_mode": "random",
        "flow_count": 50,
        "reward_mode": "load_balance",
        "gamma": 0.9,
    }


def export_reward(action_mode):
    seeds = {}
    for i, d in enumerate(DIRS[action_mode]):
        seed = SEEDS[i]
        c = get_curve(d)
        if c is None or "ep_reward" not in c:
            seeds[seed] = {"episode": [], "ep_reward": [], "missing": True}
            continue
        seeds[seed] = {
            "episode": c.get("episode", []),
            "ep_reward": c.get("ep_reward", []),
        }
    meta = base_meta(action_mode)
    meta["metric"] = "ep_reward (= -effective_peak_load)"
    return {"meta": meta, "seeds": seeds}


def export_delay(action_mode):
    seeds = {}
    for i, d in enumerate(DIRS[action_mode]):
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
    meta = base_meta(action_mode)
    meta["metric"] = "ep_delay = average_delay_ms (每 episode 调度平均时延, 单位 ms)"
    return {"meta": meta, "seeds": seeds}


def export_max_load():
    """native 与 dimred 的 eval_best per seed 汇总到一份对比文件。"""
    out = {}
    for action_mode in ["native", "dimred"]:
        per_seed = {}
        notes = []
        for i, d in enumerate(DIRS[action_mode]):
            seed = SEEDS[i]
            c = get_curve(d)
            if c is None or "eval_eff_peak" not in c:
                per_seed[seed] = None
                notes.append(f"{action_mode}/seed{seed}: 无 eval_eff_peak")
                continue
            ee = [x for x in c["eval_eff_peak"] if x is not None]
            best = round(float(min(ee)), 4) if ee else None
            per_seed[seed] = best
        out[action_mode] = {"eval_best_per_seed": per_seed, "note": notes or None}
    meta = {
        "algorithm": "TD3",
        "period_mode": "simple",
        "order_mode": "random",
        "flow_count": 50,
        "reward_mode": "load_balance",
        "gamma": 0.9,
        "metric": "eval_best = min(eval_eff_peak)  (训练全程最佳贪婪评估, lower=better)",
        "note": "dimred 的唯一合法对比基线是 native 同算法；启发式/MILP 仅作参考标尺",
    }
    return {"meta": meta, "algorithms": out}


def main():
    os.makedirs(OUT, exist_ok=True)
    print("=== 导出 TD3 native 数据 ===")
    write(os.path.join(OUT, "15_td3_native_reward.json"), export_reward("native"))
    write(os.path.join(OUT, "16_td3_native_delay.json"), export_delay("native"))

    print("=== 导出 TD3 dimred 数据 ===")
    write(os.path.join(OUT, "17_td3_dimred_reward.json"), export_reward("dimred"))
    write(os.path.join(OUT, "18_td3_dimred_delay.json"), export_delay("dimred"))

    print("=== 导出 TD3 max_load 汇总 ===")
    write(os.path.join(OUT, "19_td3_max_load.json"), export_max_load())

    print("\n完成。TD3 5 个 JSON 已写入 data/ 目录。")


if __name__ == "__main__":
    main()
