# -*- coding: utf-8 -*-
"""整理奖励曲线数据，统一规格输出 reward_unified.json。

统一口径：
  1. 横轴统一为 episode 轮数，采样步长统一为 1（对 10ep 采样的用线性插值补到 1ep）。
  2. 每组内 3 个 seed 对齐：截断到公共有效长度（取 3 seed 的最短长度）。
  3. 跨算法保留各自真实训练长度（PPO=2000ep，SAC/DDQN=800ep），不做补零
     —— 因为 ep_reward 是累积负值，补零会让曲线在训练结束后错误地跳升到 0，
        造成"训练提前结束=完美"的误导，故用"截断到公共长度"替代补零。
  4. 输出每组的 episode 轴、3 个 seed 对齐后的 ep_reward，以及 mean / std。

输出：data/reward_unified.json
"""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# id, 源文件, 算法, 模式, 颜色, 线型
GROUPS = [
    ("01_PPO_native",  "01_ppo_native_reward.json",  "PPO",  "native", "#1f77b4", "-"),
    ("02_SAC_native",  "02_sac_native_reward.json",  "SAC",  "native", "#d62728", "-"),
    ("03_DDQN_native", "03_ddqn_native_reward.json", "DDQN", "native", "#2ca02c", "-"),
    ("08_PPO_dimred",  "8_ppo_dimred_reward.json",   "PPO",  "dimred", "#1f77b4", "--"),
    ("09_SAC_dimred",  "9_sac_dimred_reward.json",   "SAC",  "dimred", "#d62728", "--"),
    ("10_DDQN_dimred", "10_ddqn_dimred_reward.json", "DDQN", "dimred", "#2ca02c", "--"),
]

# 统一截断到 800 轮（用户要求：PPO 也画到 800 轮，所有曲线同长度）
MAX_EPISODE = 800


def load_json(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        return json.load(f)


def align_seed(ep, r):
    """把 (episode, reward) 对齐到 1ep 间隔、从 ep=1 开始的整数网格。

    - 若步长 > 1，用 np.interp 线性插值到 1ep；[1, 首个ep) 用端点值前向填充。
    - 若步长 == 1，直接使用。
    返回 (episode_1d, reward_1d)。
    """
    ep = np.array(ep, dtype=float)
    r = np.array(r, dtype=float)
    if len(ep) < 2:
        return None
    start = int(round(ep[0]))
    end = int(round(ep[-1]))
    grid = np.arange(1, end + 1, dtype=float)
    if len(ep) == end - start + 1 and np.allclose(np.diff(ep), 1.0):
        # 已经是 1ep 间隔，且从 ep=start 开始；补齐前导部分
        rr = r  # 已连续
        # 如果 start > 1，前向填充前导
        if start > 1:
            pad = np.full(start - 1, r[0])
            rr = np.concatenate([pad, r])
            grid = np.arange(1, end + 1, dtype=float)
        return grid, rr
    # 插值
    rr = np.interp(grid, ep, r)
    return grid, rr


def process_group(gid, src, algo, mode, color, ls):
    d = load_json(src)
    seeds = {}
    for s in ["42", "123", "2024"]:
        v = d["seeds"].get(s, {})
        ep = v.get("episode", [])
        r = v.get("ep_reward", [])
        if ep and r and len(ep) == len(r):
            res = align_seed(ep, r)
            if res is not None:
                seeds[s] = res
    if not seeds:
        return None

    # 公共长度 = min(len)，并统一截断到 MAX_EPISODE（800 轮）
    common_len = min(len(v[1]) for v in seeds.values())
    common_len = min(common_len, MAX_EPISODE)
    aligned = {}
    for s, (grid, rr) in seeds.items():
        aligned[s] = {
            "episode": [int(x) for x in grid[:common_len]],
            "ep_reward": [float(x) for x in rr[:common_len]],
        }

    # mean / std
    Y = np.vstack([np.array(v["ep_reward"]) for v in aligned.values()])
    mean = Y.mean(axis=0)
    std = Y.std(axis=0)

    return {
        "id": gid,
        "label": f"{algo} ({mode})",
        "algorithm": algo,
        "action_mode": mode,
        "color": color,
        "linestyle": ls,
        "n_seeds": len(aligned),
        "max_episode": int(common_len),
        "episode": [int(x) for x in grid[:common_len]],
        "seeds": aligned,
        "mean": [float(x) for x in mean],
        "std": [float(x) for x in std],
    }


def main():
    groups = {}
    for gid, src, algo, mode, color, ls in GROUPS:
        g = process_group(gid, src, algo, mode, color, ls)
        if g is None:
            print(f"[SKIP] {gid}: 无有效数据")
            continue
        groups[gid] = g
        print(f"[OK] {gid}: seeds={g['n_seeds']}  max_episode={g['max_episode']}")

    out = {
        "meta": {
            "metric": "ep_reward = 累积 -effective_peak_load（越接近 0 越好）",
            "x_axis": "episode",
            "x_unit": "episode",
            "y_unit": "reward (dimensionless)",
        "sampling": "统一 1 episode 间隔（10ep 采样已线性插值）",
        "alignment": "每组 3 seed 截断到公共有效长度；所有组统一截断到 800 轮（PPO 也从 2000 截断到 800）",
            "note": "缺失不补零（reward 补零会误导），改为截断到公共长度",
        },
        "groups": groups,
    }
    out_path = os.path.join(DATA, "reward_unified.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
