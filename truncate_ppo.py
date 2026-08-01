# -*- coding: utf-8 -*-
"""把 PPO 的 training_curves.json 截断到前 800 个 episode。

做法：
  - episode 类数组（episode, ep_reward, ep_eff_peak, ep_peak, ep_drop,
    ep_delay, ep_steps, epsilon, q_value）按 episode <= 800 截断；
  - eval 类数组（eval_eff_peak, eval_peak, eval_drop, eval_episode）按
    eval_episode <= 800 截断。
  - 先备份原文件为 *_full.json，再写回截断后的内容。
"""
import os
import json
import shutil

PATH = "checkpoints_ppo_seed42_0716/training_curves.json"
BACKUP = "checkpoints_ppo_seed42_0716/training_curves_full.json"
EPISODE_CAP = 800

# 1) 备份（若已存在则跳过备份）
if not os.path.exists(BACKUP):
    shutil.copyfile(PATH, BACKUP)
    print(f"[backup] saved original -> {BACKUP}")
else:
    print(f"[backup] already exists, skip: {BACKUP}")

# 2) 加载
d = json.load(open(PATH, encoding="utf-8"))
print("keys:", list(d.keys()))

# 3) 截断 episode 类数组
ep = d["episode"]
n = sum(1 for e in ep if e <= EPISODE_CAP)
assert n > 0, "no episode <= %d found" % EPISODE_CAP
print(f"episode array len={len(ep)} -> keep first {n} (episode <= {EPISODE_CAP})")

ep_keys = [k for k in d.keys() if k not in
           ("eval_eff_peak", "eval_peak", "eval_drop", "eval_episode")]
for k in ep_keys:
    if isinstance(d[k], list) and len(d[k]) == len(ep):
        d[k] = d[k][:n]

# 4) 截断 eval 类数组
if "eval_episode" in d:
    ev = d["eval_episode"]
    m = sum(1 for e in ev if e <= EPISODE_CAP)
    print(f"eval_episode len={len(ev)} -> keep first {m} (eval_episode <= {EPISODE_CAP})")
    eval_keys = ["eval_eff_peak", "eval_peak", "eval_drop", "eval_episode"]
    for k in eval_keys:
        if k in d and isinstance(d[k], list) and len(d[k]) == len(ev):
            d[k] = d[k][:m]

# 5) 写回
json.dump(d, open(PATH, "w", encoding="utf-8"), indent=2)
print(f"[done] truncated -> {PATH}")
