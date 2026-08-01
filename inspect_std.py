# -*- coding: utf-8 -*-
import json, numpy as np

def load(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except UnicodeDecodeError:
        return json.load(open(p, encoding="gbk"))

def per_seed_final(dirlist, cap=800):
    out = []
    for d in dirlist:
        c = load(d + "/training_curves.json")
        ep = np.asarray(c["episode"], float)
        rw = np.asarray(c["ep_reward"], float)
        # 取 <= cap 的最后一点
        idx = np.where(ep <= cap)[0]
        i = idx[-1]
        out.append(rw[i])
    return np.array(out)

print("=== DDQN 3 种子在 episode=800 处的 reward ===")
ddqn = per_seed_final([
    "checkpoints_ddqn_seed42_0716",
    "checkpoints_ddqn_seed123_0716",
    "checkpoints_ddqn_seed2024_0716",
])
print("  各种子:", np.round(ddqn, 2))
print("  mean=%.2f  std=%.2f  min=%.2f  max=%.2f  (max-min)=%.2f"
      % (ddqn.mean(), ddqn.std(), ddqn.min(), ddqn.max(), ddqn.max()-ddqn.min()))

print("\n=== SAC 3 种子在 episode=800 处的 reward ===")
sac = per_seed_final([
    "checkpoints_sac_seed42_0716",
    "checkpoints_sac_seed123_0716",
    "checkpoints_sac_seed2024_0716",
])
print("  各种子:", np.round(sac, 2))
print("  mean=%.2f  std=%.2f" % (sac.mean(), sac.std()))

print("\n=== SAC 均值曲线后半段趋势（每 100 episode 抽样）===")
c = load("checkpoints_sac_seed42_0716/training_curves.json")
ep = np.asarray(c["episode"], float); rw = np.asarray(c["ep_reward"], float)
for e in [400,500,600,700,800]:
    i = np.where(ep<=e)[0][-1]
    print(f"  ep={e:4d}  reward={rw[i]:.2f}")
