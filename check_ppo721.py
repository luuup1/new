import json, os

def load(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except UnicodeDecodeError:
        return json.load(open(p, encoding="gbk"))

def summarize(tag, d):
    print(f"\n=== {tag} ===")
    keys = list(d.keys())
    print("JSON keys:", keys)
    # 长度
    for k in ("episode", "ep_reward", "ep_eff_peak", "eval_episode", "eval_eff_peak"):
        if k in d:
            v = d[k]
            print(f"  {k}: len={len(v)}  first={v[0] if v else None}  last={v[-1] if v else None}")
    # 是否有 config 记录
    for ck in ("config", "cfg", "args", "hyperparams", "params"):
        if ck in d:
            print(f"  [config found] {ck}:", d[ck])
    # eval 末值
    if "eval_eff_peak" in d and d["eval_eff_peak"]:
        ev = d["eval_eff_peak"]
        print(f"  eval_eff_peak final = {ev[-1]:.4f}  (n_eval={len(ev)})")
    if "eval_reward" in d and d["eval_reward"]:
        print(f"  eval_reward final = {d['eval_reward'][-1]:.4f}")

d721 = load("checkpoints_ppo_seed721_0721/training_curves.json")
summarize("PPO seed721_0721 (NEW)", d721)

d42 = load("checkpoints_ppo_seed42_0716/training_curves.json")
summarize("PPO seed42_0716 (existing, truncated to 800)", d42)

# 比较两者 episode 范围是否对齐
print("\n=== episode range 对齐检查 ===")
print("721 episode:", d721["episode"][0], "->", d721["episode"][-1], "len", len(d721["episode"]))
print("42  episode:", d42["episode"][0], "->", d42["episode"][-1], "len", len(d42["episode"]))
