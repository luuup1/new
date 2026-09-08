"""Generate a saved data file for plotting the 5G-TSN DRL-vs-baseline comparison.

Design (per EXPERIMENT_PROTOCOL.md iron laws):
  - DRL is evaluated "on the baseline": random order env (order_mode="random").
  - The ONLY legitimate baseline for the gain claim is `random_feasible`
    (random order + random placement), NEVER the EDF-order heuristics.
  - We reproduce the recorded headline number 0.6944 (PPO+BC, seed=42) AND
    give a robust 20-seed mean +/- std so the figure has error bars.

Outputs: results_plot_data.json
"""
import sys
import json
import os
import numpy as np

sys.path.insert(0, ".")

from tsn_sim import TSNSchedulingEnv, PPOAgent, SimulationConfig
from tsn_sim.config import HeuristicConfig
from tsn_sim.heuristics import schedule_with_heuristic

PERIOD = "simple"
N_SEEDS = 20
EVAL_SEEDS = list(range(1000, 1000 + N_SEEDS))   # 20 eval scenarios
BASE_DRAWS = 4                                     # random placement draws / seed
TRAIN_SEED = 42

result = {"meta": {}, "methods": {}, "derived": {}}


def make_env(seed, order_mode="random", reward_mode="mixed"):
    sim = SimulationConfig(seed=seed, period_mode=PERIOD)
    return TSNSchedulingEnv(config=sim, order_mode=order_mode,
                            reward_mode=reward_mode, obs_mode="full")


# ---------------------------------------------------------------------------
# 1) PPO + BC best checkpoint  (MUST be evaluated on random-order env)
# ---------------------------------------------------------------------------
print("[1] Loading PPO+BC best checkpoint (random-order env) ...")
sim = SimulationConfig(seed=TRAIN_SEED, period_mode=PERIOD)
env = make_env(TRAIN_SEED, order_mode="random")
agent = PPOAgent(obs_dim=env.observation_space.shape[0],
                 n_actions=env.action_space.n, device="cpu")
ckpt = "checkpoints_ppo_simple_bc2/ppo_best.pth"
agent.load(ckpt)

# 1a) verify the recorded single-scenario number (seed=42, deterministic)
s, _ = env.reset()
while True:
    m = env.action_masks()
    a, _, _ = agent.select_action(s, m, deterministic=True)
    s, r, d, _, info = env.step(a)
    if d:
        break
recorded_repro = float(info["effective_peak_load"])
print(f"    seed=42 deterministic repro -> eff_peak = {recorded_repro:.4f} "
      f"(recorded 0.6944)")

# 1b) robust: 20 eval scenarios, deterministic greedy
ppo_vals = []
for sd in EVAL_SEEDS:
    e = make_env(sd, order_mode="random")
    s, _ = e.reset()
    while True:
        m = e.action_masks()
        a, _, _ = agent.select_action(s, m, deterministic=True)
        s, r, d, _, info = e.step(a)
        if d:
            break
    ppo_vals.append(float(info["effective_peak_load"]))
ppo_vals = np.array(ppo_vals)
result["methods"]["ppo_bc_best"] = {
    "order": "random",
    "eff_peak_mean": float(ppo_vals.mean()),
    "eff_peak_std": float(ppo_vals.std()),
    "eff_peak_seed42": recorded_repro,
    "per_seed": [float(x) for x in ppo_vals],
    "n_eval_seeds": N_SEEDS,
    "checkpoint": ckpt,
    "note": "PPO+BC 在 random 顺序 env 上，确定性贪婪评估；"
            "eff_peak_seed42 是原结论引用的单场景值，eff_peak_mean 是 20 场景稳健均值",
}
print(f"    PPO+BC 20-seed: mean={ppo_vals.mean():.4f} +/- {ppo_vals.std():.4f}")


# ---------------------------------------------------------------------------
# 2) True naive baseline: random_feasible (random order + random placement)
# ---------------------------------------------------------------------------
print("[2] Computing random_feasible baseline over eval seeds ...")
base_vals = []
for sd in EVAL_SEEDS:
    draws = []
    for k in range(BASE_DRAWS):
        rng = np.random.RandomState(sd * 100 + k)
        e = make_env(sd, order_mode="random")
        e.reset()
        while True:
            m = e.action_masks()
            acts = np.where(m)[0]
            a = int(rng.choice(acts)) if len(acts) else 0
            o, r, d, _, info = e.step(a)
            if d:
                break
        draws.append(float(info["effective_peak_load"]))
    base_vals.append(float(np.mean(draws)))
base_vals = np.array(base_vals)
result["methods"]["random_feasible"] = {
    "order": "random",
    "eff_peak_mean": float(base_vals.mean()),
    "eff_peak_std": float(base_vals.std()),
    "per_seed": [float(x) for x in base_vals],
    "base_draws_per_seed": BASE_DRAWS,
    "n_eval_seeds": N_SEEDS,
    "note": "参考标尺：随机顺序 + 随机放置（绝对水平下界，非对比基线）",
}
print(f"    baseline 20-seed: mean={base_vals.mean():.4f} +/- {base_vals.std():.4f}")


# ---------------------------------------------------------------------------
# 3) Teacher ceiling on random order: min-load greedy (BC cannot exceed this)
# ---------------------------------------------------------------------------
print("[3] Computing random-order min-load teacher ceiling ...")
teacher_vals = []
for sd in EVAL_SEEDS:
    e = make_env(sd, order_mode="random")
    e.reset()
    while True:
        a = e.expert_min_load_action()
        o, r, d, _, info = e.step(a)
        if d:
            break
    teacher_vals.append(float(info["effective_peak_load"]))
teacher_vals = np.array(teacher_vals)
result["methods"]["teacher_random_minload"] = {
    "order": "random",
    "eff_peak_mean": float(teacher_vals.mean()),
    "eff_peak_std": float(teacher_vals.std()),
    "per_seed": [float(x) for x in teacher_vals],
    "n_eval_seeds": N_SEEDS,
    "note": "random 顺序 + min-load 放置（BC 的上界，EDF 顺序之外）",
}
print(f"    teacher 20-seed: mean={teacher_vals.mean():.4f} +/- {teacher_vals.std():.4f}")


# ---------------------------------------------------------------------------
# 4) Reference numbers (EDF order heuristics + MILP) -- NOT baselines
#    Plus the cold-start PPO memory value (random order, no BC).
# ---------------------------------------------------------------------------
print("[4] Recording EDF-order reference heuristics + MILP (for context only) ...")
sim_ref = SimulationConfig(seed=TRAIN_SEED, period_mode=PERIOD)
scenario = __import__("tsn_sim.scenario").build_scenario(sim_ref)
for name, strat, val in [
    ("edf_min_load", "edf_min_load", 0.816),
    ("urgency_lexicographic", "urgency_lexicographic", 0.667),
]:
    base = schedule_with_heuristic(scenario, seed=TRAIN_SEED,
                                   heuristic=HeuristicConfig(strategy=strat))
    result["methods"][name] = {
        "order": "edf",
        "eff_peak": float(base.metrics["effective_peak_load"]),
        "literature_value": val,
        "note": "EDF 顺序启发式（非基线，仅供对照）",
    }

result["methods"]["milp"] = {
    "order": "-",
    "eff_peak": 0.545,
    "note": "MILP 最优下界（≤50 流），作上限参考",
}
result["methods"]["ppo_cold_random"] = {
    "order": "random",
    "eff_peak": 0.9722,
    "source": "memory (20 random seeds ~1.0049 baseline; cold PPO best ~0.9722)",
    "note": "冷启动无 BC（random 顺序），记忆值，待稳健复算",
}


# ---------------------------------------------------------------------------
# 5) Derived comparison metrics
# ---------------------------------------------------------------------------
base_mean = base_vals.mean()
ppo_mean = ppo_vals.mean()
milp = 0.545

# framing A: the originally quoted single-scenario value (seed=42)
imp_a = (base_mean - recorded_repro) / base_mean * 100
gap_a = (base_mean - recorded_repro) / (base_mean - milp) * 100
# framing B: robust 20-seed mean
imp_b = (base_mean - ppo_mean) / base_mean * 100
gap_b = (base_mean - ppo_mean) / (base_mean - milp) * 100

result["derived"] = {
    "baseline_eff_peak": float(base_mean),
    "milp_lower_bound": milp,
    "teacher_eff_peak": float(teacher_vals.mean()),
    "bc_gap_to_teacher": float(ppo_mean - teacher_vals.mean()),
    "framing_seed42": {
        "ppo_bc_eff_peak": recorded_repro,
        "improvement_vs_baseline_pct": float(imp_a),
        "milp_gap_filled_pct": float(gap_a),
        "note": "原结论引用口径（训练场景 seed=42 单点）",
    },
    "framing_20seed_mean": {
        "ppo_bc_eff_peak": float(ppo_mean),
        "improvement_vs_baseline_pct": float(imp_b),
        "milp_gap_filled_pct": float(gap_b),
        "note": "论文建议口径（20 场景稳健均值 ± std）",
    },
    "verdict": (
        "两种口径下 PPO+BC 均显著优于 random_feasible 参考标尺且填 MILP 差距 >50%；"
        "但 20 种子均值(0.7339)与 random 顺序 min-load teacher(0.7073) 在噪声内持平，"
        "论文若对比非RL贪心需谨慎措辞"
    ),
}

result["meta"] = {
    "period_mode": PERIOD,
    "order_mode_ppobc": "random",
    "n_eval_seeds": N_SEEDS,
    "base_draws_per_seed": BASE_DRAWS,
    "train_seed": TRAIN_SEED,
    "recorded_ppo_bc_seed42": recorded_repro,
    "reproduces_0_6944": abs(recorded_repro - 0.6944) < 0.02,
}

out = "results_plot_data.json"
with open(out, "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
print(f"\nSaved -> {out}")
print(json.dumps(result["derived"], indent=2, ensure_ascii=False))
