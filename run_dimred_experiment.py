# -*- coding: utf-8 -*-
"""降维(dimred) cold 三算法 × 3 seed 训练驱动脚本。

口径（方案甲铁律 + 用户拍板）：
  - 统一: --order-mode random --period-mode simple --reward-mode load_balance --action-mode dimred
  - 关闭早停（--no-early-stop 跑满，排除"早停太早没给 dimred 机会"的干扰）
  - 训练规模对齐 native: PPO 2000ep / SAC 800ep / DDQN 800ep
  - 超参对齐 native: PPO γ=0.99 / SAC γ=0.5 / DDQN γ=0.9

输出目录（对齐 native 命名，加 _dimred 后缀）:
  checkpoints_ppo_cold_dimred_seed{42,123,2024}
  checkpoints_sac_dimred_seed{42,123,2024}
  checkpoints_ddqn_dimred_seed{42,123,2024}

用法:
  RL/bin/python run_dimred_experiment.py --algo ppo|sac|ddqn [--seeds 42,123,2024]
  建议 3 个算法各自起一个后台进程并行跑。
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = str(HERE / "RL" / "bin" / "python")

SEEDS = [42, 123, 2024]

CONFIG = {
    "ppo": {
        "script": "train_ppo.py",
        "episodes": 2000,
        "save_prefix": "checkpoints_ppo_cold_dimred_seed",
        "common": [
            "--order-mode", "random",
            "--period-mode", "simple",
            "--reward-mode", "load_balance",
            "--action-mode", "dimred",
            "--gamma", "0.99",
        ],
    },
    "sac": {
        "script": "train_sac.py",
        "episodes": 800,
        "save_prefix": "checkpoints_sac_dimred_seed",
        "common": [
            "--order-mode", "random",
            "--period-mode", "simple",
            "--reward-mode", "load_balance",
            "--action-mode", "dimred",
            "--gamma", "0.5",
        ],
    },
    "ddqn": {
        "script": "train_ddqn.py",
        "episodes": 800,
        "save_prefix": "checkpoints_ddqn_dimred_seed",
        "common": [
            "--order-mode", "random",
            "--period-mode", "simple",
            "--reward-mode", "load_balance",
            "--action-mode", "dimred",
            "--gamma", "0.9",
        ],
    },
}


def run_algo(algo: str, seeds):
    cfg = CONFIG[algo]
    for seed in seeds:
        save_dir = f"{cfg['save_prefix']}{seed}"
        cmd = [
            PY, cfg["script"],
            "--episodes", str(cfg["episodes"]),
            "--seed", str(seed),
            "--save-dir", save_dir,
            "--no-early-stop",
            *cfg["common"],
        ]
        print(f"\n{'='*70}")
        print(f"[{algo.upper()}] seed={seed}  save_dir={save_dir}")
        print(f"[CMD] {' '.join(cmd)}")
        print("="*70)
        t0 = time.time()
        rc = subprocess.call(cmd, cwd=str(HERE))
        dt = time.time() - t0
        print(f"[DONE] {algo} seed={seed} rc={rc} 耗时={dt/60:.1f} min")
        if rc != 0:
            print(f"[FAIL] {algo} seed={seed} 返回码 {rc}，中止该算法后续 seed")
            return rc
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True, choices=["ppo", "sac", "ddqn"])
    ap.add_argument("--seeds", default="42,123,2024")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    rc = run_algo(args.algo, seeds)
    sys.exit(rc)


if __name__ == "__main__":
    main()
