# 5G-TSN DRL 多链路调度 — 实验执行协议（每次任务前必读）

> 本文件记录本项目**已踩过的所有坑**、**正确的实验设定**、以及**实验设计思路**。
> **每次开始执行任何相关任务前，先读本文件。** 读完后，回复开头先发「泥嚎」再继续输出。
>
> 最后更新：2026-07-15

---

## 0. 一句话总纲

**DRL 必须"在基线上进行"**：使用随机顺序 env（`order_mode="random"`），对比对象**只能**是真实朴素基线 `random_feasible`（≈1.0），**绝不能**和 EDF 顺序的启发式（0.667 / 0.816）比，也**绝不能用**固定 EDF 顺序的 env 作训练或对比。

---

## 1. 核心基线定义（最重要，曾在这里栽过）

| 角色 | 方法 | eff_peak（seed 参考） | 说明 |
|---|---|---|---|
| **朴素基线** | `random_feasible` | **≈1.0**（单次 1.000，20 种子均值 1.0049） | 随机顺序 + 随机放置，DRL 唯一合法对比对象 |
| 启发式改进 | `edf_min_load` | 0.816 | EDF 顺序 + 每步选最空格 |
| 最佳启发式 | `urgency_lexicographic` | 0.667 | EDF 顺序 + 字典序评分 |
| **DRL（本次 best）** | PPO | **0.9722** | 随机顺序 + 学放置，相对基线 +3% |
| 最优下界 | MILP | 0.545 | ≤50 流可用，作上限参考 |

- `edf_min_load` / `urgency_lexicographic` 是**启发式改进，不是基线**。
- MILP 是**最优下界参考**，不是训练目标。
- **DRL 在"基线上进行" = 随机顺序 + 学放置**，对比 随机顺序 + 随机放置（同顺序、只有放置策略不同 → 公平）。

---

## 2. 所有错误要点（踩坑清单，执行前逐条自查）

1. ❌ **env 把 EDF 顺序硬编码** → DRL 白拿顺序增益（这部分不是 RL 学的），对比不公平。
   ✅ 已修：`SimulationConfig` 增 `order_mode`(edf/random)；`env._build_episode_data` 支持 `random`（用场景 seed 复现 `random_feasible` 顺序）。

2. ❌ **对比对象错**：把 PPO 和 `edf_min_load`(0.667) 比，得出"PPO 崩了 / 冷启动学不动"。
   ✅ 正确：PPO(随机顺序) vs `random_feasible`(1.0)。

3. ❌ **错比法数字 "-27%"**：EDF 顺序上训练的 DRL(0.727) vs 随机基线(1.0)，苹果比橘子，**该数字无效**。
   ✅ 正确：随机顺序 DRL(0.9722) vs 随机基线(1.0) ≈ **+3%**（真实但偏小）。

4. ❌ **A/B 路线在错误 MDP 上验证**：γ↑0.99+n-step / mixed 对准峰值，是在**固定 EDF env** 上跑的，结论不能平移到正确 MDP。
   ✅ 修正：在 `order_mode="random"` 的 env 上重做才有效。

5. ❌ **train_ppo.py 对比段默认用 `urgency_lexicographic` 且错标 "EDF+MinLoad"**。
   ✅ 已修：对比段显式传 `strategy="random_feasible"`。

6. ❌ **最终评估用 final 模型**（训练不稳定会回退到 1.0）而非 best checkpoint。
   ✅ 已修：最终评估 / eval-only 加载 `ppo_best.pth`。

7. ❌ **diag_greedy / diag_flow / diag_reward 在 EDF-env 上诊断**，非主线，仅供对照，不能据此下结论。

8. ❌ **γ=0.99 让回报量级暴涨 → critic 学崩**（valL 800~1284），polL≈0 策略几乎不更新。
   ✅ 修正后：`mixed` + `γ=0.9` + `gae_lambda=0.95` + `n_steps=4`，critic 稳定（valL~4）。

9. ❌ **A 路线（信用分配 / γ）误诊为根因**：critic 稳住后 PPO 仍=1.0，说明真因是"冷启动学不动贪婪策略"，并非 γ/信用分配。
   ✅ 真因：2 层 MLP 需从 199 维 obs 学会"对 96 个 mask 动作做 argmin(post-load)"的关系型操作，且逐格奖励差仅 ~0.18 量级，400 episode 不足以收敛 → 策略停留在接近随机。

---

## 3. 正确实验步骤（标准流程，照做）

**步骤 1：确认 env 设定**
- 构造 `SimulationConfig(order_mode="random")`，env 传 `order_mode="random"`。
- 绝不默认 `edf`（除非明确在"复现历史基线"语境下）。

**步骤 2：计算真实朴素基线**
- 在随机顺序 env 上，用随机策略（或 `schedule_with_heuristic(random_feasible)`）跑 20+ 随机种子。
- 记录 `eff_peak` 均值（当前 ≈1.0049）。这是 DRL 唯一合法对比锚点。

**步骤 3：训练 DRL**
- 推荐配置：`reward_mode=mixed`（直接惩罚峰值）、`γ=0.9`、`gae_lambda=0.95`、`n_steps=4`、`lr=1e-4`、`epochs=8`。
- 按 eval `eff_peak` 选最优，保存 `ppo_best.pth`。

**步骤 4：评估（必须加载 best）**
- `evaluate` 20 episodes，加载 `ppo_best.pth`。
- 对比：`PPO_best_eff_peak` vs `random_feasible` 基线 → 计算改进 %。
- 若 final 模型回退到 1.0，以 best 为准，不要报告 final。

**步骤 5：记录到记忆**
- 把本次配置 + 结果追加到 `.workbuddy/memory/YYYY-MM-DD.md`。
- 涉及基线/对比口径/铁律的变更，同步更新 `MEMORY.md`。

---

## 4. 实验设计思路（论文导向）

- **数据集**：多规模 `flow_count ∈ {10,20,50,100,200}` × `Simple 集`（2 固定周期） / `Random 集`（全周期随机）
  - 目的：测试算法对不同**周期复杂度**的鲁棒性（复刻论文 Fig.10 的 simple vs random 对比）。
- **算法对比（离散原生三方）**：PPO + 离散 SAC + Double-DQN + 3 种启发式 + MILP(≤50 流)
  - DDPG（连续-only）已剔除；不预设谁最优，实证。
- **指标**：`effective_peak_load`(=Max Slot Occupation) / `scheduling_success_rate`(=Acceptance Rate) / 时间开销。
- **MDP 决策**：保持离散顺序 env（96 离散 + mask），对 PPO / 离散 SAC / Double-DQN 均为原生正确选择。
- **论文 RQ**：RQ1 规模扩展性 / RQ2 周期复杂度鲁棒性 / RQ3 RL 选型 / RQ4 与 MILP 差距。
- **当前瓶颈与下一步**：
  - 冷启动仅学到 min-load 贪婪的 22%，训练不稳定 → 增益仅 ~3%。
  - 要论文级增益：
    - **C（IL/BC 暖启动）**：用 `edf_min_load` 的 (obs, 最格动作) 演示预训练 actor，先学会 min-load 贪婪再 RL 微调；
    - **Plan B（让 agent 学包顺序）**：顺序才是大杠杆（EDF+min-load 能到 0.667 主要靠顺序），让 DRL 同时学顺序+放置，可能直接大幅压低峰值。

---

## 5. 关键数字速查

| 方法 | 顺序 | eff_peak | 说明 |
|---|---|---|---|
| random_feasible | 随机 | **1.0049** | 朴素基线（单次 1.000） |
| edf_min_load | EDF | 0.816 | 启发式 |
| urgency_lexicographic | EDF | 0.667 | 最佳启发式 |
| PPO（本次 best） | 随机 | **0.9722** | DRL 在基线上，+3% vs 基线 |
| MILP | - | 0.545 | 最优下界（≤50 流） |

---

## 6. 文件索引

- **协议/规范**：本文件 `EXPERIMENT_PROTOCOL.md`（根目录）
- **env**：`tsn_sim/env.py`（`order_mode` 支持）
- **PPO**：`tsn_sim/ppo.py`（γ / n_steps）
- **训练入口**：`train_ppo.py`（`--order-mode` / `--reward-mode` / `--eval-only` / `--n-steps`）
- **基线计算**：`baseline_random.py`
- **验证配对**：`verify_order_mode.py`
- **旧诊断（EDF-env，仅供对照，非主线）**：`diag_greedy.py` / `diag_flow.py` / `diag_reward.py`
- **长期记忆**：`.workbuddy/memory/MEMORY.md`、`2026-07-15.md`

---

## 7. 铁律（违反任意一条 = 实验作废）

1. env 必须用 `order_mode="random"`，DRL 在"基线上进行"。
2. 对比对象只能是 `random_feasible`（≈1.0），绝不和启发式(0.667/0.816) 比。
3. 绝不用 EDF 顺序 env 训练出的 DRL 去对比随机基线（错比法）。
4. 最终报告用 best checkpoint，不用回退的 final 模型。
5. 任何涉及基线/对比口径的结论变更，先读本文件再动手。
