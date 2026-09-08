# 5G-TSN DRL 多链路调度 — 实验执行协议（每次任务前必读）

> 本文件记录本项目**已踩过的所有坑**、**正确的实验设定**、以及**实验设计思路**。
> **每次开始执行任何相关任务前，先读本文件。** 读完后，回复开头先发「泥嚎」再继续输出。
>
> 最后更新：2026-09-06

---

## 0. 一句话总纲

**降维（dimred）方法的对比基线是「未降维（native）版本」**：论文的创新点是「动作空间降维」，因此要回答的唯一问题是——降维后（96 → 7 维动作）能否在**大幅缩小动作空间**的同时，保持/逼近 native 的调度质量。

- **对比基线 = native（未降维）**：PPO/SAC/DDQN 各自的 native 版本，是 dimred 版本的唯一合法对比对象。
- **`random_feasible`（≈1.0）/ `edf_min_load`（0.816）/ `urgency_lexicographic`（0.667）/ MILP（0.545）只是「性能参考标尺」**，用于标注绝对水平，**不是对比基线**。
- 训练与评估统一 `order_mode="random"`（随机顺序 + 学放置），保证 native / dimred 在同一 MDP 上公平对比。

---

## 1. 核心基线定义（最重要，曾在这里栽过）

| 角色 | 方法 | eff_peak（参考） | 说明 |
|---|---|---|---|
| **降维对比基线** | native（未降维）三算法 | PPO 0.976 / SAC 0.681 / DDQN 0.754 | dimred 的唯一合法对比对象 |
| 降维方法 | dimred 三算法 | PPO 0.968 / SAC 1.040 / DDQN 1.032 | 论文创新点，与 native 成对对比 |
| 性能标尺（随机下界） | `random_feasible` | ≈1.0 | 随机顺序 + 随机放置，仅作绝对水平参考 |
| 性能标尺（启发式） | `edf_min_load` / `urgency_lexicographic` | 0.816 / 0.667 | 仅作参考，非基线 |
| 性能标尺（最优） | MILP | 0.545 | ≤50 流最优参考，非训练目标 |

- **「基线」一词在本项目专指 native（未降维）版本**。
- `random_feasible` / 启发式 / MILP 统一改称「**参考标尺（reference）**」，用于标注绝对水平。
- **降维对比 = 同一算法 native vs dimred**（PPO native↔PPO dimred 成对），同顺序、同 env、同奖励，只有动作空间维度不同 → 公平。

---

## 2. 所有错误要点（踩坑清单，执行前逐条自查）

1. ❌ **env 把 EDF 顺序硬编码** → 对比不公平（顺序增益不是 RL 学的）。
   ✅ 已修：`SimulationConfig` 增 `order_mode`(edf/random)；`env._build_episode_data` 支持 `random`。

2. ❌ **对比对象错**：曾把 PPO 和 `edf_min_load`(0.667) 比，得出"PPO 崩了"。
   ✅ 正确：降维对比应聚焦 **native vs dimred**；启发式只作参考标尺。

3. ❌ **错比法数字 "-27%"**：EDF 顺序上训练的 DRL vs random_feasible 标尺，苹果比橘子，无效。
   ✅ 正确：native / dimred 都必须在 `order_mode="random"` 的同一 env 上跑，成对对比。

4. ❌ **A/B 路线在错误 MDP 上验证**：γ↑0.99+n-step / mixed 对准峰值，是在固定 EDF env 上跑的，结论不能平移。
   ✅ 修正：在 `order_mode="random"` 的 env 上重做才有效。

5. ❌ **train_ppo.py 对比段默认用 `urgency_lexicographic` 且错标 "EDF+MinLoad"**。
   ✅ 已修：对比段不再把启发式当基线；参考标尺仅作绝对水平标注。

6. ❌ **最终评估用 final 模型**（训练不稳定会回退）而非 best checkpoint。
   ✅ 已修：最终评估 / eval-only 加载 best checkpoint。

7. ❌ **diag_greedy / diag_flow / diag_reward 在 EDF-env 上诊断**，非主线，仅供对照，不能据此下结论。

8. ❌ **γ=0.99 让回报量级暴涨 → critic 学崩**（valL 800~1284），polL≈0 策略几乎不更新。
   ✅ 修正后：`mixed` + `γ=0.9` + `gae_lambda=0.95` + `n_steps=4`，critic 稳定（valL~4）。

9. ❌ **A 路线（信用分配 / γ）误诊为根因**：critic 稳住后 PPO 仍≈1.0，说明真因是"冷启动学不动贪婪策略"，并非 γ/信用分配。

10. ❌ **dimred 在 off-policy 算法上直接用「因子化 Q」**（SAC/DDQN 的 `Q(a)=Σ q_bit[i]·bit[i]`）：link/slot 位强耦合下分解假设失效，学不动（SAC 1.040 / DDQN 1.032 反恶化）。
    ✅ 方向：TD3 用「连续松弛（softmax/sigmoid）+ 阈值化」桥接，避免因子化假设（见 `tsn_sim/td3.py`）。

---

## 3. 正确实验步骤（标准流程，照做）

**步骤 1：确认 env 设定**
- 构造 `SimulationConfig(order_mode="random")`，env 传 `order_mode="random"`。
- 绝不默认 `edf`（除非明确在"复现历史"语境下）。

**步骤 2：跑 native 基线（降维的对比对象）**
- 对每个算法（PPO/SAC/DDQN/TD3），用 `action_mode="native"` 训练，记录 best `eff_peak`。
- 这是 dimred 版本的**唯一合法对比基线**。

**步骤 3：跑 dimred 方法（论文创新点）**
- 用 `action_mode="dimred"` 训练同一算法，记录 best `eff_peak`。
- 对比：`dimred_eff_peak` vs `native_eff_peak`（同算法成对），计算降维的性能代价 / 收益。

**步骤 4：可选标注参考标尺**
- `random_feasible`（随机下界）、启发式（0.816/0.667）、MILP（0.545）仅作绝对水平标注，不进对比。

**步骤 5：记录到记忆**
- 把本次配置 + 结果追加到 `.workbuddy/memory/YYYY-MM-DD.md`。
- 涉及基线/对比口径/铁律的变更，同步更新 `MEMORY.md`。

---

## 4. 实验设计思路（论文导向）

- **数据集**：多规模 `flow_count ∈ {10,20,50,100,200}` × `Simple 集`（2 固定周期） / `Random 集`（全周期随机）
  - 目的：测试算法对不同**周期复杂度**的鲁棒性（复刻论文 Fig.10 的 simple vs random 对比）。
- **算法对比（native 三/四算法 + dimred 对应版本）**：PPO + 离散 SAC + Double-DQN + TD3，各自 native 与 dimred 成对对比。
  - 启发式（3 种）与 MILP（≤50 流）作参考标尺，不进对比基线。
- **指标**：`effective_peak_load`(=Max Slot Occupation) / `scheduling_success_rate`(=Acceptance Rate) / 时间开销。
- **MDP 决策**：离散顺序 env（96 离散 + mask 原生；dimred 为 7-bit 结构化降维）。
- **论文 RQ**：RQ1 规模扩展性 / RQ2 周期复杂度鲁棒性 / RQ3 算法选型（native）/ RQ4 降维代价（dimred vs native）/ RQ5 与 MILP 差距。
- **当前瓶颈与下一步**：
  - dimred 在 off-policy 上因子化 Q 学不动 → 用 TD3 连续松弛桥接（已写 `td3.py`）。
  - native 冷启动增益偏小（PPO 贴 random）→ 可探索 BC 暖启动 / Plan B（学顺序）。

---

## 5. 关键数字速查

| 方法 | 动作空间 | eff_peak | 角色 |
|---|---|---|---|
| PPO native | Discrete(96) | 0.976 | dimred 基线 |
| PPO dimred | MultiBinary(7) | 0.968 | 降维方法（打平） |
| SAC native | Discrete(96) | 0.681 | dimred 基线 |
| SAC dimred | MultiBinary(7) | 1.040 | 降维方法（恶化，因子化Q问题） |
| DDQN native | Discrete(96) | 0.754 | dimred 基线 |
| DDQN dimred | MultiBinary(7) | 1.032 | 降维方法（恶化，因子化Q问题） |
| random_feasible | — | 1.005 | 参考标尺（随机下界） |
| edf_min_load / urgency | — | 0.816 / 0.667 | 参考标尺（启发式） |
| MILP | — | 0.545 | 参考标尺（最优） |

---

## 6. 文件索引

- **协议/规范**：本文件 `EXPERIMENT_PROTOCOL.md`（根目录）
- **env**：`tsn_sim/env.py`（`order_mode` / `action_mode` 支持）
- **算法**：`tsn_sim/ppo.py`、`tsn_sim/sac.py`、`tsn_sim/ddqn.py`、`tsn_sim/td3.py`（TD3 连续松弛桥接）
- **训练入口**：`train_ppo.py` / `train_sac.py` / `train_ddqn.py` / `train_td3.py`
- **基线计算**：`baseline_random.py`
- **长期记忆**：`.workbuddy/memory/MEMORY.md`、`2026-07-15.md`

---

## 7. 铁律（违反任意一条 = 实验作废）

1. env 必须用 `order_mode="random"`，native / dimred 在同一 MDP 上公平对比。
2. **降维对比基线只能是 native（未降维）同算法**，绝不拿 dimred 和启发式/random 比作为"改进"。
3. `random_feasible` / 启发式 / MILP 是**参考标尺**，用于标注绝对水平，不是对比基线。
4. 最终报告用 best checkpoint，不用回退的 final 模型。
5. 任何涉及基线/对比口径的结论变更，先读本文件再动手。
