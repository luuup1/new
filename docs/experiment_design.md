# 实验设计文档（参照 SAC-FSO, Computer Networks 2025）

> 目标：以论文撰写为导向，按 SAC-FSO 论文的实验方法论，设计一套可复现、可对比、可写进论文的实验方案。
> 参考：Wang et al., "Towards wireless time-sensitive networking: Multi-link deterministic scheduling via deep reinforcement learning", *Computer Networks* 261 (2025) 111119.

---

## 1. 参考论文方法论提炼（必须对齐的点）

| 维度 | 论文做法 | 我们的适配 |
|---|---|---|
| 数据集规模 | 流数 10/100/200/400/600 | 我们为**包级** formulation，规模档位需下调（见 §2.1） |
| 数据集分类 | **Simple Set**（仅 2 种固定周期）/ **Random Set**（随机周期） | 直接对齐：用 `period_mode` 控制 |
| 对比指标 | Max Slot Occupation, Acceptance Rate, Time Cost | effective_peak_load / 调度成功率 / 推理+训练时间 |
| 算法对比 | SAC vs PPO vs DDPG（3 种 RL，均为**连续 MDP**）+ Tabu/Greedy/No-wait SMT | **PPO + 离散 SAC + Double-DQN**（均为**离散+mask 原生**，匹配当前 96 维离散动作）；DDPG 因连续-only，仅在连续 MDP(Plan B)下才有意义 |
| 奖励 | 指数 `r=δ·exp(ζ·D_max+μ)` | 我们已验证 `load_balance` 更优，但保留 exponential 作对照 |
| 折扣因子 | γ=0.99/0.7/0.5，结论 **γ=0.5 最优** | 与我们的 `load_balance+γ=0.5` 结论一致 |
| MDP | 决策变量映射 3D→2D 连续动作（每流 link+offset） | 见 §2.4，需决定离散顺序 env 还是连续映射 env |

**论文核心结论（可作为我们论文的对照基准）**：SAC-FSO 在收敛平滑度、最终奖励、训练稳定性上均优于 PPO/DDPG；在 Random Set 上各类指标（峰值、耗时）均高于 Simple Set；规模越大 SAC 优势越明显（大集上接近 Tabu/Greedy 的两倍）。

---

## 2. 我们的实验设计

### 2.1 数据集设计（规模 × 周期复杂度）

**规模档位（建议）**：`flow_count ∈ {10, 20, 50, 100, 200}`
- 说明：论文用流级 formulation（每步=1 流），我们为包级（每步=1 包），50 流≈310 包。直接照搬 600 流会到 ~3700 步/episode 且 MILP 不可解，故下调上限。
- MILP 最优参考仅在 **≤50 流**（scipy MILP 可解范围）提供；>50 流用启发式作下界。

**周期复杂度（Simple / Random）**——这是你要的"测试不同周期复杂度鲁棒性"：

| 集合 | period 生成方式 | 复杂度 | 预期 |
|---|---|---|---|
| **Simple Set** | 每流周期从 **固定 2 值** 取，如 `{4, 8}` ms | 低 | 接受率高、峰值低、易调度 |
| **Random Set** | 每流周期从 **全集** `{2,4,8,16,32}` ms 均匀随机取 | 高 | 接受率较低、峰值高、考验鲁棒性 |

- 评测矩阵：`规模(5) × 集合(2) × seed(≥5)` = 至少 50 个场景。
- 目的：在**相同流数**下，对比 Simple/Random 两集，量化"周期多样性"对峰值、接受率、耗时的影响（复刻论文 Fig.10 的 (a)(b) vs (c)(d) 对比）。

### 2.2 评测指标（对齐论文 + 我们的物理意义）

| 指标 | 含义 | 论文对应 |
|---|---|---|
| `effective_peak_load` | 归一化峰值负载（含丢弃惩罚） | Max Slot Occupation |
| `scheduling_success_rate` | 调度成功流/包占比 | Acceptance Rate |
| `true_deadline_satisfaction_rate` | 截止期满足率 | — |
| `drop_ratio` | 丢弃率 | — |
| 推理时间 / 训练时间 | 单场景求解耗时 | Time Cost |
| reward 曲线 | 训练收敛过程 | Fig.8 / Fig.9 |

> **基线定义（2026-09-06 更新）**：**对比基线 = native（未降维）同算法**。降维（dimred）方法的唯一合法对比对象是「同一算法在 `action_mode="native"` 下的结果」，用于回答"动作空间 96→7 维降维的代价/收益"。`random_feasible`（随机顺序+随机放置）、EDF/urgency 启发式、MILP 最优统一作为**参考标尺（reference）**，仅标注绝对水平，不进对比。

### 2.3 算法对比矩阵（离散原生三方）

> **选择依据（关键）**：当前 agent 是 **96 维离散动作 + action masking**，本质是离散决策。算法必须**原生支持离散 + 干净处理 mask**，否则对比不公平。据此：

| 类别 | 算法 | 动作范式 | 状态 |
|---|---|---|---|
| 降维对比基线 | native 版本（PPO/SAC/DDQN/TD3） | Discrete(96) | ✅ 已有 |
| 降维方法 | dimred 版本（PPO/SAC/DDQN/TD3） | MultiBinary(7) | ✅ 已有（TD3 新增） |
| 参考标尺 | `random_feasible`（随机下界） | — | ✅ 已有 |
| 参考标尺 | `edf_min_load`, `urgency_lexicographic`（启发式） | — | ✅ 已有 |
| 参考标尺 | **MILP**（scipy 最优） | — | ✅ 已有（≤50 流） |

**为什么不用 DDPG（针对当前离散 agent）**：DDPG 是**连续控制-only**算法（确定性策略 μ(s)+噪声），原论文用它是因为其 MDP 是连续的（每流输出连续 link+offset）。在我们 96 维离散 + mask 的 MDP 上，DDPG 必须靠 Gumbel-Softmax 松弛硬套、mask 也不自然，对比不公平且结论易被"松弛质量"而非算法本身干扰。故**第一阶段剔除 DDPG**，用离散原生的 **Double-DQN** 替代，使三方恰好覆盖三种不同范式（策略梯度 / 最大熵策略梯度 / 价值法），RQ3 升级为"哪类范式最适配本场景"。

> 若后续要**复刻论文的 SAC/PPO/DDPG 三方对比**，需先把 env 改为**连续映射 MDP**（Plan B：每流输出连续 link+offset），DDPG 才原生——列为可选升级，不在第一阶段。

- **公平性原则**：每个算法的 native 与 dimred 版本在**同一 env、同一状态表示、同一奖励**下训练对比；mask 统一以"不可行动作 logits/Q 置 -inf"实现；`random_feasible`/启发式/MILP 作为参考标尺，仅标注绝对水平。
- **不预设谁最优**：论文结论是 SAC 最优，但本场景（包级、离散放置、RB 硬约束）可能不同，三方需实跑后综合给结论（尤其关注 Double-DQN 在大动作空间+mask 下是否稳定）。

### 2.4 MDP / 动作空间一致性

当前 env 是**顺序式离散** MDP：每步对"当前包"输出 `(link, slot)`（96 维离散，含 mask），包顺序**硬编码 EDF**。

**结论（已定）**：因为我们选定的三方 **PPO / 离散 SAC / Double-DQN 全部原生支持离散 + mask**，所以**保持当前离散顺序 env（即原方案 A）就是算法学上正确的选择**，而非"退而求其次的最小改动"。SAC 结果可直接复用，无需重写 env。

- **方案 A（采用，离散顺序 env）**：PPO(categorical)、离散 SAC、Double-DQN 均为原生离散，mask 处理干净，对比公平。← **第一阶段就走这条**。
- **方案 B（可选升级，连续映射 env）**：改为每流输出连续 `(q_i→link, u_i→slot offset)`，`2×flow_count` 维。此时 SAC/PPO/DDPG 全部原生连续，可复刻论文 SAC/PPO/DDPG 三方对比，也支持"让 agent 学调度顺序"（见 `docs/planB_design.md`）。缺点：重写 env + 接口，约 1~2 周，且会偏离当前已验证的离散管线。

> 第一阶段聚焦方案 A；方案 B 视论文需要（是否要复刻 SAC/PPO/DDPG 或让 agent 学顺序）再决定是否投入。

### 2.5 评估协议

1. **每场景跑 N=5~10 个 seed**，报告 `mean ± std`（论文用 10 次独立仿真）。
2. **训练**：每算法固定总 episode/步数；SAC/PPO/DDPG 各自调出最佳超参（LR、γ、batch 等），调参过程可附消融（如 γ 扫描，复刻论文 Fig.8）。
3. **输出物（直接对应论文图表）**：
   - 训练 reward 曲线（SAC/PPO/DDPG 对比，复刻 Fig.9）。
   - 各规模下 Simple/Random 集的 **峰值负载 + 耗时** 双指标表（复刻 Fig.10(a)(c)）。
   - 各规模下 Simple/Random 集的 **接受率** 表（复刻 Fig.10(b)(d)）。
   - 规模 vs 峰值 / 接受率 折线（鲁棒性随规模变化）。
4. **公平性校验**：所有 RL 用同一测试场景集评估（同 seed 同场景），避免训练/测试场景错位。

---

## 3. 与现有代码的映射（实现清单）

**已有（可直接用）**
- `tsn_sim/env.py`：顺序式离散 env（96 动作）
- `tsn_sim/sac.py` + `train_sac.py`：SAC（load_balance, γ=0.5, 固定 α=0.1）
- `tsn_sim/heuristics.py`：`random_feasible` / `edf_min_load` / `urgency_lexicographic`
- `tsn_sim/milp.py`：scipy MILP（≤50 流最优）
- `tsn_sim/metrics.py`：effective_peak_load、scheduling_success_rate、deadline 满足率
- `tsn_sim/config.py`：`flow_count`、`packet_size_bits`、`periods_ms`

**待实现**
1. `tsn_sim/ppo.py` + `train_ppo.py`：categorical PPO（mask 经 logits 置 -inf），接现有 env。
2. `tsn_sim/double_dqn.py` + `train_double_dqn.py`：Double-DQN，Q 网络输出 96 维，不可行动作 Q 置 -inf。
3. `config.py` 增加 `period_mode: "simple" | "random"`（simple=固定 2 周期，random=全周期随机）。
4. `scenario.py` 支持按 `period_mode` 生成 Simple/Random 集。
5. `eval_benchmark.py`：遍历 `规模 × 集合 × seed × 算法`，输出表格 + 收敛曲线。
6. `generate_benchmark_figures.py`：复刻论文 Fig.8~10 风格图表。

---

## 4. 论文章节对应的研究问题（RQ）

- **RQ1（规模扩展性）**：流数增大时，各算法的峰值/接受率/耗时如何变化？native 相对参考标尺（random/启发式）的水平如何随规模扩大？
- **RQ2（周期复杂度鲁棒性）**：Simple vs Random 集下，各算法指标差异多大？哪种算法在 Random 集（高复杂度）上最稳健？
- **RQ3（RL 算法选型）**：**PPO（策略梯度）/ 离散 SAC（最大熵策略梯度）/ Double-DQN（价值法）**，哪类范式最适配本场景的离散+mask 决策？是否 SAC（最大熵）最优，还是本场景下 PPO/DQN 反超？（**不预设，实证**）
- **RQ4（与最优的差距）**：native/dimred 相对 MILP 最优的 gap 多大？相对 random 标尺（1.0）的水平如何？时间开销权衡如何？

---

## 5. 实施步骤（建议顺序）

1. **阶段 0（配置）**：在 `config.py` / `scenario.py` 加入 `period_mode`（simple/random），确认 Simple/Random 集能生成且指标合理（参考 `check_physical_config.py`）。
2. **阶段 1（PPO）**：实现 categorical PPO，在 50 流+seed=7 上先跑通，对比离散 SAC。
3. **阶段 2（Double-DQN）**：实现 Double-DQN，同场景对比。
4. **阶段 3（benchmark harness）**：写 `eval_benchmark.py`，跑全矩阵（规模×集合×seed×算法）。
5. **阶段 4（绘图）**：生成 Fig.8~10 风格结果图。
6. **阶段 5（论文写作）**：按 RQ1~4 组织结果，给出综合结论（含"哪类 RL 范式最适配本场景"的实证判断）。

---

## 6. 待确认（已基本敲定，仅余可选项）

- ✅ **算法三方**：PPO + 离散 SAC + Double-DQN（离散原生，剔除 DDPG）；DDPG 仅作为可选连续 MDP(Plan B) 升级项。
- ✅ **规模档位**：`{10, 20, 50, 100, 200}`（因含频域 RB 硬约束，比论文 600 流更小，合理）。
- ❓ **可选**：是否额外投入 Plan B（连续 MDP）以复刻论文 SAC/PPO/DDPG 三方对比，或让 agent 学调度顺序？**默认不做，除非论文需要**。
