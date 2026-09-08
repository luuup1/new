# 方案B 设计文档：让 DRL Agent 同时学习「调度顺序」与「放置位置」

> 目标：突破当前"固定 EDF 顺序 + 仅学放置"MDP 的天花板，使 DRL 真正超越固定顺序启发式，逼近 MILP 最优。

---

## 1. 背景与动机

**当前 MDP（路线1 既定设计）**
- 包顺序**写死为 EDF**（最早截止时间优先），env 按序把包一个个递给 agent。
- agent 每步只对"当前这个包"输出 `(link, slot)`：动作空间 = `L×S = 3×32 = 96`（扁平 Discrete）。
- 奖励：`load_balance`（`-cell_load` 步奖）+ 终端 `-effective_peak_load`。

**已验证的局限**
- 全局峰值是 **episode 级统计量**，单步放置几乎不改变它。
- 任何**局部**奖励都只能把 agent 引导到"局部 min_load"——而这正好等于固定顺序启发式本身。
- 实测：DRL（仅放置）eff_peak=0.727 vs `urgency_lexicographic`=0.704，仅差 **-3.57%**，无法超越。
- MILP 最优 peak=**0.545**（gap 49.6%）——MILP 同时优化"顺序+放置"，这就是 DRL 缺的那块自由度。

**方案B 的核心想法**
把"先调度哪个包"也变成动作的一部分 → agent 同时决策 **(packet, link, slot)**。这样 agent 可以：
- 先把大包放进余量大的链路（如 L3），再用小包填补 L1 缝隙；
- 故意延迟某个灵活包，为后续关键包预留热点 cell；
- 本质上获得与 MILP 同等的决策自由度。

---

## 2. 新 MDP 定义

### 2.1 State（变化最大）
当前 obs 只含"单个当前包"特征。方案B 必须让 agent 看到**整个未调度包池**。

```
obs = [
  grid_load   : L×S = 96        # 各 cell 当前负载比例
  channel_eff : L = 3           # 各链路频谱效率（bits/RB）
  global_stats: k              # 已调度数/总包数、剩余需求总量、剩余时隙等
  pool        : N_max × d_pkt  # 未调度包特征，pad 到最大包数，配 valid mask
]
```

每个包特征 `d_pkt` 建议含：
- `bits`（包大小）
- `deadline_window`：最早/最晚可调度 slot（相对超周期）
- `available_links`：one-hot 编码（哪些链路可用）
- `arrival_ms`、`urgency = 1/window_width`
- `required_rb_per_link`：在各链路所需 RB（用于可行性判断）

> **实现注**：当前 SAC 接受定长向量 obs（现 199 维）。方案B 要么
> (a) 把 pad 后的 pool 扁平化进定长向量（简单，维度 = 96+3+k+N_max×d_pkt）；
> 要么 (b) 改造 SAC 接受结构化/序列 obs，配合注意力编码器（更省参数、更干净）。
> 见 §4。

### 2.2 Action
动作从"选 cell"扩展为"选 **(包, cell)**"：
```
a = (p, link, slot),  p ∈ 未调度包池,  cell ∈ {1..L}×{0..S-1}
```

### 2.3 Transition
- `(p, cell)` 合法 → 占用该 cell RB，把 `p` 从 unscheduled 池移除，写入 schedule。
- `(p, cell)` 非法（p 不在池 / cell 对 p 不可行 / 超时）→ `p` 标记 infeasible、移出池、跳过，给惩罚。
- **池空 → episode 结束**（不再固定 310 步）。

### 2.4 Reward（沿用并增强）
- 步奖：`load_balance`（`-cell_load`），可叠加增量峰值惩罚 `-2.0·peak_increase`。
- 终奖：`-effective_peak_load`。
- 信用分配增强（§6）：加 n-step / λ-return，让"为后续包预留关键 cell"的全局收益回传到当前决策。

### 2.5 Masking（关键改动）
掩码从 1-D（96）升级为 **2-D**：对每个未调度包 `p`，计算其可行 `(link, slot)` 集合；只有 `(p, cell)` **联合可行**的动作才放开。实现为 `P × 96` 布尔矩阵（P 随步数减少，pad 到 N_max）。

### 2.6 Episode 长度
动态：直到池空为止。最坏 310 步（每步调度 1 个），也可能因丢弃而更短。

---

## 3. 环境改动（`tsn_sim/env.py`）

| 现有 | 改为 |
|---|---|
| `self._packets` + `self._current_idx` | `self._unscheduled: List[packet_key]`（包池） |
| `reset()` 顺序固定遍历 | 初始化池为全部包（离线调度，全部已知） |
| `step(action)` 解析单 cell | 解析 `(p, cell)` → 校验 → 更新 → 池移除 |
| `action_masks()` 返回 96 维 | 返回 `P×96` 联合可行掩码（pad 到 N_max） |
| `_get_obs()` 单包特征 | 编码 grid + channel + global + **pool**（pad + valid mask） |
| `render()` / info | 相应调整（显示池大小、已调度/丢弃计数） |

---

## 4. 网络架构三方案

### 方案 B1：扁平联合离散（最易实现，先跑通证伪）⚡
- 动作 = `Discrete(N_max × 96)`（≈ 310×96 ≈ **29760**），用联合掩码把不可行动作 logit 置 `-inf`。
- **复用现有 SAC**，只需：action_space 96→N_max×96；obs 加入 pad 池块。
- 优点：改动最小，1~2 天可跑通。
- 缺点：softmax 量级大且极稀疏，样本效率差；N_max 变化需 pad。

### 方案 B2：双头结构（折中）
- 共享 backbone 编码 obs；两个头：
  - `packet_head`：在 unscheduled 池上 masked softmax → 选 `p`；
  - `place_head`：在 96 cell 上 masked softmax（用 `p` 的 feasibility 掩码）→ 选 cell。
- 训练时联合似然 `log π(a) = log π(p) + log π(cell|p)` 送入 SAC actor loss。
- 优点：比 B1 样本效率高，仍用定长 obs。缺点：两阶段独立采样需保证联合可行。

### 方案 B3：指针 / 注意力网络（推荐，论文贡献强）★

```
Encoder: grid + 每个未调度包 → 向量序列  [h_grid, h_p1, h_p2, ...]
Decoder:
  (1) Attention over pool → 指针选 p        (可解释：attention 权重=调度优先级)
  (2) (h_p + h_grid) → placement head → 选 (link, slot)
```
- 天然处理**变长包池**、参数与包数无关、注意力权重可解释。
- 优点：最贴近"学习调度策略"的叙事，泛化好，是论文核心卖点。
- 缺点：实现量最大（需改 SAC 接受序列 obs + attention actor），约 1 周。

**建议路线**：先用 **B1** 低成本验证"学顺序能超越固定顺序启发式"这一假设（证伪/证实都只需 1~2 天）；确认有效后，再投入 **B3** 做论文级实现。

---

## 5. 训练改动（`train_sac.py` / `tsn_sim/sac.py`）

- action_space 维度、obs 接口按所选方案调整。
- Actor/Critic 网络按 B1/B2/B3 重写（主要在 `sac.py` 的 Actor/Critic 类）。
- **沿用已验证的稳定配置**：固定 `alpha=0.1`、`gamma=0.5`、`load_balance` 奖励、action masking。
- 评估：固定 seed=7 场景 + 开 `multi_scenario` 各训一轮，报告 eff_peak、丢弃率、与固定顺序启发式/MILP 的 gap。
- 保留 best/last checkpoint、H/Q 诊断日志。

---

## 6. 信用分配增强（配合方案B）

即便有了顺序自由度，终端峰值的 episode 级延迟仍在。建议：
- 保持 `load_balance` 步奖（强局部信号）；
- 加 **n-step（k=3~5）** 或 **λ-return**，让"为后续大包预留关键 cell"的全局收益回传到当前决策；
- 若仍不稳，试 `gamma=0.7~0.9`（比纯 placement 时更稳，因为顺序决策使步间相关性更强）。

---

## 7. 公平对比协议

| 对比对象 | 说明 | 期望 |
|---|---|---|
| `edf_min_load` / `urgency_lexicographic` | 固定顺序启发式（参考标尺） | DRL-B 应**明显超越** |
| **MILP optimal (peak=0.545)** | 上限（同享顺序+放置自由度） | DRL-B 的目标天花板 |
| DRL-A（仅放置，固定 EDF） | 自身消融 | 证明"学顺序"带来的增益 |

> 关键点：原启发式固定顺序，本身处于劣势；DRL-B 与 MILP 才在同一起跑线（都可自选顺序）。**若 DRL-B 能逼近 MILP，即证明"EDF 顺序并非最优"——这是有力的论文结论。**

---

## 8. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 动作空间爆炸（B1，~3万） | 用联合掩码 + 先 B1 证伪，再上 B3 |
| 训练不稳定 | 固定 `alpha=0.1`（已验证）+ n-step + 分期调 gamma |
| 过拟合单场景 | `multi_scenario` 打开训泛化轮 |
| 顺序自由度反而更差（agent 乱序致丢弃） | infeasible 强惩罚 + 早停守护 |

---

## 9. 实施步骤（分阶段）

1. **Phase B0（1~2天）**：env 重构为"包池 + 联合动作"，B1 扁平实现，obs 加入 pad 池；跑通单 episode。
2. **Phase B1（1天）**：SAC 接 B1，固定 `alpha=0.1` + `load_balance`，seed=7 训 200 ep，看是否超越 0.704。
   - 超越 → 假设成立，进 B3。
   - 仍不匹配 → 查 masking/奖励，或问题本身顺序无关（罕见）。
3. **Phase B2（~1周）**：实现 B3 指针网络 + 结构化 obs + attention actor。
4. **Phase B3**：`multi_scenario` 泛化训练 + 完整对比表 + 出图。

---

## 10. 成功标准

- DRL-B (eff_peak) **<** `urgency_lexicographic` (0.704)，理想逼近 MILP (0.545)；
- 丢弃率 = **0%**；
- 在 `multi_scenario` 下仍稳定优于固定顺序启发式。

---

*附：当前物理配置（seed=7，已复检）— 50 流 / 310 包实例，3×32=96 cell，总容量 5836 RB，100% 调度成功、0 丢包、峰值 0.816、利用率 20.6%。配置合法、有冗余、问题有意义（MILP gap 49.6%）。*
