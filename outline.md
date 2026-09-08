# 5G-TSN 多链路确定性调度 —— 项目宏观概述（outline）

> 本文档从宏观角度梳理整个项目的思路、结构与数据流，帮助建立整体认知，不深入具体实现细节。

---

## 1. 总体目标与功能定位

**一句话定位**：一个面向论文的「5G-TSN 多链路确定性调度」研究管线，核心目标是**最小化 `effective_peak_load`（归一化峰值负载 + 丢包惩罚）**，同时满足到达时间、截止时间（deadline）、链路可用性与 RB（资源块）容量等约束。

项目经历了三个演进阶段，功能也随之分层：

1. **启发式仿真器（第一阶段）**：用经典调度启发式（随机 / EDF / 紧迫度）快速求解，并用 MILP 精确求解器做小规模最优参考，验证问题建模是否成立。
2. **DRL 环境 + 三算法（第二阶段）**：将调度问题封装为 Gymnasium 环境，用三种离散原生强化学习算法（PPO / 离散 SAC / Double-DQN）学习「包放置」策略。
3. **实验评估与论文出图（第三阶段）**：统一口径、多种子、多规模对比，产出论文所需的表格与图（figure1/3/4 等）。

**研究导向**：以论文 *SAC-FSO（Computer Networks 2025）* 的方法论为参照，回答四个研究问题——RQ1 规模扩展性、RQ2 周期复杂度鲁棒性、RQ3 RL 算法选型、RQ4 与 MILP 最优的差距。

---

## 2. 主要模块划分及各模块职责

### 2.1 核心库 `tsn_sim/`（领域层，可复用）

| 模块 | 职责 |
|---|---|
| `config.py` | 配置对象（`SimulationConfig`/`HeuristicConfig`）与默认值、JSON 读写 |
| `models.py` | 核心数据模型（`Flow`/`Packet`/`Candidate`/`ScheduleEntry`/`Scenario`/`SimulationResult`） |
| `scenario.py` | 场景生成：流、链路、RB 容量、信道质量，支持 `period_mode`（cyclic/simple/random） |
| `candidate.py` | 包实例化 + 可行候选生成（时间窗、链路、RB、deadline 约束过滤） |
| `metrics.py` | 指标计算（峰值负载、丢弃率、成功率、利用率等）与 CSV/JSON 导出 |
| `stats.py` | 问题规模诊断（包数、候选数、MILP 变量/约束规模） |
| `heuristics.py` | 经典启发式：`random_feasible`/`edf_min_load`/`urgency_lexicographic`/`edf_min_peak` |
| `milp.py` | 基于 scipy 的小规模 MILP 最优求解器（≤50 流） |
| `experiment.py` | 实验编排：单次运行、链路数对比、负载扫描、启发式对比、MILP gap 对比 |
| `env.py` | **Gymnasium 环境** `TSNSchedulingEnv`（MDP 核心，最大模块） |
| `ppo.py` / `sac.py` / `ddqn.py` | 三种 DRL 算法实现（离散 + 动作掩码） |
| `bc.py` | 行为克隆（BC）暖启动：用 min-load 教师演示预训练 actor |
| `__init__.py` | 包对外统一导出（env、四种 agent、场景构建、启发式等） |

### 2.2 训练入口（应用层，根目录）

- `train_ppo.py` / `train_sac.py` / `train_ddqn.py`：三个训练脚本，含 warmup、eval、early-stop、checkpoint 保存。

### 2.3 评估与绘图（产出层，根目录）

- `eval_plot_data.py`：生成 `results_plot_data.json`（多种子评估数据）。
- `plot_figures.py`：论文主绘图脚本（图1 方法对比、图3 训练曲线）。
- `plot_reward_curves.py` / `plot_reward_curves_seeds.py` / `plot_three_algo_comparison.py` / `duibi.py`：各专题出图脚本。
- 产物 PNG：`figure1_method_comparison.png`、`figure3*.png`、`figure4_eff_peak_convergence.png` 等；`figures/` 目录存启发式/MILP 专题图。

### 2.4 CLI 入口

- `tsn_scheduler.py`：命令行入口，支持启发式/MILP 对比、负载扫描、导出 CSV/JSON。

### 2.5 配置 / 文档 / 产物

- `config.example.json`：配置文件示例。
- `EXPERIMENT_PROTOCOL.md`：**实验铁律**（基线定义、踩坑清单、正确流程，任务前必读）。
- `docs/experiment_design.md` / `docs/planB_design.md`：实验设计与 Plan B（学顺序）方案。
- `checkpoints_*/`：各算法 × 种子 × 配置的模型权重与 `training_curves.json`。
- `archive_scratch/`：历史诊断/验证脚本归档（非主线）。

---

## 3. 模块之间的调用关系与依赖结构

依赖方向自底向上，整体呈清晰的分层结构：

```
config ──────────────► models
                          ▲
scenario ◄─── (config, models)
candidate ◄─── (scenario, models)
metrics / stats ◄─── (candidate, models)
                          ▲
heuristics / milp ◄─── (candidate, metrics)
experiment ◄─── (heuristics, milp, scenario, metrics)
                          ▲
env ◄─── (config, models, scenario, candidate, metrics)
                          ▲
ppo / sac / ddqn ────►（sac 提供 MLP / ReplayBuffer 供 ppo、ddqn 复用）
bc ◄─── (ppo, env)
                          ▲
train_ppo / train_sac / train_ddqn ◄─── (env + 对应 agent)
tsn_scheduler ◄─── (experiment, metrics, stats, scenario)
plot_* / eval_plot_data ◄─── (env, agent, checkpoint, training_curves.json)
```

关键依赖要点：

- **`sac.py` 是算法层的公共底座**：`MLP`、`ReplayBuffer` 被 `ppo.py`、`ddqn.py` 直接复用，避免重复实现。
- **`env.py` 是 MDP 核心**：三种算法与 BC 都围绕同一个 `TSNSchedulingEnv` 接口（obs/action/mask/reward）工作，保证对比公平。
- **训练脚本 ↔ 环境**：训练脚本通过 `--order-mode` / `--period-mode` / `--reward-mode` 控制环境行为，这是实验口径统一的关键开关。

---

## 4. 核心数据流向与关键技术栈

### 4.1 一次调度的数据流

```
配置(config)
  → 场景生成(scenario)：流 + 链路 + RB容量 + 信道质量
  → 包实例化 + 可行候选(candidate)：时间窗/链路/容量/deadline 过滤
  → 调度决策（三选一）：
       ① 启发式 heuristics（顺序 + 评分选格）
       ② MILP 精确求解（≤50 流）
       ③ DRL env（逐步放包）
  → 指标计算(metrics)：effective_peak_load / 成功率 / 丢弃率 / 利用率
  → 结果导出(CSV/JSON) 或 绘图(PNG)
```

### 4.2 DRL 的 MDP 数据流

- **Episode**：调度一个超周期（约 310 个包，50 流场景）。
- **State（obs，199 维 full 模式）**：`grid_load(96) + channel_eff(96) + packet_features(7)`。
- **Action**：`Discrete(L×S)` = 3 链路 × 32 时隙 = 96 维（含动作掩码 mask）；另有 `dimred` 降维分支（7-bit 二进制动作）。
- **Reward**：`r_t = -effective_peak_load`（统一奖励，峰值 + 丢弃惩罚）。
- **包顺序**：`order_mode` 决定（random / edf）。native 与 dimred 必须用同一 `order_mode="random"`，保证降维对比在同一 MDP 上公平。

### 4.3 关键技术栈

| 层次 | 技术 |
|---|---|
| 语言/数据结构 | Python + `dataclasses`（不可变数据模型） |
| 数值/优化 | `numpy` + `scipy.optimize.milp`（MILP 求解） |
| 深度学习 | `PyTorch`（MLP / LayerNorm / 掩码 softmax / Polyak 目标网络） |
| RL 接口 | `gymnasium`（Env / spaces / action masking） |
| 可视化 | `matplotlib`（论文图，300 DPI） |
| 训练运行环境 | Windows `E:\anaconda3\envs\RL_gpu`（torch 2.7.1 + CUDA），仓库本身跨平台 |

---

## 5. 潜在重点模块与可改进之处

### 5.1 重点模块（需重点关注）

- **`tsn_sim/env.py`**：MDP 的全部核心逻辑（obs 构造、动作掩码、奖励、顺序、降维投影）都集中在此，是理解项目行为的第一入口。
- **`tsn_sim/sac.py`**：最完整、最成熟的算法实现，且被 PPO/DDQN 复用，是算法层的「标准件」。
- **`tsn_sim/metrics.py` 的 `effective_peak_load`**：整个项目统一的 objective，口径必须一致。

### 5.2 可改进之处 / 潜在风险

1. **native 冷启动增益小**：PPO native 相对 `random_feasible`（随机标尺 1.0）仅 ~3%，核心原因是「关系型 argmin 放置」难以从 199 维 obs 冷启动学出。已有两条突破路径：
   - **BC 暖启动**（`bc.py`，已被方案甲从主线移除但留存）；
   - **Plan B（让 agent 学调度顺序）**（`docs/planB_design.md`），顺序才是逼近 MILP（0.545）的大杠杆。
2. **动作空间降维（dimred）是论文创新点**：`env.py` 已实现 `action_mode="dimred"`（7-bit），其对比基线是 native（未降维）同算法，而非启发式/random。SAC/DDQN dimred 因因子化 Q 学不动（1.040/1.032），已新增 TD3（连续松弛桥接）作为替代路线。
2. **动作空间降维未成主线**：`env.py`/`sac.py`/`ddqn.py` 已实现 `action_mode="dimred"`（7-bit），但仅作实验性分支，尚未纳入论文主线。
3. **实验缺口**：RQ1（多规模）、RQ2（Random 集）、RQ3（三算法 3-seed std）、RQ4（多规模 gap 曲线）、Phase 4 指标（吞吐/时延可靠性/时间开销）尚未系统采集。
4. **冗余与卫生问题**：
   - `tsn_sim/env_reward_v1_backup.py`、`archive_scratch/` 大量一次性脚本，可清理归档。
   - `truncate_ppo.py`、`check_*.py`、`_test_sac_dimred.py` 等一次性工具脚本散落根目录。
5. **跨平台编码坑**：部分 JSON 在 Windows（gbk）下写出、macOS（utf-8）下读取，绘图脚本已做 `utf-8 → gbk` 回退兼容，但仍是隐患。
6. **训练脚本默认参数坑**：三脚本 `--period-mode` 默认均为 `cyclic` 而非 `simple`，任何「统一对比」必须显式指定，否则口径不一致导致实验作废（已在 `EXPERIMENT_PROTOCOL.md` 明确）。
