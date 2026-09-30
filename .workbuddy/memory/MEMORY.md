# 5G-TSN 项目长期记忆

## 项目概述
- **实际工作目录: e:\Master\new**（旧记 e:\Master\codex 已过时，作废）
- 目标: 5G-TSN 多链路调度，minimize effective_peak_load，满足 deadline 约束
- 当前阶段: Phase 3 论文实验设计（方案甲 cold 三算法对比口径）
- **运行环境: macOS（用户 2026-09-06 明确，以后都用 Mac）**，RL venv = `RL/bin/python`（torch 2.13.0 + gymnasium 1.3.0，CPU）
- **用户偏好：训练轮数自己决定，指令里显式给 --episodes（不依赖默认值）**

## 核心文件
- `tsn_scheduler.py` CLI入口; `train_ppo.py`/`train_sac.py`/`train_td3.py` 训练; `plot_figures.py` 唯一画图; `eval_plot_data.py` 出图数据
- `tsn_sim/env.py`(TSNSchedulingEnv, obs199/act96 native, act7 dimred); `tsn_sim/td3.py`(TD3Agent 连续松弛桥接); `tsn_sim/milp.py`(MILP 最优参考)
- `EXPERIMENT_PROTOCOL.md` 实验铁律（每次任务先读）

## 指标与基线铁律（2026-09-06 重大变更）
- effective_peak_load = peak_load + drop_ratio（α=1.0），统一 objective
- **对比基线 = native（未降维）同算法**：论文创新点是动作空间降维(dimred 96→7)，dimred 的唯一合法对比对象是同一算法 native 版本
- random_feasible(≈1.0)/edf_min_load(0.816)/urgency(0.667)/MILP(0.545) 统一降级为「参考标尺(reference)」，仅标注绝对水平，**不再作对比基线**
- 铁律: env 必须 `order_mode="random"`；native/dimred 同 MDP 成对对比；最终报告用 best checkpoint 不用 final
- 算法集: PPO/SAC/TD3

## ✅ dimred 学不动的真正根因 + 正解（2026-09-06 关键突破）
- **根因**：不是算法选型，而是「**价值函数被降维**」。SAC dimred 用因子化 Q `Q(a)=Σ q_bit[i]·bit[i]` 强制 Q 是 7 bit 线性组合；TD3 用 sigmoid 也隐含 bit 独立性假设。link/slot 强耦合 → 假设失效 → 全卡 1.0 学不动。
- **正解 = 策略降维 + 完整 critic**：actor 保持 7 维 Bernoulli（降维卖点），critic 输出完整 96 维 cell Q（价值函数无损）。7 位概率诱导 96 cell 分布 `π(c|s) ∝ Π p_i^{b_i(c)}(1-p_i)^{1-b_i(c)}`，mask 无效 cell 后 softmax 归一化。buffer 统一存 cell index(int)，critic gather 与 native 完全一致。
- **已实现**：`tsn_sim/sac.py` 已修复（`_cell_bits` 编码矩阵 + `cell_to_bits` + 重写 `_update_dimred`/`select_action`）；`train_sac.py` 训练循环适配 cell index
- **短训验证（seed42, 600ep, simple, γ=0.5）**：native best=0.9487 vs dimred best=0.9444 → **dimred 不再卡 1.0，追近 native（差 0.004）**

## ✅ TD3 同样修复成功（2026-09-06）
- **`tsn_sim/td3.py` 已重写**：critic 输入统一 96 维 cell 分布（不再吃 7 维 sigmoid），actor 7 维 logits→sigmoid→乘积诱导 96 cell 分布；`select_action` 返回 `(a_exec, a_cont)` 且 a_cont 恒 96 维；新增 `_cell_bits`/`cell_to_bits`
- **`train_td3.py`**：warmup 改随机 cell→96 维 one-hot；agent 传 n_links/n_slots
- **短训（seed42, 600ep, simple, γ=0.9）**：TD3 native best=**0.8095** vs dimred best=**0.7656** → **dimred 反超 native（低 0.044），实现真实性能优化**
- 这是用户核心目标「降维优化负载」的首个成功案例
- **待办**：TD3 正式训练 native/dimred 各 3 seed（2000ep/早停），出论文级对比 + std 带

## ⚠️ 训练脚本默认参数坑（关键，2026-07-20 确认）
- `train_ppo.py`: `--order-mode` 默认 **edf**（必须强制 random）, `--period-mode` 默认 **cyclic**, `--reward-mode` 默认 load_balance, `--gamma` 默认 0.99
- `train_sac.py`: `--order-mode` 默认 random, `--period-mode` 默认 **cyclic**, `--gamma` 默认 0.5
- **脚本 `--period-mode` 默认全是 cyclic（不是 simple）** → 任何"统一对比"必须显式加 `--period-mode simple`

## 已验证结果
### PPO 冷启动 3 种子（random-order + simple，2026-07-16 已跑）
- 配置: seed∈{42,123,2024} × 2000ep × `--order-mode random --period-mode simple --reward-mode load_balance --no-early-stop`
- 目录: `checkpoints_ppo_cold_seed{42,123,2024}`
- eval eff-peak: **0.976 / 0.973 / 1.000** → 均值≈0.983（仅比基线 1.005 降 ~3%）
- 结论: PPO cold 在 random env 学不动关系型 argmin，绝对增益仅~3%

### 当前算法 seed42（figure4 数据源）
- PPO: `checkpoints_ppo_cold_seed42`（0.976，random+simple ✅）
- SAC: `checkpoints_sac_seed42_0716`，best eval eff-peak≈**0.61**（520ep 早停，已收敛）
- ⚠️ SAC seed42 当时 period-mode 未知（默认 cyclic），方案甲统一 simple 需重跑

### 启发式 / MILP（simple 集 / 50 流 / random-order，20 评估种子）
- random_feasible: **1.0052 ± 0.0038**（唯一合法基线）
- edf_min_load: 0.817; urgency_lexicographic: 0.667（仅对照，非基线）
- MILP optimal: **0.545**（最优下界）
- 论文核心卖点: SAC cold 在 random 顺序下压到 0.61，**低于最佳 EDF 启发式 0.667，接近 MILP 0.545**

### 配置与 MILP gap（50 流 + 均匀 size 800-2400）
- 3链路×32slot=96 cell，episode 310 步
- 5-15流 gap=0%；20流 gap=2.3%；30流 gap=20.5%；**50流 gap=49.6%**（heuristic 0.816 vs MILP 0.545）

## 方案甲决策（2026-07-20，写论文主线，用户拍板）
- **统一 cold 三算法对比（PPO/SAC/TD3 全 cold），不讲 BC**（BC 从论文主线移除，checkpoints_ppo_simple_bc*/bc2 留存备查）
- 数据集口径统一 **simple 集**（对齐 figure1 主集）
- 三算法统一超参: `--order-mode random --period-mode simple --reward-mode load_balance`; PPO γ=0.99 / SAC γ=0.5 / TD3 γ=0.9
- figure1 改写: random_feasible + PPO/SAC/TD3(cold) + MILP，删 PPO+BC/Teacher 作为 headline
- figure3 改画 eff_peak（原 reward 不可比，用户已否）；figure4 补 3-seed std 带

## 待跑命令（方案甲，simple 口径，800ep）
- **PPO: 已齐（checkpoints_ppo_cold_seed*），无需跑**（纠正原"风险1"误判）
- SAC: seed42 重跑(simple) + seed123 + seed2024 → `checkpoints_sac_seed{42,123,2024}_0716`
- SAC seed42 重跑仅当原是 cyclic；若原已 simple 则冗余无害
- 运行环境: `E:\anaconda3\envs\RL_gpu\python.exe`（torch 2.7.1+cu118, CUDA MX450）

## 后续实验缺口（RQ 映射）
- RQ1 规模扩展性: flow∈{10,20,100,200}（现仅 50）
- RQ3 RL 选型: 三算法 3-seed std（补跑中）
- Phase 4 指标: 吞吐量 / 时延可靠性 / 调度成功率 / 时间开销（尚未系统采集）

## 用户偏好与约定
- 中文沟通；先分析根因再决定修改，确认后说"先改吧"
- 分阶段迭代；输出具体数值而非比例；所有 episode 跑至结束不提前终止
- 交互格式: 每次产出开头先发「泥嚎」
- 每次任务先读 `EXPERIMENT_PROTOCOL.md`

## 文献研读：ME-DDPG (Zhang et al., Ad Hoc Networks 2025)
- **核心思想**：DDPG 只输出 0/1 调度决策，再用 MCS-based priority heuristic 把 RB 实际分给 urgent Ft / Fv / non-urgent Ft；通过动作空间降维降低训练难度。
- **状态**：`{RBexp, L, Dnor, G}`；动作：M+N 维 0/1；奖励：对 Ft 用 `(dqos-di-dGCL)` 分段，对 Fv 用吞吐量阈值，并带 φi 缩放。
- **吞吐计算**：`sum(RB_scheduled,i × bytes_per_RB) / T`，其中 bytes_per_RB 由 SINR→MCS 查表得到（QPSK/16-QAM/64-QAM 三档）。
- **实验设计借鉴**：500 episodes × 500 TTIs；6 组用户配比（|Ft|:|Fv|=2:7~12:12）；3 种 GCL 配置；对比 O-DDPG/DPF；评估奖励收敛、时延、可靠性、吞吐量。
- **关键结果**：ME-DDPG 可靠性 100%，吞吐比 O-DDPG 提升 9.16%–10.84%，奖励收敛更快更稳。
- **对本项目启示**：
  1. DRL 决策 + 启发式执行的解耦思路可降低动作空间，尤其适合离散放置问题；
  2. 奖励函数可分段、带阈值和缩放因子，避免单一奖励曲线异常；
  3. CQI/MCS/bytes 映射表是吞吐量模块的可行实现路径；
  4. 多用户配比、多 GCL 配置的系统实验比单点实验更具说服力。

