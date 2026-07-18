# 5G-TSN 项目长期记忆

## 项目概述
- 项目路径: e:\Master\codex
- 当前状态: 启发式调度仿真（baseline版本），后续计划逐步加入DRL
- 目标: 5G-TSN多链路调度，minimize peak_load，满足deadline约束

## 核心文件
- tsn_scheduler.py: CLI入口
- tsn_sim/config.py: SimulationConfig + HeuristicConfig
- tsn_sim/models.py: Flow, Packet, Candidate, Scenario, ScheduleEntry, SimulationResult
- tsn_sim/scenario.py: 场景生成（3链路×32slot，50流，均匀size 800-2400）
- tsn_sim/candidate.py: 包实例化+可行候选过滤
- tsn_sim/heuristics.py: 3种启发式策略(random_feasible, edf_min_load, urgency_min_peak)
- tsn_sim/metrics.py: 指标计算
- tsn_sim/milp.py: MILP精确求解器(scipy)
- tsn_sim/experiment.py: 实验runner
- tsn_sim/stats.py: 问题规模诊断

## 已修复的问题（Phase 0 三项关键修复已完成）
1. ✅ deadline_satisfaction_ratio → 拆分为 scheduling_success_rate + true_deadline_satisfaction_rate（保留 backward compat alias）
2. ✅ peak_load 误导 → 新增 effective_peak_load = peak_load + drop_ratio（α=1.0），objective 统一用 effective_peak_load
3. ✅ urgency_min_peak 名不副实 → 改名 urgency_lexicographic（全局替换，代码+配置+测试+README）

## 仍存在的问题（优先级较低）
4. 候选生成遍历全超周期（32slot），大部分为无效遍历
5. 评分函数O(cells)复杂度，每次调用全量扫描
6. 资源极度不均衡：peak=0.91, average=0.11, utilization=8.7%
7. link_base/link_fading硬编码，应提取到config
8. HeuristicConfig 的 peak_weight/slot_load_weight 等参数已死（方案B改成了tuple），但还保留在config中

## 方案A/B实验结果（已完成）
- 方案A（纯peak tuple排序）：urgency_min_peak peak不变(0.909)，与方案B结果完全一致
- 方案B（字典序tuple评分）：同样无法改善urgency排序下的peak
- 核因：排序策略比选择策略更重要，urgency排序先处理窄窗包导致堆积
- edf_min_peak(EDF排序+纯peak选择) ≡ edf_min_load 完全一致
- 已新增edf_min_peak策略到代码库

## 当前配置（路线1调整后）
- flow_count=50, packet_size_bits=(800,1200,1800,2400)
- 3链路×32slot=96 cell，episode 310步
- 原始size(800-9600)在50流下过载(peak=1.0)，9600-bit大包独霸40RB cell
- 均匀size(800-2400)消除极端不均衡，peak/avg比大幅降低
- seed=7指标: edf=0.816, urgency=0.667, MILP=0.545, **gap=49.6%**
- 论文用100-200流是流级formulation(每步=1流)，我们包级formulation下50流是合理上限

## MILP Gap 实验数据
- 5-15流: gap=0%（问题太小）
- 20流: heuristic=0.744, MILP=0.727, gap=2.3%
- 30流: heuristic=0.925, MILP=0.767, gap=20.5%
- **50流+均匀size**: heuristic=0.816, MILP=0.545, **gap=49.6%** ← 当前配置

## Baseline评估结论（50流+均匀size配置）
### 基线定义（⚠️ 用户纠正过，重要）
- **真正的朴素 baseline = random_feasible**（随机顺序+随机选可行cell），seed=7 peak=1.000
- EDF/urgency 是"对随机调度的启发式改进"，不是基线，是待超越的启发式方法
- DRL 加入后应在"随机情况"上改进，而非在"EDF+随机"上改进
- 注意：env **原硬编码 EDF 顺序**，DRL 原实为"EDF顺序+学放置"含免费顺序增益。2026-07-15 已修复：SimulationConfig 增 `order_mode`(edf/random)，env._build_episode_data 支持随机顺序（用场景seed复现random_feasible顺序）。**正确实验必须用 order_mode="random"**，让 DRL 在基线上进行（随机顺序+学放置 vs 随机顺序+随机放置）
### 各方法 peak（seed=7，DRL 为 seed=42 训练 best）
- random_feasible: 1.000  ← 朴素基线
- edf_min_load: 0.816
- urgency_lexicographic: 0.667
- DRL(load_balance): 0.727
- MILP optimal: 0.545
### 各方法 peak（seed=7 启发式基线；DRL 同为 seed=42 训练 best）
- random_feasible: 1.000  ← 朴素基线（随机顺序+随机放置）
- edf_min_load: 0.816
- urgency_lexicographic: 0.667
- DRL(load_balance, EDF顺序env): 0.727  ← ⚠️ 此数在EDF顺序env上，含免费顺序增益
- MILP optimal: 0.545

### 对比口径（⚠️ 2026-07-15 重大修正：之前的"-27%"是错比法！）
- ❌ 旧结论"DRL(0.727) vs 随机基线(1.0)=降低27%" **无效**：DRL 在 EDF 顺序 env 训练（含免费EDF顺序增益），而基线在随机顺序 → 苹果比橘子。
- ✅ 修正后正确实验（order_mode=random，simple集seed=42）：
  - 真实朴素基线(random顺序+随机放置)均值 = **1.0049**（20随机种子；单次random_feasible=1.000）
  - PPO best(随机顺序+学放置) = **0.9722** → 相对基线 **+2.8%~3.2%**（真实但偏小）
  - 但 PPO 仅学到 min-load 贪婪的 22%（冷启动学不动关系型argmin），训练不稳定(final回退到1.0)
- 结论：在"基线上进行"的正确设定下，DRL 确实优于随机基线，但增益仅~3%，远不如手写min-load启发式(EDF顺序下0.667)。
- 要得到论文级增益，必须：C(IL/BC暖启动让agent学会min-load贪婪) 或 Plan B(让agent学包顺序，顺序才是大杠杆)。
- 100%deadline满足+0%丢包+100%调度成功（DRL在随机顺序env上仍成立）
- Env完全兼容: obs(199), action(96), 310步/episode（现支持 order_mode=edf/random）

## DRL集成路线图
- Phase 0: ✅ 修复启发式baseline（指标修正+策略验证+MILP对照）
- Phase 1: ✅ Gym环境封装 — `tsn_sim/env.py` (TSNSchedulingEnv)
  - MDP: State(199,) = grid_load + channel_eff + pkt_features, Action(96) = (link, slot)
  - Fixed EDF packet ordering, action masking for feasibility
  - Reward: -effective_peak_load (terminal), optional step shaping
  - Verified: 9/9 tests pass, exact match with baseline (eff_peak=0.7619)
  - 验证脚本: `verify_env.py`
- Phase 2: ✅ SAC训练修复完成（50流+均匀size配置）
  - 离散SAC + action masking + twin Q + **固定alpha=0.1** (auto-alpha已弃用，会塌缩)
  - **运行环境已切换到 RL_gpu** (torch 2.7.1+cu118, CUDA MX450)
  - 命令: `E:\anaconda3\envs\RL_gpu\python.exe train_sac.py ...`
  - env奖励模式: terminal/shaping/exponential/**load_balance**(最优)/mixed(失败)
  - **最优组合**: load_balance + gamma=0.5 + 固定alpha=0.1
  - 最佳结果: eff_peak=0.727 (best) vs baseline(urgency_lexicographic)=0.704, 差**-3.57%**
  - 关键修复1: alpha塌缩→固定0.1
  - 关键修复2: reward_mode映射bug（load_balance被吞成terminal，导致Q锁死）
  - best checkpoint: checkpoints_load_balance2/sac_best.pth
  - **无法靠调奖励超越baseline的根因**: 全局峰值是episode级统计量，局部奖励只能引导到局部min_load(=baseline)
  - 超越需改MDP(让agent学packet顺序)或gamma结构(n-step/lambda return)
  - 下一步: 用户决定接受匹配结论 / 改MDP超越 / 调gamma结构
- Phase 3: 🔄 论文实验设计（按 SAC-FSO 论文方法论，Computer Networks 2025 111119）
  - **参考论文**: Wang et al. "Towards wireless time-sensitive networking: Multi-link deterministic scheduling via DRL"
  - **数据集**: 多规模(flow_count∈{10,20,50,100,200}) × 2类(Simple Set=固定2周期 / Random Set=全周期随机)
    - 目的: 测试算法对不同**周期复杂度**的鲁棒性（复刻论文 Fig.10 的 simple vs random 对比）
  - **算法对比(离散原生三方)**: PPO(✅) + 离散SAC(✅) + Double-DQN(✅ 2026-07-15新增 ddqn.py+train_ddqn.py) + 3种启发式 + MILP(≤50流最优参考)
    - 用户最新(7/13): "不一定是PPO/DDPG，要针对当前离散agent决策选算法" → 当前agent=96维离散+mask，故选离散原生三方；DDPG(连续-only)剔除，仅作可选连续MDP(Plan B)升级；不预设谁最优，实证
  - **BC暖启动(路线C)**: ✅ 已验证论文级有效。PPO+BC(simple集, epochs=120/300演示) → eff_peak=**0.6944** (+30.56% vs 随机基线1.0)；BC greedy_match=0.887。冷启动无BC仅+3%(0.9722)。**BC预算是关键**：epochs=30/150演示仅 greedy_match=0.651，吃亏。
  - **指标**: effective_peak_load(=Max Slot Occupation) / scheduling_success_rate(=Acceptance Rate) / 时间开销
  - **MDP决策(已定)**: 方案A=保持离散顺序env(96离散+mask)，对PPO/离散SAC/Double-DQN均为原生正确选择→采用; 方案B=改连续映射env(复刻论文SAC/PPO/DDPG或让agent学顺序)降为可选升级，默认不做
  - **文档**: `docs/experiment_design.md` 已写；`docs/planB_design.md`(学顺序)暂搁置
  - **已落地文件(2026-07-15)**: tsn_sim/ddqn.py(DDQNAgent+bc_train_dqn)、train_ddqn.py、train_sac.py补--order-mode+修正baseline对比、tsn_sim/__init__.py导出DDQNAgent
- Phase 4: 吞吐量与时延可靠性模块
- Phase 5: 论文写作（按 RQ1~4 组织：规模扩展性/周期复杂度鲁棒性/RL选型/与MILP差距）

## 实验设计关键约束（用户纠正，务必遵守）
- **朴素 baseline = random_feasible**（随机顺序+随机放置），EDF/urgency 是启发式改进不是基线
- ⚠️ env 必须用 order_mode="random"，DRL 在"基线上进行"（随机顺序+学放置），不能用原EDF顺序env
- DRL 主叙事应是"vs 随机基线（随机顺序）的小幅改进(~3%)"，而非之前错误的"−27%"（那是EDF顺序DRL vs 随机基线，错比法）
- 论文导向：一切以可写进论文的实验对比为准；多规模+Simple/Random集+多RL算法(离散原生:PPO/离散SAC/Double-DQN，剔除DDPG)+启发式+MILP
- 要显著超越基线，需 IL暖启动(C) 或 让agent学顺序(Plan B)

## 用户偏好
- 中文沟通
- 先分析根因再决定修改，确认后说"先改吧"
- 分阶段迭代，不追求一步到位
- 输出包含具体数值而非比例
- 所有episode运行至结束，不提前终止

## 工作约定（用户强制，务必遵守 — 2026-07-15 起）
- **每次执行本项目相关任务前，必须先读根目录 `EXPERIMENT_PROTOCOL.md`**（实验执行协议：含所有错误要点、正确步骤、实验设计、铁律）。
- **读该协议后，每次产出内容开头先发「泥嚎」再继续输出**（用户明确要求的交互格式）。
- **基线铁律（违反任意一条 = 实验作废）**：
  1. env 必须用 `order_mode="random"`，DRL 在"基线上进行"（随机顺序+学放置）；
  2. 对比对象只能是 `random_feasible`（≈1.0），绝不和启发式(0.667/0.816) 比；
  3. 绝不用 EDF 顺序 env 训练出的 DRL 去对比随机基线（错比法，"-27%"已作废）；
  4. 最终报告用 best checkpoint，不用回退的 final 模型；
  5. 任何涉及基线/对比口径的变更，先读协议再动手。
- 协议文件索引见 `EXPERIMENT_PROTOCOL.md` 第 6 节。

## 项目根目录结构（2026-07-15 清理后）
- ⚠️ MEMORY.md 顶部"项目路径: e:\Master\codex"已过时，**实际工作目录为 `e:\Master\new`**。
- **根目录保留（核心 + 画图子系统，13 个文件）**：
  - 核心代码: `tsn_scheduler.py`(CLI入口), `train_ppo.py`/`train_sac.py`/`train_ddqn.py`(训练), `test_tsn_scheduler.py`(测试)
  - 配置/协议: `config.example.json`, `EXPERIMENT_PROTOCOL.md`, `README.md`
  - 画图子系统: `plot_figures.py`(**唯一画图代码文件**), `eval_plot_data.py`(出图数据生成), `results_plot_data.json`(图数据), `figure1_method_comparison.png`/`figure3_training_curves.png`(图产物)
  - 包代码: `tsn_sim/`
- **归档（可恢复，非永久删除）**: `e:\Master\new\archive_scratch\`（26 个文件）
  - 内容: 探索期诊断脚本(diag_*.py, analyze_env_params.py, check_*.py, find_*.py)、验证脚本(verify_*.py)、冗余画图脚本(generate_*.py)、PDF提取(extract_pdf.py+pdf_text.txt)、陈旧产物(metrics.json, schedule.csv, figure2_seed_distribution.png)
  - 若要永久删除，用户确认后 `Remove-Item archive_scratch -Recurse -Force`
- ⚠️ `checkpoints_*`(多个) 训练权重目录**未动**（非代码文件，保留以防丢失模型）；如需清理需单独确认。

## 冷启动 PPO 3种子实测（random-order env，2026-07-16 已验证）
- 配置: seed∈{42,123,2024} × 2000ep × `--rollout-episodes 10 --reward-mode load_balance --order-mode random --period-mode simple --no-early-stop`
- 产物: `checkpoints_ppo_cold_seed{42,123,2024}/training_curves.json`，画图 `figure3_training_curves.png`（3种子 mean±std，A1 平移正数 offset=+111）
- **最终 eval eff-peak**: seed42=0.976 / seed123=0.973 / seed2024=1.000 → 均值≈**0.983**，仅比随机基线(1.005)降 ~3%
- 结论: 冷启动 PPO 在 random-order env 收敛良好（曲线正数+上升+走平+std带，符合论文范式），但**绝对增益仅~3%**（只学到局部min-load贪婪）。论文核心卖点需 BC 暖启动（收敛到0.69）。
- 图3 现可交差作"收敛性证明"；要展示大幅超越基线需补 BC 3种子收敛曲线（加 `--bc-pretrain`）。
- 画图入口: `E:\anaconda3\python.exe plot_figures.py`（生成图1+图3；依赖 results_plot_data.json + checkpoints_ppo_simple_bc2/random 的 training_curves.json）。
