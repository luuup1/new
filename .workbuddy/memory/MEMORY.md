# 5G-TSN 项目长期记忆

## 项目概述
- 项目路径: e:\Master\codex
- 当前状态: 启发式调度仿真（baseline版本），后续计划逐步加入DRL
- 目标: 5G-TSN多链路调度，minimize peak_load，满足deadline约束

## 核心文件
- tsn_scheduler.py: CLI入口
- tsn_sim/config.py: SimulationConfig + HeuristicConfig
- tsn_sim/models.py: Flow, Packet, Candidate, Scenario, ScheduleEntry, SimulationResult
- tsn_sim/scenario.py: 场景生成（3链路×32slot，20流）
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

## MILP Gap 实验数据
- 5-15流: MILP gap=0%（问题太小，贪心已达最优）
- 20流: heuristic_peak=0.7442, MILP_peak=0.7273, gap=2.3%
- 30流: heuristic_peak=0.9250, MILP_peak=0.7674, gap=20.5%
- 30流下gap显著增大，说明问题规模上升后贪心策略有明显优化空间

## Baseline评估结论
- edf_min_load是最优启发式策略（5个seed下peak 0.74-0.84）
- peak/avg比约7-8:1，负载极度不均衡
- 100%deadline满足+0%丢包（约束满足没问题）
- Link1是瓶颈(peak=0.74)，Link3几乎空(peak=0.13)
- MILP在30流时有20%改善空间 → DRL的目标就是缩小这个gap

## DRL集成路线图
- Phase 0: 修复启发式baseline（当前阶段）
- Phase 1: Gym环境封装
- Phase 2: SAC智能体实现
- Phase 3: 训练管道
- Phase 4: 吞吐量与时延可靠性模块
- Phase 5: 论文实验

## 用户偏好
- 中文沟通
- 先分析根因再决定修改，确认后说"先改吧"
- 分阶段迭代，不追求一步到位
- 输出包含具体数值而非比例
- 所有episode运行至结束，不提前终止
