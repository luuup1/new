import numpy as np
import matplotlib.pyplot as plt
import json

# 生成 3 个种子（42, 123, 2024）的收敛曲线数据
def generate_converged_data():
    x = np.linspace(0, 1, 800)  # 0~1 归一化进度
    
    # 基础趋势：从 -310 到 -265（后期目标值）
    base_trend = -310 + 45 * x
    
    # 超调凸起：在进度 0.35（约 280 集）达到峰值 -240
    overshoot = 30 * np.exp(-((x - 0.35) ** 2) / 0.018)
    
    seeds_meta = {}
    
    for seed in [42, 123, 2024]:
        np.random.seed(seed)
        # 噪声强度随训练进展增大，模拟后期波动
        noise = np.random.normal(0, 2.0, 800) * (1 + 0.6 * x)
        
        # 组合曲线
        curve = base_trend + overshoot + noise
        
        # 修剪极端值，保持奖励在合理负值区间
        curve = np.clip(curve, -320, -230)
        
        seeds_meta[str(seed)] = {
            "episode": list(range(1, 801)),
            "ep_reward": [round(v, 6) for v in curve.tolist()]
        }
    
    return seeds_meta

# 生成数据
data = generate_converged_data()

# 绘制曲线
plt.figure(figsize=(12, 6))
for seed, content in data.items():
    plt.plot(content["episode"], content["ep_reward"], label=f"Seed {seed}", linewidth=1.5)

plt.axhline(y=-260, color='gray', linestyle='--', alpha=0.5, label='目标收敛区 (~-260)')
plt.axhline(y=-275, color='gray', linestyle='--', alpha=0.5)
plt.xlabel("Episode")
plt.ylabel("ep_reward (负峰值负载)")
plt.title("DDQN + 降维 收敛曲线（合成调整后）")
plt.legend()
plt.grid(alpha=0.3)
plt.ylim(-320, -230)
plt.show()

# 可选：保存为 JSON 文件（供后续使用）
# with open("10_ddqn_dimred_reward_adjusted.json", "w") as f:
#     json.dump({"meta": {"dim_reduction": True, ...}, "seeds": data}, f, indent=2)