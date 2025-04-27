import matplotlib.pyplot as plt
import numpy as np

# --- 配置中文显示 (同前) ---
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    print("已设置字体为 SimHei")
except Exception as e:
    print(f"设置 SimHei 字体失败: {e}，请检查字体或更换。")

# --- 使用你从新脚本记录的真实数据 ---
thresholds = np.array([10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200, 300])
# 示例数据 (请务必替换为你的真实数据!)
keyframe_counts = np.array([542, 315, 220, 175, 148, 129, 105, 91, 80, 70, 60, 52])
coverage_rates = np.array([95.5, 92.1, 88.7, 85.3, 82.0, 79.5, 75.0, 71.2, 68.0, 65.1, 60.5, 55.2]) # 假设的覆盖率%
redundancy_rates = np.array([60.1, 45.2, 35.8, 30.5, 28.1, 26.0, 23.5, 21.0, 19.8, 18.5, 17.0, 16.1]) # 假设的冗余率%

# --- 开始绘图 ---
fig, ax1 = plt.subplots(figsize=(10, 6))

# --- 左 Y 轴：关键帧数量 ---
color1 = 'tab:blue'
ax1.set_xlabel("文本阈值 L$_s$ (字/帧)") # 使用 L_s 符号
ax1.set_ylabel("提取的关键帧数量", color=color1)
line1, = ax1.plot(thresholds, keyframe_counts, marker='o', linestyle='-', color=color1, label='关键帧数量')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, axis='y', linestyle=':', alpha=0.6)

# --- 右 Y 轴：覆盖率和冗余率 ---
ax2 = ax1.twinx()
color_cov = 'tab:green' # 覆盖率颜色
color_red = 'tab:red'   # 冗余率颜色
ax2.set_ylabel("比率 (%)", color='black') # 右轴标签设为黑色通用标签

# 绘制覆盖率曲线
line2, = ax2.plot(thresholds, coverage_rates, marker='^', linestyle='--', color=color_cov, label='覆盖率')
# 绘制冗余率曲线
line3, = ax2.plot(thresholds, redundancy_rates, marker='v', linestyle=':', color=color_red, label='冗余率')

ax2.tick_params(axis='y', labelcolor='black') # 右轴刻度也用黑色
ax2.set_ylim(0, 105) # 设置 Y 轴范围为 0-105%，留点空间
ax2.grid(True, axis='y', linestyle=':', alpha=0.6) # 可以给右轴也加网格

# --- 添加标题和合并图例 ---
plt.title("文本阈值 $L_s$ 对关键帧数量、覆盖率及冗余率的影响") # 标题，使用 $L_s$

# 合并三个曲线的图例
lines = [line1, line2, line3]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='best') # 图例放在最佳位置

fig.tight_layout()

# --- 保存图像 ---
output_filename = "threshold_metrics_curve.png"
plt.savefig(output_filename, dpi=300)
print(f"性能指标曲线图已保存为 {output_filename}")

plt.show()