import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, kendalltau
import matplotlib as mpl

# 设置绘图风格为学术论文风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 16

# 加载数据
df = pd.read_csv('1000_results.csv')

# 确保可视化目录存在
import os
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

# 计算排序一致性
def plot_academic_rank_consistency():
    # 确保列名正确
    edit_score_col = 'editscore'
    gpt_score_col = 'gpt_total_score'
    
    # 检查列是否存在
    if edit_score_col not in df.columns:
        print(f"Error: {edit_score_col} column not found")
        return
    if gpt_score_col not in df.columns:
        print(f"Error: {gpt_score_col} column not found")
        return
    
    # 获取两个评分
    edit_scores = df[edit_score_col].values
    gpt_scores = df[gpt_score_col].values
    
    # 计算排名
    edit_ranks = np.argsort(np.argsort(edit_scores * -1))  # 降序排列
    gpt_ranks = np.argsort(np.argsort(gpt_scores * -1))    # 降序排列
    
    # 计算相关系数
    spearman_corr, spearman_p = spearmanr(edit_scores, gpt_scores)
    kendall_corr, kendall_p = kendalltau(edit_scores, gpt_scores)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    
    # 创建六边形箱图以减少过度绘制
    hb = ax.hexbin(edit_ranks, gpt_ranks, gridsize=30, cmap='Blues', 
                   mincnt=1, bins='log', alpha=0.9)
    
    # 添加对角线（完美一致性参考线）
    max_rank = float(max(np.max(edit_ranks), np.max(gpt_ranks)))
    ax.plot([0.0, max_rank], [0.0, max_rank], 'r--', alpha=0.7, linewidth=2, 
            label='Perfect Consistency')
    
    # 添加颜色条
    cbar = plt.colorbar(hb, ax=ax)
    cbar.set_label('log10(count)', fontsize=12)
    
    # 设置标题和标签
    ax.set_title('Rank Consistency between Edit Score and GPT Score', fontsize=16, pad=15)
    ax.set_xlabel('Edit Score Rank', fontsize=14)
    ax.set_ylabel('GPT Score Rank', fontsize=14)
    
    # 添加相关系数文本
    correlation_text = f'Spearman: {spearman_corr:.2f} (p={spearman_p:.3f})\nKendall: {kendall_corr:.2f} (p={kendall_p:.3f})'
    ax.text(0.05, 0.95, correlation_text, transform=ax.transAxes, 
            fontsize=12, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 添加图例
    ax.legend(loc='lower right', frameon=True)
    
    # 设置网格线
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 设置轴范围
    ax.set_xlim(0.0, float(max_rank))
    ax.set_ylim(0.0, float(max_rank))
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('visualizations/academic_rank_consistency.png', dpi=300, bbox_inches='tight')
    print("Academic rank consistency visualization saved to 'visualizations/academic_rank_consistency.png'")
    plt.close()

if __name__ == "__main__":
    plot_academic_rank_consistency()