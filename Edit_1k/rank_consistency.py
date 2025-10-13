import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, kendalltau

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)

# 加载数据
df = pd.read_csv('1000_results.csv')

# 确保可视化目录存在
import os
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

# 计算排序一致性
def plot_rank_consistency():
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
    edit_ranks = np.argsort(np.argsort(-edit_scores))  # 降序排列
    gpt_ranks = np.argsort(np.argsort(-gpt_scores))    # 降序排列
    
    # 计算相关系数
    spearman_corr, spearman_p = spearmanr(edit_scores, gpt_scores)
    kendall_corr, kendall_p = kendalltau(edit_scores, gpt_scores)
    
    # 创建散点图
    plt.figure(figsize=(10, 8))
    
    # 绘制散点图
    scatter = plt.scatter(edit_ranks, gpt_ranks, alpha=0.6, s=50, c=edit_scores, cmap='viridis')
    
    # 添加对角线（完美一致性参考线）
    max_rank = max(np.max(edit_ranks), np.max(gpt_ranks))
    plt.plot([0, max_rank], [0, max_rank], 'r--', alpha=0.7, label='Perfect Consistency')
    
    # 添加颜色条
    cbar = plt.colorbar(scatter)
    cbar.set_label('Edit Score Value')
    
    # 设置标题和标签
    plt.title(f'Rank Consistency between Edit Score and GPT Score\nSpearman: {spearman_corr:.2f}, Kendall: {kendall_corr:.2f}', fontsize=16)
    plt.xlabel('Edit Score Rank', fontsize=14)
    plt.ylabel('GPT Score Rank', fontsize=14)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('visualizations/rank_consistency.png', dpi=300, bbox_inches='tight')
    print("Rank consistency visualization saved to 'visualizations/rank_consistency.png'")
    plt.close()

if __name__ == "__main__":
    plot_rank_consistency()