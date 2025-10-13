import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# 确保可视化目录存在
os.makedirs('visualizations', exist_ok=True)

# 假设数据在1000_results.csv中
df = pd.read_csv('1000_results.csv')

def create_rank_consistency_plot():
    # 计算Edit Score和Human Score的排名
    df['edit_score_rank'] = df['editscore'].rank()
    df['human_score_rank'] = df['gpt_score_norm'].rank()  # 使用gpt_score_norm作为human_score
    
    # 计算Spearman和Kendall相关系数
    spearman_corr, spearman_p = stats.spearmanr(df['editscore'], df['gpt_score_norm'])
    kendall_corr, kendall_p = stats.kendalltau(df['editscore'], df['gpt_score_norm'])
    
    # 创建图形
    plt.figure(figsize=(10, 8), dpi=100)
    
    # 创建六边形箱图
    hb = plt.hexbin(df['edit_score_rank'], df['human_score_rank'], 
                   gridsize=30, cmap='Blues', 
                   mincnt=1, bins='log')
    
    # 添加对角线表示完美一致性
    max_rank = max(float(df['edit_score_rank'].max()), float(df['human_score_rank'].max()))
    plt.plot([0, max_rank], [0, max_rank], 'r--', label='Perfect Consistency')
    
    # 添加相关系数文本框
    corr_text = f"Spearman: {spearman_corr:.2f} (p={spearman_p:.3f})\nKendall: {kendall_corr:.2f} (p={kendall_p:.3f})"
    plt.text(0.05, 0.95, corr_text, transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # 设置标题和标签
    plt.title('Rank Consistency between Edit Score and GPT Score', fontsize=16)
    plt.xlabel('Edit Score Rank', fontsize=14)
    plt.ylabel('GPT Score Rank', fontsize=14)
    
    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 添加颜色条
    cb = plt.colorbar(hb)
    cb.set_label('log10(count)')
    
    # 添加图例
    plt.legend(loc='lower right')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    plt.savefig('visualizations/rank_consistency_human_score.png', dpi=300, bbox_inches='tight')
    print("等级一致性图表已保存到 'visualizations/rank_consistency_human_score.png'")
    
    # 关闭图形
    plt.close()

if __name__ == "__main__":
    create_rank_consistency_plot()