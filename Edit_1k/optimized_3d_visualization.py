import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 必须导入，用于3D投影
import os

# 确保可视化目录存在
os.makedirs('visualizations', exist_ok=True)

# 读取数据
df = pd.read_csv('1000_results.csv')

def create_optimized_3d_visualization():
    # 创建图形
    fig = plt.figure(figsize=(12, 10), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置视角 - 将z轴移到右侧
    ax.elev = 20
    ax.azim = -60  # 负值使z轴移到右侧
    
    # 绘制散点图，使用更好的颜色映射和透明度
    scatter = ax.scatter(
        df['image_cosine_sim'], 
        df['text_similarity'], 
        df['image_rls'],
        c=df['editscore'], 
        cmap='viridis', 
        alpha=0.8,
        s=30,  # 点的大小
        edgecolors='w',  # 白色边缘 (修正为edgecolors)
        linewidths=0.2  # 边缘线宽 (修正为linewidths)
    )
    
    # 添加颜色条并优化样式
    cbar = plt.colorbar(scatter, shrink=0.8, pad=0.1)
    cbar.set_label('EditScore', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)
    
    # 设置轴标签并优化样式
    ax.set_xlabel('Image Cosine Similarity', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('Text Similarity', fontsize=14, fontweight='bold', labelpad=10)
    ax.zaxis.set_rotate_label(False)  # 防止z轴标签旋转
    ax.set_zlabel('Image RLS', fontsize=14, fontweight='bold', labelpad=10)
    
    # 设置刻度标签大小 - 只使用支持的轴
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(labelsize=12)  # 这会应用到所有轴
    
    # 设置网格线样式 - 使用正确的方法
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 设置背景色为白色
    fig.patch.set_facecolor('white')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    plt.savefig('visualizations/optimized_3d_visualization.png', dpi=300, bbox_inches='tight')
    print("优化的3D可视化图像已保存到 'visualizations/optimized_3d_visualization.png'")
    plt.close()

if __name__ == "__main__":
    create_optimized_3d_visualization()