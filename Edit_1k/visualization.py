import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import pearsonr, spearmanr
from mpl_toolkits.mplot3d import Axes3D

# Ensure visualization directory exists
os.makedirs('visualizations', exist_ok=True)

# Read data
df = pd.read_csv('/Users/yinshuo/Documents/1y/code/BAGELScore/Edit_1k_0924/1000_results.csv')

# 1. Basic Statistical Analysis
def basic_stats():
    # Select columns for analysis
    stats_cols = ['image_rls', 'image_cosine_sim', 'text_similarity', 
                 'gpt_editing_accuracy', 'gpt_visual_quality', 
                 'gpt_content_preservation', 'gpt_style_consistency', 
                 'gpt_overall_effect', 'gpt_total_score', 
                 'editscore']
    
    # Calculate basic statistics
    stats_df = df[stats_cols].describe()
    
    # Print statistical results
    print("Basic Statistical Analysis Results:")
    print(stats_df)
    
    # Save statistics to CSV
    stats_df.to_csv('visualizations/basic_stats.csv')
    
    return stats_df

# 2. Correlation Analysis
def correlation_analysis():
    # Select numeric columns for correlation analysis
    corr_cols = ['image_rls', 'image_cosine_sim', 'text_similarity', 
                'gpt_editing_accuracy', 'gpt_visual_quality', 
                'gpt_content_preservation', 'gpt_style_consistency', 
                'gpt_overall_effect', 'gpt_total_score', 
                'editscore', 'editscore_norm', 'gpt_score_norm']
    
    # Calculate correlation coefficients
    corr_df = df[corr_cols].dropna().corr(method='pearson')
    
    # Create correlation heatmap
    plt.figure(figsize=(12, 10))
    
    # Custom color map
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#4575b4', 'white', '#d73027'])
    
    # Draw heatmap
    sns.heatmap(corr_df, annot=True, cmap=cmap, vmin=-1, vmax=1, 
                linewidths=0.5, annot_kws={"size": 8}, fmt=".2f")
    
    plt.title('Correlation Analysis of Evaluation Metrics', fontsize=16)
    plt.tight_layout()
    plt.savefig('visualizations/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Focus on correlation between editscore and gpt_total_score
    plt.figure(figsize=(10, 8))
    
    # Calculate Pearson and Spearman correlation coefficients
    pearson_corr, pearson_p = pearsonr(df['editscore'].dropna(), df['gpt_total_score'].dropna())
    spearman_corr, spearman_p = spearmanr(df['editscore'].dropna(), df['gpt_total_score'].dropna())
    
    # Draw scatter plot
    sns.regplot(x='editscore', y='gpt_total_score', data=df, 
                scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    
    plt.title(f'Correlation between EditScore and GPT Score\nPearson: {pearson_corr:.3f} (p={pearson_p:.3e})\nSpearman: {spearman_corr:.3f} (p={spearman_p:.3e})', 
              fontsize=14)
    plt.xlabel('EditScore', fontsize=12)
    plt.ylabel('GPT Total Score', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('visualizations/editscore_gpt_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return corr_df

# 3. Distribution Visualization
def distribution_visualization():
    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    
    # 1. EditScore distribution
    ax1 = fig.add_subplot(gs[0, 0])
    editscore_data = df['editscore'].dropna().values
    sns.histplot(x=editscore_data, kde=True, ax=ax1, color='skyblue')
    ax1.set_title('EditScore Distribution', fontsize=14)
    ax1.set_xlabel('EditScore', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    
    # 2. GPT total score distribution
    ax2 = fig.add_subplot(gs[0, 1])
    gpt_score_data = df['gpt_total_score'].dropna().values
    sns.histplot(x=gpt_score_data, kde=True, ax=ax2, color='salmon')
    ax2.set_title('GPT Total Score Distribution', fontsize=14)
    ax2.set_xlabel('GPT Total Score', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    
    # 3. Comparison of normalized scores
    ax3 = fig.add_subplot(gs[1, :])
    
    # Prepare data
    norm_data = pd.DataFrame({
        'EditScore': df['editscore_norm'].dropna(),
        'GPT Score': df['gpt_score_norm'].dropna()
    })
    
    # Draw violin plot
    melted_norm_data = pd.melt(norm_data)
    sns.violinplot(x='variable', y='value', data=melted_norm_data, ax=ax3, palette=['skyblue', 'salmon'])
    ax3.set_title('Comparison of Normalized EditScore and GPT Score Distributions', fontsize=14)
    ax3.set_ylabel('Normalized Score', fontsize=12)
    ax3.set_xlabel('')
    
    plt.tight_layout()
    plt.savefig('visualizations/score_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

# 4. GPT Score Dimension Analysis
def gpt_dimensions_analysis():
    # Select GPT score dimensions
    gpt_dims = ['gpt_editing_accuracy', 'gpt_visual_quality', 
                'gpt_content_preservation', 'gpt_style_consistency', 
                'gpt_overall_effect', 'gpt_total_score']
    
    # Calculate average for each dimension
    gpt_means_series = df[gpt_dims].mean()
    # Convert to DataFrame before sorting
    gpt_means_df = pd.DataFrame({'mean': gpt_means_series})
    gpt_means_sorted = gpt_means_df.sort_values(by='mean', ascending=False)
    
    # Draw bar chart
    plt.figure(figsize=(12, 8))
    bars = plt.bar(gpt_means_sorted.index.tolist(), gpt_means_sorted['mean'].values, 
                   color=sns.color_palette("muted", len(gpt_dims)))
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.title('Average Scores Across GPT Evaluation Dimensions', fontsize=16)
    plt.ylabel('Average Score', fontsize=14)
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('visualizations/gpt_dimensions_avg.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Draw boxplot to compare distributions across dimensions
    plt.figure(figsize=(14, 8))
    melted_df = pd.melt(df[gpt_dims].dropna(), var_name='variable', value_name='value')
    sns.boxplot(x='variable', y='value', data=melted_df, palette="muted")
    plt.title('Distribution of GPT Scores Across Evaluation Dimensions', fontsize=16)
    plt.ylabel('Score', fontsize=14)
    plt.xlabel('')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('visualizations/gpt_dimensions_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()

# 5. EditScore Component Analysis
def editscore_components_analysis():
    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    
    # 1. Relationship between image_rls and editscore
    ax1 = fig.add_subplot(gs[0, 0])
    sns.scatterplot(x='image_rls', y='editscore', data=df, alpha=0.6, ax=ax1)
    ax1.set_title('Relationship between Image RLS and EditScore', fontsize=14)
    ax1.set_xlabel('Image RLS', fontsize=12)
    ax1.set_ylabel('EditScore', fontsize=12)
    
    # 2. Relationship between image_cosine_sim and editscore
    ax2 = fig.add_subplot(gs[0, 1])
    sns.scatterplot(x='image_cosine_sim', y='editscore', data=df, alpha=0.6, ax=ax2)
    ax2.set_title('Relationship between Image Cosine Similarity and EditScore', fontsize=14)
    ax2.set_xlabel('Image Cosine Similarity', fontsize=12)
    ax2.set_ylabel('EditScore', fontsize=12)
    
    # 3. Relationship between text_similarity and editscore
    ax3 = fig.add_subplot(gs[1, 0])
    sns.scatterplot(x='text_similarity', y='editscore', data=df, alpha=0.6, ax=ax3)
    ax3.set_title('Relationship between Text Similarity and EditScore', fontsize=14)
    ax3.set_xlabel('Text Similarity', fontsize=12)
    ax3.set_ylabel('EditScore', fontsize=12)
    
    # 4. Comparison of distributions of the three components
    ax4 = fig.add_subplot(gs[1, 1])
    components = pd.DataFrame({
        'Image RLS': df['image_rls'].dropna(),
        'Image Cosine Sim': df['image_cosine_sim'].dropna(),
        'Text Similarity': df['text_similarity'].dropna()
    })
    melted_components = pd.melt(components)
    sns.boxplot(x='variable', y='value', data=melted_components, palette="Set2", ax=ax4)
    ax4.set_title('Distribution Comparison of EditScore Components', fontsize=14)
    ax4.set_ylabel('Value', fontsize=12)
    ax4.set_xlabel('')
    
    plt.tight_layout()
    plt.savefig('visualizations/editscore_components.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create 3D scatter plot to show relationship between the three components and EditScore
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw scatter points
    scatter = ax.scatter(df['image_cosine_sim'], df['text_similarity'], df['image_rls'], 
                         c=df['editscore'], cmap='viridis', alpha=0.7)
    
    # Add color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label('EditScore', fontsize=12)
    
    # Set labels
    ax.set_xlabel('Image Cosine Similarity', fontsize=12)
    ax.set_ylabel('Text Similarity', fontsize=12)
    # 正确设置z轴标签
    ax.set_zlabel('Image RLS', fontsize=12)
    # 确保z轴标签可见
    ax.zaxis.set_rotate_label(False)  # 防止标签旋转
    ax.zaxis.label.set_fontsize(12)
    
    plt.title('3D Visualization of EditScore Components', fontsize=16)
    plt.tight_layout()
    plt.savefig('visualizations/editscore_3d.png', dpi=300, bbox_inches='tight')
    plt.close()

# 6. High and Low Score Case Analysis
def high_low_score_analysis():
    # Sort by editscore and gpt_total_score
    df_sorted_edit = df.sort_values(by='editscore', ascending=False)
    df_sorted_gpt = df.sort_values(by='gpt_total_score', ascending=False)
    
    # Get top 10 and bottom 10 cases
    top10_edit = df_sorted_edit.head(10)
    bottom10_edit = df_sorted_edit.tail(10)
    top10_gpt = df_sorted_gpt.head(10)
    bottom10_gpt = df_sorted_gpt.tail(10)
    
    # Create tables
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # Top EditScore cases
    axes[0, 0].axis('tight')
    axes[0, 0].axis('off')
    table_data = top10_edit[['sequence_number', 'image_name', 'editscore', 'gpt_total_score']]
    table = axes[0, 0].table(cellText=table_data.values, colLabels=table_data.columns, 
                             loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    axes[0, 0].set_title('Top 10 Cases by EditScore', fontsize=14)
    
    # Bottom EditScore cases
    axes[0, 1].axis('tight')
    axes[0, 1].axis('off')
    table_data = bottom10_edit[['sequence_number', 'image_name', 'editscore', 'gpt_total_score']]
    table = axes[0, 1].table(cellText=table_data.values, colLabels=table_data.columns, 
                             loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    axes[0, 1].set_title('Bottom 10 Cases by EditScore', fontsize=14)
    
    # Top GPT cases
    axes[1, 0].axis('tight')
    axes[1, 0].axis('off')
    table_data = top10_gpt[['sequence_number', 'image_name', 'gpt_total_score', 'editscore']]
    table = axes[1, 0].table(cellText=table_data.values, colLabels=table_data.columns, 
                             loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    axes[1, 0].set_title('Top 10 Cases by GPT Score', fontsize=14)
    
    # Bottom GPT cases
    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
    table_data = bottom10_gpt[['sequence_number', 'image_name', 'gpt_total_score', 'editscore']]
    table = axes[1, 1].table(cellText=table_data.values, colLabels=table_data.columns, 
                             loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    axes[1, 1].set_title('Bottom 10 Cases by GPT Score', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('visualizations/high_low_cases.png', dpi=300, bbox_inches='tight')
    plt.close()

# 7. Create Comprehensive Dashboard
def create_dashboard():
    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(20, 24))
    gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=[1, 1, 1, 1.5])
    
    # 1. Scatter plot of EditScore vs GPT score
    ax1 = fig.add_subplot(gs[0, 0])
    sns.scatterplot(x='editscore', y='gpt_total_score', data=df, alpha=0.6, ax=ax1)
    ax1.set_title('Relationship between EditScore and GPT Score', fontsize=14)
    ax1.set_xlabel('EditScore', fontsize=12)
    ax1.set_ylabel('GPT Total Score', fontsize=12)
    
    # 2. Comparison of normalized scores
    ax2 = fig.add_subplot(gs[0, 1])
    norm_data = pd.DataFrame({
        'EditScore': df['editscore_norm'].dropna(),
        'GPT Score': df['gpt_score_norm'].dropna()
    })
    melted_norm_data = pd.melt(norm_data)
    sns.boxplot(x='variable', y='value', data=melted_norm_data, palette=['skyblue', 'salmon'], ax=ax2)
    ax2.set_title('Comparison of Normalized Scores', fontsize=14)
    ax2.set_ylabel('Normalized Score', fontsize=12)
    ax2.set_xlabel('')
    
    # 3. Average GPT scores across dimensions
    ax3 = fig.add_subplot(gs[1, 0])
    gpt_dims = ['gpt_editing_accuracy', 'gpt_visual_quality', 
                'gpt_content_preservation', 'gpt_style_consistency', 
                'gpt_overall_effect', 'gpt_total_score']
    gpt_means_series = df[gpt_dims].mean()
    gpt_means_sorted = gpt_means_series.sort_values(ascending=False)
    
    # Set x-axis labels
    x_labels = gpt_means_sorted.index.tolist()
    x_pos = np.arange(len(x_labels))
    
    bars = ax3.bar(x_pos, gpt_means_sorted.values, color=sns.color_palette("muted", len(gpt_dims)))
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8, color='black')
    ax3.set_title('Average GPT Scores by Dimension', fontsize=14)
    ax3.set_ylabel('Average Score', fontsize=12)
    ax3.set_ylim(0, 1)
    ax3.set_xticks(x_pos)
    # Fix set_xticklabels by ensuring x_labels is a list of strings
    ax3.set_xticklabels([str(label) for label in x_labels], rotation=45, ha='right')
    
    # 4. EditScore component distribution
    ax4 = fig.add_subplot(gs[1, 1])
    components = pd.DataFrame({
        'Image RLS': df['image_rls'].dropna(),
        'Image Cosine Sim': df['image_cosine_sim'].dropna(),
        'Text Similarity': df['text_similarity'].dropna()
    })
    melted_components = pd.melt(components)
    sns.boxplot(x='variable', y='value', data=melted_components, palette="Set2", ax=ax4)
    ax4.set_title('Distribution of EditScore Components', fontsize=14)
    ax4.set_ylabel('Value', fontsize=12)
    ax4.set_xlabel('')
    
    # 5. EditScore distribution
    ax5 = fig.add_subplot(gs[2, 0])
    editscore_data = df['editscore'].dropna().values
    sns.histplot(x=editscore_data, kde=True, ax=ax5, color='skyblue')
    ax5.set_title('EditScore Distribution', fontsize=14)
    ax5.set_xlabel('EditScore', fontsize=12)
    ax5.set_ylabel('Frequency', fontsize=12)
    
    # 6. GPT total score distribution
    ax6 = fig.add_subplot(gs[2, 1])
    gpt_score_data = df['gpt_total_score'].dropna().values
    sns.histplot(x=gpt_score_data, kde=True, ax=ax6, color='salmon')
    ax6.set_title('GPT Total Score Distribution', fontsize=14)
    ax6.set_xlabel('GPT Total Score', fontsize=12)
    ax6.set_ylabel('Frequency', fontsize=12)
    
    # 7. Correlation heatmap
    ax7 = fig.add_subplot(gs[3, :])
    corr_cols = ['image_rls', 'image_cosine_sim', 'text_similarity', 
                'gpt_editing_accuracy', 'gpt_visual_quality', 
                'gpt_content_preservation', 'gpt_style_consistency', 
                'gpt_overall_effect', 'gpt_total_score', 
                'editscore']
    corr_df = df[corr_cols].dropna().corr()
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#4575b4', 'white', '#d73027'])
    sns.heatmap(corr_df, annot=True, cmap=cmap, vmin=-1, vmax=1, 
                linewidths=0.5, annot_kws={"size": 8}, fmt=".2f", ax=ax7)
    ax7.set_title('Correlation Analysis of Evaluation Metrics', fontsize=14)
    
    plt.suptitle('Comprehensive Dashboard for Image Editing Quality Evaluation', fontsize=20, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig('visualizations/dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()

# Execute all visualization functions
if __name__ == "__main__":
    print("Generating visualizations...")
    basic_stats()
    correlation_analysis()
    distribution_visualization()
    gpt_dimensions_analysis()
    editscore_components_analysis()
    high_low_score_analysis()
    create_dashboard()
    print("All visualizations have been generated. Please check the 'visualizations' directory.")