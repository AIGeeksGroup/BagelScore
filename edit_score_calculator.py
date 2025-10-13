#!/usr/bin/env python3
"""
EditScore Base Metrics Calculator for BAGEL Framework
只计算EditScore的三个基础指标，不计算最终的EditScore：
1. image_rls (Relative Latent Shift) - 图像变化幅度
2. image_cosine_sim (Cosine Similarity) - 图像保持程度
3. text_similarity - 文本一致性
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from PIL import Image


class EditScoreCalculator:
    """EditScore基础指标计算器，只计算三个基础指标：image_rls、image_cosine_sim、text_similarity"""
    
    def __init__(self, device='cuda'):
        self.device = device
        
    def compute_rls(self, original_latent: torch.Tensor, generated_latent: torch.Tensor) -> float:
        """
        计算 Relative Latent Shift (RLS)
        RLS = ||generated - original||_2 / ||original||_2
        
        Args:
            original_latent: 原始图像的 VAE latent (1, C, H, W)
            generated_latent: 生成图像的 latent (1, C, H, W)
            
        Returns:
            RLS 值，越大表示变化越大
        """
        # 确保维度一致并展平
        orig_flat = original_latent.view(-1).float()
        gen_flat = generated_latent.view(-1).float()
        
        # 计算 L2 norm
        diff_norm = torch.norm(gen_flat - orig_flat, p=2)
        orig_norm = torch.norm(orig_flat, p=2)
        
        # 避免除零
        if orig_norm > 1e-8:
            rls = diff_norm / orig_norm
        else:
            rls = diff_norm
            
        return float(rls.cpu())
    
    def compute_cosine_similarity(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
        """
        计算两个张量的余弦相似度
        
        Args:
            tensor1: 第一个张量
            tensor2: 第二个张量
            
        Returns:
            余弦相似度值 [-1, 1]
        """
        # 展平并归一化
        vec1 = tensor1.view(-1).float()
        vec2 = tensor2.view(-1).float()
        
        # 计算余弦相似度
        similarity = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0), dim=1)
        return float(similarity.cpu())
    
    def compute_text_embedding_similarity(self, text1_emb: torch.Tensor, text2_emb: torch.Tensor) -> float:
        """
        计算文本嵌入的余弦相似度
        
        Args:
            text1_emb: 输入文本的嵌入向量
            text2_emb: Think 文本的嵌入向量
            
        Returns:
            文本相似度
        """
        return self.compute_cosine_similarity(text1_emb, text2_emb)
    
    def compute_image_latent_metrics(self, original_latent: torch.Tensor, generated_latent: torch.Tensor) -> Dict[str, float]:
        """
        计算图像 latent 的所有指标
        
        Args:
            original_latent: 原始图像 VAE latent
            generated_latent: 生成图像 latent
            
        Returns:
            包含 RLS, cosine_sim, 和组合指标的字典
        """
        rls = self.compute_rls(original_latent, generated_latent)
        cosine_sim = self.compute_cosine_similarity(original_latent, generated_latent)
        
        return {
            'rls': rls,
            'cosine_sim': cosine_sim,
            'change_magnitude': rls,  # RLS 表示变化幅度
            'preservation': cosine_sim  # 余弦相似度表示保持程度
        }
    
    # compute_editscore method removed - only compute base metrics
    
    def compute_base_metrics(self, 
                            original_vae_latent: torch.Tensor,
                            generated_latent: torch.Tensor,
                            input_text_emb: torch.Tensor,
                            think_text_emb: torch.Tensor) -> Dict[str, Any]:
        """
        计算EditScore的三个基础指标（不计算最终的EditScore）
        
        Args:
            original_vae_latent: 原始图像VAE latent
            generated_latent: 生成图像latent
            input_text_emb: 输入文本嵌入
            think_text_emb: Think文本嵌入
            
        Returns:
            包含三个基础指标的字典：image_rls, image_cosine_sim, text_similarity
        """
        # Compute image metrics
        image_metrics = self.compute_image_latent_metrics(original_vae_latent, generated_latent)
        
        # Compute text similarity
        text_similarity = self.compute_text_embedding_similarity(input_text_emb, think_text_emb)
        
        # Return only the three base metrics (no final EditScore calculation)
        return {
            'image_rls': image_metrics['rls'],
            'image_cosine_sim': image_metrics['cosine_sim'],
            'text_similarity': text_similarity
        }


def plot_roc_curve(scores: np.ndarray, labels: np.ndarray, score_name: str = "EditScore") -> Dict[str, float]:
    """
    绘制 ROC 曲线
    
    Args:
        scores: 预测分数数组
        labels: 真实标签数组 (1=好编辑, 0=差编辑)
        score_name: 分数名称
        
    Returns:
        包含 FPR, TPR, AUC 的字典
    """
    # 验证输入
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    
    if len(scores) != len(labels):
        raise ValueError("分数和标签数组长度必须相同")
    
    # 手写 ROC 计算（避免依赖 sklearn）
    order = np.argsort(-scores)  # 降序排列
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    
    # 计算正负样本数
    n_pos = np.sum(sorted_labels == 1)
    n_neg = np.sum(sorted_labels == 0)
    
    if n_pos == 0 or n_neg == 0:
        print(f"警告：样本中缺少正样本({n_pos})或负样本({n_neg})，无法计算有效的 ROC")
        return {'fpr': np.array([0, 1]), 'tpr': np.array([0, 1]), 'auc': 0.5}
    
    # 计算 TPR 和 FPR
    tpr_list = [0.0]
    fpr_list = [0.0]
    
    tp = 0
    fp = 0
    
    for i in range(len(sorted_scores)):
        if sorted_labels[i] == 1:
            tp += 1
        else:
            fp += 1
            
        tpr = tp / n_pos
        fpr = fp / n_neg
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    fpr = np.array(fpr_list)
    tpr = np.array(tpr_list)
    
    # 计算 AUC (使用梯形积分)
    auc = np.trapz(tpr, fpr)
    
    # 绘制 ROC 曲线（不在这里显示，让调用者决定）
    plt.plot(fpr, tpr, linewidth=2, label=f'{score_name} (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'ROC Curve for {score_name}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    print(f"{score_name}: AUC = {auc:.4f}, 正样本: {n_pos}, 负样本: {n_neg}")
    
    return {'fpr': fpr, 'tpr': tpr, 'auc': auc}


def analyze_score_distribution(scores: np.ndarray, labels: np.ndarray, score_name: str = "EditScore"):
    """
    分析分数分布
    
    Args:
        scores: 分数数组
        labels: 标签数组
        score_name: 分数名称
    """
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(pos_scores, bins=20, alpha=0.7, label=f'Good Edits (n={len(pos_scores)})', color='green')
    plt.hist(neg_scores, bins=20, alpha=0.7, label=f'Bad Edits (n={len(neg_scores)})', color='red')
    plt.xlabel(score_name)
    plt.ylabel('Frequency')
    plt.title(f'{score_name} Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.boxplot([pos_scores, neg_scores], labels=['Good Edits', 'Bad Edits'])
    plt.ylabel(score_name)
    plt.title(f'{score_name} Box Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n{score_name} 统计:")
    print(f"Good Edits: mean={np.mean(pos_scores):.4f}, std={np.std(pos_scores):.4f}")
    print(f"Bad Edits: mean={np.mean(neg_scores):.4f}, std={np.std(neg_scores):.4f}")
    print(f"Overall: mean={np.mean(scores):.4f}, std={np.std(scores):.4f}")


