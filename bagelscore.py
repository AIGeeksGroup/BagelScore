#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os
import torch
import numpy as np
from PIL import Image
import re
import time
from typing import Dict, List, Tuple, Any, Set, Union
import gc

# 设置PyTorch CUDA内存分配配置，避免内存碎片化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 导入BAGEL相关模块
from data.transforms import ImageTransform
from data.data_utils import pil_img2rgb, add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.autoencoder import load_ae
from inferencer import InterleaveInferencer
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights


class BagelScorer:
    """
    基于推理的语义判断评分器 - 实现BagelScore计算方法
    
    BagelScore通过直接询问模型"图像和文本是否描述相同内容"来评估语义兼容性，
    而不是在嵌入空间中测量相似度。这种方法能够捕捉到细微的语义不匹配，
    如否定和替换，并能够对长篇复杂描述进行组合推理。
    """
    
    def __init__(self, model_path: str, device_id: int = 0):
        """
        初始化BagelScorer
        
        Args:
            model_path: BAGEL模型路径
            device_id: 使用的GPU设备ID
        """
        self.device_id = device_id
        self.model_path = model_path
        self.setup_model()
        self.setup_inferencer()
        
        # 定义肯定性verbalizer集合
        self.verbalizer_yes = {"Yes", "yes", "YES"}
        
    def setup_model(self):
        """初始化BAGEL模型 - 强制使用单GPU"""
        print(f"Setting up BAGEL model on device {self.device_id}...")
        
        # 设置环境变量避免内存碎片
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # 强制使用指定的GPU
        torch.cuda.set_device(self.device_id)
        target_device = f"cuda:{self.device_id}"
        print(f"强制使用指定的设备: {target_device}")
        
        # 检查可用GPU数量并警告
        if torch.cuda.device_count() > 1:
            print(f"警告: 系统有{torch.cuda.device_count()}个GPU，但只会使用device_id={self.device_id}的GPU")
        
        # LLM config preparing
        llm_config = Qwen2Config.from_json_file(os.path.join(self.model_path, "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"
        
        # ViT config preparing
        vit_config = SiglipVisionConfig.from_json_file(os.path.join(self.model_path, "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1
        
        # VAE loading
        self.vae_model, vae_config = load_ae(local_path=os.path.join(self.model_path, "ae.safetensors"))
        # 确保VAE模型在指定GPU上
        self.vae_model = self.vae_model.to(target_device)
        
        # Bagel config preparing
        config = BagelConfig(
            visual_gen=True,
            visual_und=True,
            llm_config=llm_config, 
            vit_config=vit_config,
            vae_config=vae_config,
            vit_max_num_patch_per_side=70,
            connector_act='gelu_pytorch_tanh',
            latent_patch_size=2,
            max_latent_size=64,
        )
        
        # 使用原始方法初始化模型
        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)
            self.model = Bagel(language_model, vit_model, config)
            self.model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)
        
        # 设置最大内存只针对指定的GPU，并确保其他GPU不被使用
        available_gpus = list(range(torch.cuda.device_count()))
        max_memory = {}
        
        # 为指定GPU分配大部分内存
        max_memory[self.device_id] = "80GiB"
        
        # 为其他GPU分配极小内存，确保不会被使用
        for gpu_id in available_gpus:
            if gpu_id != self.device_id:
                max_memory[gpu_id] = "1MiB"  # 分配极小内存，实际上禁用这些GPU
        
        print(f"内存分配策略: {max_memory}")
        
        # 使用指定的GPU进行设备映射
        device_map = infer_auto_device_map(
            self.model,
            max_memory=max_memory,
            no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
        )
        print("原始设备映射:", device_map)
        
        # 强制所有模块使用指定的GPU
        cleaned_device_map = {key: target_device for key in device_map.keys()}
        device_map = cleaned_device_map
        print("修改后的设备映射 (强制使用单GPU):", device_map)
        
        # 确保关键模块在同一设备上
        same_device_modules = [
            'language_model.model.embed_tokens',
            'time_embedder',
            'latent_pos_embed',
            'vae2llm',
            'llm2vae',
            'connector',
            'vit_pos_embed'
        ]
        
        # 确保所有模块都在指定的GPU上
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = target_device
        
        # 使用原始方法加载模型权重
        self.model = load_checkpoint_and_dispatch(
            self.model,
            checkpoint=os.path.join(self.model_path, "ema.safetensors"),
            device_map=device_map,
            offload_buffers=True,
            dtype=torch.bfloat16,
            force_hooks=True,
            offload_folder="/tmp/offload"
        )
        
        self.model = self.model.eval()
        print('Model loaded')
        
        # Tokenizer Preparing
        self.tokenizer = Qwen2Tokenizer.from_pretrained(self.model_path)
        self.tokenizer, self.new_token_ids, _ = add_special_tokens(self.tokenizer)
        
        # Image Transform Preparing
        self.vae_transform = ImageTransform(1024, 512, 16)
        self.vit_transform = ImageTransform(980, 224, 14)
        
        print("Model setup completed!")
    
    def setup_inferencer(self):
        """初始化推理器"""
        self.inferencer = InterleaveInferencer(
            model=self.model,
            vae_model=self.vae_model,
            tokenizer=self.tokenizer,
            vae_transform=self.vae_transform,
            vit_transform=self.vit_transform,
            new_token_ids=self.new_token_ids
        )
    
    def calculate_bagelscore(self, image: Image.Image, caption: str) -> Tuple[float, Dict[str, Any]]:
        """
        计算BagelScore - 基于推理的语义判断
        
        使用数学定义:
        1. 计算肯定性verbalizer集合的logit: ℓ_yes = log(sum(exp(z_t))) for t in V_yes
        2. 应用sigmoid函数得到最终分数: S(x,y) = σ(ℓ_yes)
        
        Args:
            image: 输入图像
            caption: 输入文本描述
            
        Returns:
            Tuple[float, Dict]: (bagelscore, 详细信息)
        """
        try:
            # 构建二元查询prompt
            prompt = f"""Are the IMAGE and the TEXT describing the same content?

TEXT: "{caption}"

Please answer with only Yes or No:"""
            
            # 使用understanding_output模式进行推理，并返回logits
            output_dict = self.inferencer(
                image=image,
                text=prompt,
                understanding_output=True,
                max_think_token_n=500,
                do_sample=False,
                text_temperature=0.1,  # 修改为text_temperature
                return_logits=True  # 请求返回logits
            )
            
            # 获取模型输出的tokens、文本和logits
            tokens = output_dict.get('tokens', None)
            response_text = output_dict.get('text', '')
            logits = output_dict.get('logits', None)
            
            # 使用基于logits的方法计算BagelScore
            print("使用基于logits的方法计算BagelScore")
            
            # 检查是否成功获取到logits
            if logits is None or tokens is None:
                error_msg = "无法获取logits或tokens，无法计算BagelScore"
                print(error_msg)
                result_info = {
                    'method': 'error',
                    'error': error_msg,
                    'response': response_text,
                    'logits_available': logits is not None,
                    'tokens_available': tokens is not None
                }
                return 0.0, result_info
                
            # 找到答案位置
            answer_position = self._find_answer_position(tokens)
            
            if answer_position is None:
                error_msg = "无法找到答案位置，无法计算BagelScore"
                print(error_msg)
                result_info = {
                    'method': 'error',
                    'error': error_msg,
                    'response': response_text
                }
                return 0.0, result_info
                
            # 获取该位置的logits
            position_logits = logits[0, answer_position, :]
            
            # 获取Yes和No的token IDs
            yes_token_ids = self._get_yes_token_ids()
            
            # 计算Yes的logits总和
            yes_logits_sum = 0
            for yes_id in yes_token_ids:
                yes_logits_sum += position_logits[yes_id].item()
            
            # 应用sigmoid函数计算BagelScore
            bagelscore = torch.sigmoid(torch.tensor(yes_logits_sum)).item()
            
            # 构建详细信息
            result_info = {
                'method': 'logits_based',
                'response': response_text,
                'answer_position': answer_position,
                'yes_logits_sum': yes_logits_sum,
                'score': bagelscore
            }
            
            score = bagelscore
            
            # 清理内存
            del output_dict, tokens
            torch.cuda.empty_cache()
            gc.collect()
            
            return score, result_info
                
        except Exception as e:
            print(f"Error in calculating BagelScore: {e}")
            # 清理内存
            torch.cuda.empty_cache()
            gc.collect()
            return 0.0, {'error': str(e)}
    
    def _find_answer_position(self, tokens: List[int]) -> Union[int, None]:
        """
        找到答案位置的索引
        
        Args:
            tokens: 模型输出的token序列
            
        Returns:
            int or None: 答案位置的索引，如果找不到则返回None
        """
        # 获取Yes和No的token IDs
        yes_token_ids = self._get_yes_token_ids()
        
        # 从后向前搜索，找到第一个Yes或No token
        for i in range(len(tokens) - 1, -1, -1):
            if tokens[i] in yes_token_ids:
                return i
        
        return None
    
    def _get_yes_token_ids(self) -> List[int]:
        """
        获取肯定性verbalizer集合的token IDs
        
        Returns:
            List[int]: Yes, yes, YES的token IDs
        """
        yes_token_ids = []
        for yes_word in self.verbalizer_yes:
            # 获取token ID
            ids = self.tokenizer.encode(yes_word, add_special_tokens=False)
            if ids:
                yes_token_ids.append(ids[0])  # 只取第一个token
        
        return yes_token_ids
    
    def _sigmoid(self, x: float) -> float:
        """
        计算sigmoid函数: σ(x) = 1 / (1 + exp(-x))
        
        Args:
            x: 输入值
            
        Returns:
            float: sigmoid(x)
        """
        return 1.0 / (1.0 + np.exp(-x))


def evaluate_dataset_with_bagelscore(
    model_path: str,
    data_file: str,
    images_dir: str,
    output_file: str,
    dataset_type: str = "expert",
    device_id: int = 0,
    limit: int = None,
    batch_size: int = 10,
    start_from: int = 0,
    reinit_model_freq: int = 3  # 每处理多少批次重新初始化模型一次
):
    """
    使用BagelScore评估数据集
    
    Args:
        model_path: BAGEL模型路径
        data_file: 数据集文件路径
        images_dir: 图像目录路径
        output_file: 输出文件路径
        dataset_type: 数据集类型，"expert"或"cf"
        device_id: 使用的GPU设备ID
        limit: 限制评估的样本数量
        batch_size: 批处理大小
        start_from: 从哪个样本开始评估
        reinit_model_freq: 每处理多少批次重新初始化模型一次
    """
    # 导入Flickr8kDataset
    from flickr8k_memory_fixed import Flickr8kDataset
    
    # 设置设备
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        device = f"cuda:{device_id}"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    print(f"强制使用GPU: {device}")
    
    # 创建数据集
    dataset = Flickr8kDataset(data_file, images_dir, dataset_type, limit)
    
    # 计算总批次数
    total_samples = len(dataset)
    if limit:
        total_samples = min(total_samples, limit)
    
    # 确定起始样本和结束样本
    start_sample = start_from
    end_sample = total_samples
    
    # 计算批次
    num_batches = (end_sample - start_sample + batch_size - 1) // batch_size
    
    print(f"Total samples: {total_samples}")
    print(f"Starting from sample: {start_sample}")
    print(f"Total batches: {num_batches}")
    
    # 创建结果目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 检查是否有已存在的结果文件
    import csv
    all_results = []
    if os.path.exists(output_file) and start_from > 0:
        print(f"Loading existing results from {output_file}...")
        try:
            with open(output_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                all_results = list(reader)
                print(f"Loaded {len(all_results)} existing results")
        except Exception as e:
            print(f"Error loading existing results: {e}")
    
    # 初始化评估器
    evaluator = None
    
    # 处理每个批次
    for batch_idx in range(num_batches):
        # 计算当前批次的样本范围
        batch_start = start_sample + batch_idx * batch_size
        batch_end = min(batch_start + batch_size, end_sample)
        
        print(f"\nProcessing batch {batch_idx + 1}/{num_batches}: samples {batch_start}-{batch_end - 1}")
        
        # 定期重新初始化模型以防止内存泄漏
        if batch_idx % reinit_model_freq == 0 or evaluator is None:
            if evaluator is not None:
                # 清理旧模型
                print("开始清理旧模型...")
                # 第一阶段：删除模型组件
                if hasattr(evaluator, 'model') and evaluator.model is not None:
                    del evaluator.model
                if hasattr(evaluator, 'vae_model') and evaluator.vae_model is not None:
                    del evaluator.vae_model
                if hasattr(evaluator, 'inferencer') and evaluator.inferencer is not None:
                    del evaluator.inferencer
                # 第二阶段：删除评估器
                del evaluator
                
                # 第三阶段：强制清理内存
                torch.cuda.empty_cache()
                gc.collect()
                print("Reinitializing model to prevent memory leaks...")
                # 强制等待GPU清理完成
                import time
                print("等待GPU内存释放...")
                time.sleep(5)
                
                # 再次清理
                torch.cuda.empty_cache()
                gc.collect()
                
                # 检查内存状态
                print(f"当前GPU内存使用情况: {torch.cuda.memory_allocated(device_id) / 1024**3:.2f} GB")
            
            # 初始化新的评估器 - 添加重试机制
            try:
                print("开始初始化新的评估器...")
                evaluator = BagelScorer(model_path, device_id)
                print("评估器初始化成功")
            except Exception as e:
                print(f"评估器初始化失败: {str(e)}")
                # 再次尝试清理内存并重试
                torch.cuda.empty_cache()
                gc.collect()
                time.sleep(15)
                print("第二次尝试初始化评估器...")
                evaluator = BagelScorer(model_path, device_id)
        
        # 批次结果
        batch_results = []
        
        # 处理当前批次的样本
        for i in range(batch_start, batch_end):
            try:
                item = dataset[i]
                image = item['image']
                caption = item['caption']
                image_id = item['image_id']
                human_rating = item['rating']
                image_path = item['image_path']
                
                # 计算BagelScore
                bagelscore, result_info = evaluator.calculate_bagelscore(image, caption)
                
                # 添加到批次结果
                batch_results.append({
                    'image_id': image_id,
                    'image_path': image_path,
                    'caption': caption,
                    'human_rating': float(human_rating),
                    'bagelscore': float(bagelscore),
                    'bagel_response': result_info.get('response', ''),
                    'calculation_method': result_info.get('method', 'unknown')
                })
                
                # 打印进度
                print(f"  Sample {i}: Human={human_rating}, BagelScore={bagelscore:.3f}")
                
                # 每个样本后清理内存
                torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                # 记录错误样本
                batch_results.append({
                    'image_id': item['image_id'] if 'item' in locals() and 'image_id' in item else f"error_{i}",
                    'image_path': item['image_path'] if 'item' in locals() and 'image_path' in item else "unknown",
                    'caption': item['caption'] if 'item' in locals() and 'caption' in item else "unknown",
                    'human_rating': float(item['rating']) if 'item' in locals() and 'rating' in item else 0.0,
                    'bagelscore': 0.0,
                    'bagel_response': f"ERROR: {str(e)}",
                    'calculation_method': 'error'
                })
                # 清理内存
                torch.cuda.empty_cache()
                gc.collect()
                continue
        
        # 将批次结果添加到总结果
        all_results.extend(batch_results)
        
        # 保存当前批次结果（增量保存）
        print(f"Saving batch results to {output_file}...")
        try:
            # 确定CSV列
            fieldnames = ['image_id', 'image_path', 'caption', 'human_rating', 
                         'bagelscore', 'bagel_response', 'calculation_method']
            
            # 写入CSV
            with open(output_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_results)
            
            print(f"Saved {len(all_results)} results to {output_file}")
            
            # 创建备份文件
            backup_file = f"{output_file}.bak.{batch_idx}"
            with open(backup_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_results)
            
            print(f"Created backup at {backup_file}")
            
        except Exception as e:
            print(f"Error saving results: {e}")
    
    print(f"\nEvaluation completed. Total samples processed: {len(all_results)}")
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate dataset using BagelScore")
    parser.add_argument("--model_path", type=str, required=True, help="Path to BAGEL model")
    parser.add_argument("--data_file", type=str, required=True, help="Path to dataset file")
    parser.add_argument("--images_dir", type=str, required=True, help="Path to images directory")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output file")
    parser.add_argument("--dataset_type", type=str, default="expert", choices=["expert", "cf"], 
                        help="Dataset type: 'expert' or 'cf'")
    parser.add_argument("--device_id", type=int, default=0, help="GPU device ID to use")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for processing")
    parser.add_argument("--start_from", type=int, default=0, help="Start from this sample index")
    parser.add_argument("--reinit_model_freq", type=int, default=3, 
                        help="Reinitialize model every N batches")
    
    args = parser.parse_args()
    
    evaluate_dataset_with_bagelscore(
        model_path=args.model_path,
        data_file=args.data_file,
        images_dir=args.images_dir,
        output_file=args.output_file,
        dataset_type=args.dataset_type,
        device_id=args.device_id,
        limit=args.limit,
        batch_size=args.batch_size,
        start_from=args.start_from,
        reinit_model_freq=args.reinit_model_freq
    )