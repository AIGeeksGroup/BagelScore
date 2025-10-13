#!/usr/bin/env python3
"""
BAGEL EditScore Base Metrics Evaluation Script
只计算EditScore的三个基础指标：image_rls、image_cosine_sim、text_similarity

Usage:
python run_editscore_evaluation.py --mode single --prompt "Apply a cartoon style to the whole image."
python run_editscore_evaluation.py --mode batch --limit 10 --prompt "Apply a cartoon style to the whole image."
python run_editscore_evaluation.py --mode analyze --results_csv results/editscore_results.csv
"""

import os
import sys
import argparse
import glob
import csv
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import time
import hashlib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import threading
import multiprocessing as mp

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入 BAGEL 相关模块
try:
    import torch
    from inferencer import InterleaveInferencer
    from edit_score_calculator import EditScoreCalculator
    from data.transforms import ImageTransform
    from data.data_utils import pil_img2rgb, add_special_tokens
    from modeling.bagel import BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
    from modeling.qwen2 import Qwen2Tokenizer
    from modeling.autoencoder import load_ae
    from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you run this script from the BAGEL project root directory")
    sys.exit(1)


def process_images_on_gpu(gpu_id: int, image_paths: List[str], model_path: str, prompt: str, results_dir: str) -> List[Dict[str, Any]]:
    """
    在指定GPU上独立处理图片列表
    """
    import os
    import sys
    import time
    import hashlib
    import re
    import json
    import numpy as np
    from pathlib import Path
    from PIL import Image
    
    # 添加项目根目录到 Python 路径
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    import torch
    from inferencer import InterleaveInferencer
    from edit_score_calculator import EditScoreCalculator
    from data.transforms import ImageTransform
    from data.data_utils import pil_img2rgb, add_special_tokens
    from modeling.bagel import BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
    from modeling.qwen2 import Qwen2Tokenizer
    from modeling.autoencoder import load_ae
    from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
    
    print(f"GPU {gpu_id}: Initializing model...")
    
    try:
        # 设置当前进程使用的GPU
        torch.cuda.set_device(gpu_id)
        
        # 确保使用绝对路径
        abs_model_path = os.path.abspath(model_path)
        print(f"GPU {gpu_id}: Using model path: {abs_model_path}")
        
        # 初始化模型（直接使用JSON配置文件）
        try:
            llm_config = Qwen2Config.from_json_file(os.path.join(abs_model_path, "llm_config.json"))
        except Exception as e:
            print(f"GPU {gpu_id}: Error loading LLM config from JSON: {e}")
            raise e
        
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"
        
        try:
            vit_config = SiglipVisionConfig.from_json_file(os.path.join(abs_model_path, "vit_config.json"))
        except Exception as e:
            print(f"GPU {gpu_id}: Error loading ViT config from JSON: {e}")
            raise e
        
        vit_config.rope = False
        vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1
        
        vae_model, vae_config = load_ae(local_path=os.path.join(abs_model_path, "ae.safetensors"))
        
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
        
        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)
            model = Bagel(language_model, vit_model, config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)
        
        tokenizer = Qwen2Tokenizer.from_pretrained(abs_model_path)
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
        
        vae_transform = ImageTransform(1024, 512, 16)
        vit_transform = ImageTransform(980, 224, 14)
        
        # 将模型加载到指定GPU
        device_map = {name: f"cuda:{gpu_id}" for name, _ in model.named_modules()}
        
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=os.path.join(abs_model_path, "ema.safetensors"),
            device_map=device_map,
            offload_buffers=True,
            dtype=torch.bfloat16,
            force_hooks=True,
            offload_folder="/tmp/offload"
        )
        
        model = model.eval()
        
        inferencer = InterleaveInferencer(
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            vae_transform=vae_transform,
            vit_transform=vit_transform,
            new_token_ids=new_token_ids
        )
        
        edit_score_calc = EditScoreCalculator()
        
        inference_hyper = {
            'max_think_token_n': 1000,
            'do_sample': False,
            'cfg_text_scale': 4.0,
            'cfg_img_scale': 2.0,
            'cfg_interval': [0.0, 1.0],
            'timestep_shift': 3.0,
            'num_timesteps': 50,
            'cfg_renorm_min': 0.0,
            'cfg_renorm_type': "text_channel",
        }
        
        print(f"GPU {gpu_id}: Model loaded successfully, processing {len(image_paths)} images...")
        
        results = []
        for i, image_path in enumerate(image_paths):
            try:
                image = Image.open(image_path).convert("RGB")
                out = inferencer(
                    image=image,
                    text=prompt,
                    think=True,
                    return_edit_score_data=True,
                    **inference_hyper
                )
                
                edit_data = out.get("edit_score_data")
                if edit_data:
                    # 提取图片索引
                    basename = os.path.basename(image_path)
                    match = re.match(r"(\d+)_?", basename)
                    image_index = match.group(1) if match else basename.split('.')[0]
                    
                    # 计算基础指标
                    scores = edit_score_calc.compute_base_metrics(
                        original_vae_latent=edit_data['original_vae_latent'],
                        generated_latent=edit_data['generated_latent'],
                        input_text_emb=edit_data['input_text_emb'],
                        think_text_emb=edit_data['think_text_emb']
                    )
                    
                    # 生成图片ID
                    content = f"{image_path}_{prompt}"
                    image_id = hashlib.md5(content.encode()).hexdigest()[:8]
                    
                    result = {
                        'image_id': image_id,
                        'image_path': image_path,
                        'prompt': prompt,
                        'think_text': out.get("text", ""),
                        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                        'image_rls': scores['image_rls'],
                        'image_cosine_sim': scores['image_cosine_sim'],
                        'text_similarity': scores['text_similarity'],
                        'gpu_id': gpu_id
                    }
                    
                    results.append(result)
                    
                    # 保存中间数据
                    intermediate_data_dir = Path(results_dir) / "intermediate_data"
                    intermediate_data_dir.mkdir(exist_ok=True)
                    intermediate_file = intermediate_data_dir / f"{image_index}_analysis.json"
                    
                    serializable_data = {}
                    for key, value in edit_data.items():
                        if hasattr(value, 'cpu') and hasattr(value, 'numpy'):
                            serializable_data[key] = value.cpu().numpy().tolist()
                        elif isinstance(value, np.ndarray):
                            serializable_data[key] = value.tolist()
                        else:
                            serializable_data[key] = value
                    
                    with open(intermediate_file, 'w', encoding='utf-8') as f:
                        json.dump(serializable_data, f, ensure_ascii=False, indent=2)
                    
                    # 保存生成的图片
                    generated_image = out.get("image")
                    if generated_image:
                        output_image_path = Path(results_dir) / f"{image_index}_edited.jpg"
                        generated_image.save(output_image_path)
                    
                    # 保存详细结果
                    result_file = Path(results_dir) / f"{image_index}_edited.json"
                    with open(result_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                
                if (i + 1) % 10 == 0:
                    print(f"GPU {gpu_id}: Processed {i + 1}/{len(image_paths)} images")
                    
            except Exception as e:
                print(f"GPU {gpu_id}: Error processing {image_path}: {e}")
                results.append({
                    'image_id': hashlib.md5(f"{image_path}_{prompt}".encode()).hexdigest()[:8],
                    'image_path': image_path,
                    'error': str(e),
                    'gpu_id': gpu_id
                })
        
        print(f"GPU {gpu_id}: Completed processing {len(results)} images")
        return results
        
    except Exception as e:
        print(f"GPU {gpu_id}: Failed to initialize: {e}")
        return [{'error': f'GPU {gpu_id} initialization failed: {e}'}]


def plot_roc_curve(scores: np.ndarray, labels: np.ndarray, score_name: str = "EditScore") -> Dict[str, float]:
    """
    Plot ROC curve
    
    Args:
        scores: Prediction scores array
        labels: True labels array (1=good edit, 0=bad edit)
        score_name: Score name
        
    Returns:
        Dictionary containing FPR, TPR, AUC
    """
    # Validate input
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    
    if len(scores) != len(labels):
        raise ValueError("Scores and labels arrays must have the same length")
    
    # Manual ROC calculation (avoid sklearn dependency)
    order = np.argsort(-scores)  # Descending order
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    
    # Calculate positive and negative samples
    n_pos = np.sum(sorted_labels == 1)
    n_neg = np.sum(sorted_labels == 0)
    
    if n_pos == 0 or n_neg == 0:
        print(f"Warning: Missing positive({n_pos}) or negative({n_neg}) samples, cannot calculate valid ROC")
        return {'fpr': np.array([0, 1]), 'tpr': np.array([0, 1]), 'auc': 0.5}
    
    # Calculate TPR and FPR
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
    
    # Calculate AUC (using trapezoidal integration)
    auc = np.trapz(tpr, fpr)
    
    # Plot ROC curve (let caller decide to show)
    plt.plot(fpr, tpr, linewidth=2, label=f'{score_name} (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'ROC Curve for {score_name}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    print(f"{score_name}: AUC = {auc:.4f}, Positive samples: {n_pos}, Negative samples: {n_neg}")
    
    return {'fpr': fpr, 'tpr': tpr, 'auc': auc}


class EditScoreEvaluator:
    """EditScore基础指标评估器，只计算三个基础指标"""
    
    def __init__(self, model_path: str = "./models/BAGEL-7B-MoT", images_dir: str = "./images", results_dir: str = "./results", num_gpus: int = 1, device: str = "auto"):
        self.model_path = model_path
        self.images_dir = Path(images_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.num_gpus = num_gpus
        self.device = device
        
        # Results table path (in results directory)
        self.results_table_path = self.results_dir / "editscore_results.csv"
        self.intermediate_data_dir = self.results_dir / "intermediate_data"
        self.intermediate_data_dir.mkdir(exist_ok=True)
        
        # Inference parameters
        self.inference_hyper = {
            'max_think_token_n': 1000,
            'do_sample': False,
            'cfg_text_scale': 4.0,
            'cfg_img_scale': 2.0,
            'cfg_interval': [0.0, 1.0],
            'timestep_shift': 3.0,
            'num_timesteps': 50,
            'cfg_renorm_min': 0.0,
            'cfg_renorm_type': "text_channel",
        }
        
        # Initialize components
        self.inferencer = None
        self.edit_score_calc = None
        self._load_model()
        
    def _generate_image_id(self, image_path: str, prompt: str) -> str:
        """Generate unique image ID based on path and prompt"""
        content = f"{image_path}_{prompt}"
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    def _extract_image_index(self, image_path: str) -> str:
        """从文件名中提取原图序号（假定为数字部分）"""
        basename = os.path.basename(image_path)
        match = re.match(r"(\d+)_?", basename)
        if match:
            return match.group(1)
        return basename.split('.')[0]
    
    def _get_processed_images(self, prompt: str) -> set:
        """获取已经处理过的图片ID集合"""
        processed = set()
        if self.results_table_path.exists():
            try:
                df = pd.read_csv(self.results_table_path)
                if not df.empty and 'image_id' in df.columns:
                    # 只考虑相同prompt的结果
                    if 'prompt' in df.columns:
                        processed = set(df[df['prompt'] == prompt]['image_id'].tolist())
                    else:
                        processed = set(df['image_id'].tolist())
            except Exception as e:
                print(f"Warning: Could not read existing results: {e}")
        return processed

    def _save_intermediate_data(self, image_index: str, data: Dict[str, Any]) -> str:
        """Save intermediate calculation data to JSON, 优化命名为原图序号+analysis.json"""
        intermediate_file = self.intermediate_data_dir / f"{image_index}_analysis.json"
        serializable_data = {}
        for key, value in data.items():
            if hasattr(value, 'cpu') and hasattr(value, 'numpy'):
                serializable_data[key] = value.cpu().numpy().tolist()
            elif isinstance(value, np.ndarray):
                serializable_data[key] = value.tolist()
            else:
                serializable_data[key] = value
        with open(intermediate_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, ensure_ascii=False, indent=2)
        return str(intermediate_file)
    
    def _update_results_table(self, result_data: Dict[str, Any]):
        """Update results table with new data"""
        # Load existing data or create new DataFrame
        if self.results_table_path.exists():
            df = pd.read_csv(self.results_table_path)
        else:
            df = pd.DataFrame()
        
        # Check if this image_id already exists, if so, update it
        image_id = result_data['image_id']
        if not df.empty and 'image_id' in df.columns:
            existing_index = df[df['image_id'] == image_id].index
            if len(existing_index) > 0:
                # Update existing row
                for key, value in result_data.items():
                    df.loc[existing_index[0], key] = value
            else:
                # Append new row
                df = pd.concat([df, pd.DataFrame([result_data])], ignore_index=True)
        else:
            # First entry
            df = pd.DataFrame([result_data])
        
        # Save updated table
        df.to_csv(self.results_table_path, index=False)
        print(f"Results table updated: {self.results_table_path}")
    
    def _load_model(self):
        """Load BAGEL model"""
        print("Loading BAGEL model...")
        
        try:
            # LLM config - directly load from JSON file
            llm_config = Qwen2Config.from_json_file(os.path.join(self.model_path, "llm_config.json"))
            llm_config.qk_norm = True
            llm_config.tie_word_embeddings = False
            llm_config.layer_module = "Qwen2MoTDecoderLayer"
            
            # ViT config - directly load from JSON file
            vit_config = SiglipVisionConfig.from_json_file(os.path.join(self.model_path, "vit_config.json"))
            vit_config.rope = False
            vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1
            
            # VAE loading
            vae_model, vae_config = load_ae(local_path=os.path.join(self.model_path, "ae.safetensors"))
            
            # Bagel config
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
            
            # Initialize model
            with init_empty_weights():
                language_model = Qwen2ForCausalLM(llm_config)
                vit_model = SiglipVisionModel(vit_config)
                model = Bagel(language_model, vit_model, config)
                model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)
            
            # Tokenizer
            tokenizer = Qwen2Tokenizer.from_pretrained(self.model_path)
            tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
            
            # Image Transform
            vae_transform = ImageTransform(1024, 512, 16)
            vit_transform = ImageTransform(980, 224, 14)
            
            # Device mapping and model loading
            max_mem_per_gpu = "80GiB"
            
            # Handle device specification
            if self.device == "auto":
                # Auto mode: use specified number of GPUs
                available_gpus = min(torch.cuda.device_count(), self.num_gpus)
                valid_devices = [f"cuda:{i}" for i in range(available_gpus)]
                if available_gpus == 0:
                    valid_devices = ["cpu"]
            elif self.device == "cpu":
                # Force CPU mode
                valid_devices = ["cpu"]
                available_gpus = 0
            elif self.device.startswith("cuda:"):
                # Specific GPU mode
                gpu_id = int(self.device.split(":")[1])
                if gpu_id < torch.cuda.device_count():
                    valid_devices = [self.device]
                    available_gpus = 1
                else:
                    print(f"Warning: GPU {gpu_id} not available, falling back to auto mode")
                    available_gpus = min(torch.cuda.device_count(), self.num_gpus)
                    valid_devices = [f"cuda:{i}" for i in range(available_gpus)]
            else:
                # Invalid device, fall back to auto
                print(f"Warning: Invalid device '{self.device}', falling back to auto mode")
                available_gpus = min(torch.cuda.device_count(), self.num_gpus)
                valid_devices = [f"cuda:{i}" for i in range(available_gpus)]
            
            print(f"Using device configuration: {valid_devices}")
            
            if available_gpus > 0:
                device_map = infer_auto_device_map(
                    model,
                    max_memory={i: max_mem_per_gpu for i in range(available_gpus)},
                    no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
                )
            else:
                # CPU mode
                device_map = {name: "cpu" for name, _ in model.named_modules()}
                
            if available_gpus > 0:
                cleaned_device_map = {}
                for key, device in device_map.items():
                    if device in valid_devices:
                        cleaned_device_map[key] = device
                    elif device == "disk":
                        cleaned_device_map[key] = valid_devices[0]
                    else:
                        cleaned_device_map[key] = valid_devices[0]
                device_map = cleaned_device_map
            
            # Ensure key modules are on the same device
            same_device_modules = [
                'language_model.model.embed_tokens',
                'time_embedder',
                'latent_pos_embed',
                'vae2llm',
                'llm2vae',
                'connector',
                'vit_pos_embed'
            ]
            
            if available_gpus > 0:
                if available_gpus == 1:
                    first_device = valid_devices[0]
                    for k in same_device_modules:
                        device_map[k] = first_device
                else:
                    first_device = device_map.get(same_device_modules[0], valid_devices[0])
                    for k in same_device_modules:
                        if k in device_map:
                            device_map[k] = first_device
            
            # Load weights
            model = load_checkpoint_and_dispatch(
                model,
                checkpoint=os.path.join(self.model_path, "ema.safetensors"),
                device_map=device_map,
                offload_buffers=True,
                dtype=torch.bfloat16,
                force_hooks=True,
                offload_folder="/tmp/offload"
            )
            
            model = model.eval()
            
            # Create inferencer
            self.inferencer = InterleaveInferencer(
                model=model,
                vae_model=vae_model,
                tokenizer=tokenizer,
                vae_transform=vae_transform,
                vit_transform=vit_transform,
                new_token_ids=new_token_ids
            )
            
            # Create EditScore calculator
            self.edit_score_calc = EditScoreCalculator()
            
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            raise
    
    def run_single_test(self, prompt: str = "Apply a cartoon style to the whole image.", image_path: Optional[str] = None):
        """Run single image test"""
        print(f"=== Single Image EditScore Test ===")
        
        # Start timing
        start_time = time.time()
        
        # Select test image
        if image_path is None:
            candidates = sorted(glob.glob(str(self.images_dir / "*.jpg")))
            if not candidates:
                print(f"❌ No image files found in {self.images_dir}")
                return None
            image_path = candidates[0]
        
        # Generate image index & image ID
        image_index = self._extract_image_index(image_path)
        image_id = self._generate_image_id(image_path, prompt)
        
        print(f"Test image: {image_path}")
        print(f"Image ID: {image_id}")
        print(f"Prompt: {prompt}")
        
        try:
            # Run inference
            image = Image.open(image_path).convert("RGB")
            out = self.inferencer(
                image=image,
                text=prompt,
                think=True,
                return_edit_score_data=True,
                **self.inference_hyper
            )
            
            print(f"\\n=== Think Text ===")
            print(out.get("text", "Not generated"))
            
            # Calculate EditScore
            edit_data = out.get("edit_score_data")
            if edit_data:
                print(f"\\n=== Intermediate Data ===")
                print(f"Original VAE latent shape: {edit_data['original_vae_latent'].shape}")
                print(f"Generated latent shape: {edit_data['generated_latent'].shape}")
                print(f"Input text embedding shape: {edit_data['input_text_emb'].shape}")
                print(f"Think text embedding shape: {edit_data['think_text_emb'].shape}")
                
                # Save intermediate data (analysis)
                intermediate_file = self._save_intermediate_data(image_index, edit_data)
                print(f"Intermediate data saved to: {intermediate_file}")
                
                # Calculate base metrics only
                scores = self.edit_score_calc.compute_base_metrics(
                    original_vae_latent=edit_data['original_vae_latent'],
                    generated_latent=edit_data['generated_latent'],
                    input_text_emb=edit_data['input_text_emb'],
                    think_text_emb=edit_data['think_text_emb']
                )
                
                print(f"\\n=== Base Metrics Results ===")
                for key, value in scores.items():
                    if isinstance(value, float):
                        print(f"{key:30}: {value:.4f}")
                    else:
                        print(f"{key:30}: {value}")
                
                # Prepare result data for table (only three base metrics)
                result_data = {
                    'image_id': image_id,
                    'image_path': image_path,
                    'prompt': prompt,
                    'think_text': out.get("text", ""),
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'image_rls': scores['image_rls'],
                    'image_cosine_sim': scores['image_cosine_sim'],
                    'text_similarity': scores['text_similarity'],
                    'intermediate_file': intermediate_file
                }
                
                # Update results table
                self._update_results_table(result_data)
                
                # Save detailed result (edited)
                result_file = self.results_dir / f"{image_index}_edited.json"
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(result_data, f, ensure_ascii=False, indent=2)
                print(f"\nDetailed results saved to: {result_file}")
                
                # Save generated image
                generated_image = out.get("image")
                if generated_image:
                    output_image_path = self.results_dir / f"{image_index}_edited.jpg"
                    generated_image.save(output_image_path)
                    print(f"Generated image saved to: {output_image_path}")
                
                # Calculate and display timing statistics
                end_time = time.time()
                total_time = end_time - start_time
                print(f"\\n=== Timing Statistics ===")
                print(f"Total processing time: {total_time:.2f} seconds")
                
                return result_data
            else:
                print("❌ Failed to get edit_score_data")
                return None
                
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return None
    
    def run_batch_evaluation(self, prompt: str = "Apply a cartoon style to the whole image.", limit: int = 1000):
        """Run batch evaluation with multi-GPU support and resume capability"""
        print(f"=== Batch EditScore Evaluation (Multi-GPU: {self.num_gpus}) ===")
        
        # Start timing
        start_time = time.time()
        
        # Get image list
        all_candidates = sorted(glob.glob(str(self.images_dir / "*.jpg")))[:limit]
        if not all_candidates:
            print(f"❌ No image files found in {self.images_dir}")
            return None
        
        # Check for already processed images
        processed_images = self._get_processed_images(prompt)
        print(f"Found {len(processed_images)} already processed images")
        
        # Filter out already processed images
        candidates = []
        for img_path in all_candidates:
            image_id = self._generate_image_id(img_path, prompt)
            if image_id not in processed_images:
                candidates.append(img_path)
        
        if not candidates:
            print("✅ All images have been processed!")
            return str(self.results_table_path)
            
        print(f"Found {len(all_candidates)} total images, {len(candidates)} remaining to process...")
        
        results = []
        lock = threading.Lock()
        
        def process_image(image_path):
            image_index = self._extract_image_index(image_path)
            image_id = self._generate_image_id(image_path, prompt)
            try:
                image = Image.open(image_path).convert("RGB")
                out = self.inferencer(
                    image=image,
                    text=prompt,
                    think=True,
                    return_edit_score_data=True,
                    **self.inference_hyper
                )
                edit_data = out.get("edit_score_data")
                if edit_data:
                    intermediate_file = self._save_intermediate_data(image_index, edit_data)
                    scores = self.edit_score_calc.compute_base_metrics(
                        original_vae_latent=edit_data['original_vae_latent'],
                        generated_latent=edit_data['generated_latent'],
                        input_text_emb=edit_data['input_text_emb'],
                        think_text_emb=edit_data['think_text_emb']
                    )
                    result = {
                        'image_id': image_id,
                        'image_path': image_path,
                        'prompt': prompt,
                        'think_text': out.get("text", ""),
                        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                        'image_rls': scores['image_rls'],
                        'image_cosine_sim': scores['image_cosine_sim'],
                        'text_similarity': scores['text_similarity'],
                        'intermediate_file': intermediate_file
                    }
                    # Save edited result json
                    edited_file = self.results_dir / f"{image_index}_edited.json"
                    with open(edited_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    # Save generated image
                    generated_image = out.get("image")
                    if generated_image:
                        output_image_path = self.results_dir / f"{image_index}_edited.jpg"
                        generated_image.save(output_image_path)
                    with lock:
                        self._update_results_table(result)
                    return result
                else:
                    return {
                        'image_id': image_id,
                        'image_path': image_path,
                        'error': 'No edit_score_data'
                    }
            except Exception as e:
                return {
                    'image_id': image_id,
                    'image_path': image_path,
                    'error': str(e)
                }
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        # Use multiple workers based on number of GPUs
        max_workers = min(self.num_gpus * 3, len(candidates))  # 3 workers per GPU
        print(f"Using {max_workers} workers across {self.num_gpus} GPUs")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {executor.submit(process_image, path): path for path in candidates}
            for i, future in enumerate(as_completed(future_to_path)):
                result = future.result()
                results.append(result)
                if (i + 1) % 10 == 0:
                    success_count = sum(1 for r in results if 'error' not in r)
                    print(f"Progress: {i + 1}/{len(candidates)}, Success: {success_count}")
        
        # Save results
        results_csv = self.results_dir / "editscore_results.csv"
        if results:
            fieldnames = set()
            for r in results:
                fieldnames.update(r.keys())
            fieldnames = sorted(list(fieldnames))
            
            with open(results_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            
            print(f"\\n✅ Results saved to: {results_csv}")
            
            # Statistics
            success_count = sum(1 for r in results if 'error' not in r)
            print(f"Successfully processed: {success_count}/{len(results)} images")
            
            if success_count > 0:
                valid_results = [r for r in results if 'error' not in r and 'image_rls' in r]
                if valid_results:
                    # Print statistics for the three base metrics
                    print(f"\\n=== Base Metrics Statistics ===")
                    for metric in ['image_rls', 'image_cosine_sim', 'text_similarity']:
                        values = [float(r[metric]) for r in valid_results if metric in r]
                        if values:
                            print(f"{metric:20}: min={min(values):.4f}, max={max(values):.4f}, mean={np.mean(values):.4f}, std={np.std(values):.4f}")
        
        # Calculate and display timing statistics
        end_time = time.time()
        total_time = end_time - start_time
        processed_count = len(candidates)
        
        print(f"\\n=== Timing Statistics ===")
        print(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        if processed_count > 0:
            avg_time_per_image = total_time / processed_count
            print(f"Average time per image: {avg_time_per_image:.2f} seconds")
            print(f"Processing rate: {processed_count/total_time:.2f} images/second")
        
        return str(results_csv)
    
    def run_multi_gpu_evaluation(self, prompt: str = "Apply a cartoon style to the whole image.", limit: int = 1000):
        """Run multi-GPU evaluation with independent model initialization on each GPU"""
        print(f"=== Multi-GPU EditScore Evaluation (Independent GPU Processing) ===")
        
        # Start timing
        start_time = time.time()
        
        # Get image list
        all_candidates = sorted(glob.glob(str(self.images_dir / "*.jpg")))[:limit]
        if not all_candidates:
            print(f"❌ No image files found in {self.images_dir}")
            return None
        
        # Check for already processed images
        processed_images = self._get_processed_images(prompt)
        print(f"Found {len(processed_images)} already processed images")
        
        # Filter out already processed images
        candidates = []
        for img_path in all_candidates:
            image_id = self._generate_image_id(img_path, prompt)
            if image_id not in processed_images:
                candidates.append(img_path)
        
        if not candidates:
            print("✅ All images have been processed!")
            return str(self.results_table_path)
            
        print(f"Found {len(all_candidates)} total images, {len(candidates)} remaining to process...")
        
        # Distribute images across GPUs
        images_per_gpu = len(candidates) // self.num_gpus
        remainder = len(candidates) % self.num_gpus
        
        gpu_image_lists = []
        start_idx = 0
        
        for gpu_id in range(self.num_gpus):
            # Add one extra image to first 'remainder' GPUs
            end_idx = start_idx + images_per_gpu + (1 if gpu_id < remainder else 0)
            gpu_images = candidates[start_idx:end_idx]
            gpu_image_lists.append(gpu_images)
            start_idx = end_idx
            
            print(f"GPU {gpu_id}: {len(gpu_images)} images")
        
        # Process images on each GPU independently
        print(f"Starting independent processing on {self.num_gpus} GPUs...")
        
        # Use spawn method to avoid issues with shared objects
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set
        
        # 设置环境变量优化内存管理
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        with ProcessPoolExecutor(max_workers=self.num_gpus) as executor:
            # Submit tasks to each GPU
            futures = []
            for gpu_id in range(self.num_gpus):
                if gpu_image_lists[gpu_id]:  # Only submit if there are images for this GPU
                    future = executor.submit(
                        process_images_on_gpu,
                        gpu_id,
                        gpu_image_lists[gpu_id],
                        self.model_path,
                        prompt,
                        str(self.results_dir)
                    )
                    futures.append(future)
            
            # Collect results
            all_results = []
            for future in as_completed(futures):
                try:
                    gpu_results = future.result()
                    all_results.extend(gpu_results)
                except Exception as e:
                    print(f"Error in GPU processing: {e}")
        
        # Save results to CSV
        results_csv = self.results_dir / "editscore_results.csv"
        if all_results:
            # Load existing results if any
            existing_df = pd.DataFrame()
            if self.results_table_path.exists():
                try:
                    existing_df = pd.read_csv(self.results_table_path)
                except:
                    pass
            
            # Create new results DataFrame
            new_df = pd.DataFrame(all_results)
            
            # Combine with existing results
            if not existing_df.empty:
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                combined_df = new_df
            
            # Remove duplicates based on image_id
            combined_df = combined_df.drop_duplicates(subset=['image_id'], keep='last')
            
            # Save combined results
            combined_df.to_csv(self.results_table_path, index=False)
            print(f"\\n✅ Results saved to: {self.results_table_path}")
            
            # Statistics
            success_count = sum(1 for r in all_results if 'error' not in r)
            print(f"Successfully processed: {success_count}/{len(all_results)} images")
            
            if success_count > 0:
                valid_results = [r for r in all_results if 'error' not in r and 'image_rls' in r]
                if valid_results:
                    print(f"\\n=== Base Metrics Statistics ===")
                    for metric in ['image_rls', 'image_cosine_sim', 'text_similarity']:
                        values = [float(r[metric]) for r in valid_results if metric in r]
                        if values:
                            print(f"{metric:20}: min={min(values):.4f}, max={max(values):.4f}, mean={np.mean(values):.4f}, std={np.std(values):.4f}")
        
        # Calculate and display timing statistics
        end_time = time.time()
        total_time = end_time - start_time
        processed_count = len(candidates)
        
        print(f"\\n=== Timing Statistics ===")
        print(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        if processed_count > 0:
            avg_time_per_image = total_time / processed_count
            print(f"Average time per image: {avg_time_per_image:.2f} seconds")
            print(f"Processing rate: {processed_count/total_time:.2f} images/second")
        
        return str(self.results_table_path)
    
    def analyze_results(self, results_csv: str, create_sample_labels: bool = True):
        """Analyze evaluation results"""
        print(f"=== Analyzing EditScore Base Metrics Results ===")
        
        if not os.path.exists(results_csv):
            print(f"❌ Results file not found: {results_csv}")
            return None
        
        # Read results
        with open(results_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            results = [row for row in reader if 'error' not in row]
        
        if not results:
            print("❌ No valid result data")
            return None
            
        print(f"Found {len(results)} valid results")
        
        # Create sample labels (simple rule based on scores)
        if create_sample_labels:
            labels_csv = self.results_dir / "sample_labels.csv"
            with open(labels_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['image_path', 'label'])
                writer.writeheader()
                
                for row in results:
                    try:
                        # Use image_rls as primary indicator: higher RLS indicates more editing
                        rls = float(row.get('image_rls', 0))
                        # Simple rule: RLS > 0.5 considered good edit (more change)
                        label = 1 if rls > 0.5 else 0
                        writer.writerow({'image_path': row['image_path'], 'label': label})
                    except:
                        continue
            
            print(f"Sample labels file created: {labels_csv}")
            print("Note: These are automatic labels based on scores, manual annotation needed for actual use!")
            
            # Plot ROC curves
            try:
                self._plot_roc_curves(results_csv, str(labels_csv))
            except Exception as e:
                print(f"Error plotting ROC curves: {e}")
        
        # Basic statistics for three base metrics only
        score_columns = ['image_rls', 'image_cosine_sim', 'text_similarity']
        print(f"\\n=== Base Metrics Statistics ===")
        for col in score_columns:
            try:
                values = [float(r[col]) for r in results if col in r]
                if values:
                    print(f"{col:25}: mean={np.mean(values):.4f}, std={np.std(values):.4f}, "
                          f"min={min(values):.4f}, max={max(values):.4f}")
            except:
                continue
    
    def _plot_roc_curves(self, results_csv: str, labels_csv: str):
        """绘制三个基础指标的ROC曲线"""
        score_columns = ['image_rls', 'image_cosine_sim', 'text_similarity']
        
        # Read data
        results = {}
        with open(results_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'error' not in row:
                    path = os.path.abspath(row['image_path'])
                    results[path] = row
        
        labels_map = {}
        with open(labels_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                path = os.path.abspath(row['image_path'])
                try:
                    label = int(row['label'])
                    labels_map[path] = label
                except:
                    continue
        
        # Match data
        matched_data = []
        for path in labels_map:
            if path in results:
                row_data = {'label': labels_map[path]}
                for col in score_columns:
                    try:
                        row_data[col] = float(results[path].get(col, 0))
                    except:
                        row_data[col] = 0.0
                matched_data.append(row_data)
        
        if not matched_data:
            print("No matching data found, cannot plot ROC curves")
            return
        
        print(f"Found {len(matched_data)} matched samples, plotting ROC curves...")
        
        labels = np.array([d['label'] for d in matched_data])
        
        plt.figure(figsize=(15, 5))
        roc_results = {}
        
        for i, score_col in enumerate(score_columns):
            scores = np.array([d[score_col] for d in matched_data])
            
            # For RLS, smaller values indicate better edit quality, need to negate
            if 'rls' in score_col.lower():
                scores = -scores
            
            try:
                plt.subplot(1, 3, i + 1)
                roc_result = plot_roc_curve(scores, labels, score_col)
                roc_results[score_col] = roc_result
                
                # Add score distribution info
                pos_scores = scores[labels == 1]
                neg_scores = scores[labels == 0]
                plt.text(0.6, 0.2, f'Good: μ={np.mean(pos_scores):.3f}\\nBad: μ={np.mean(neg_scores):.3f}', 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            except Exception as e:
                print(f"Error plotting ROC curve for {score_col}: {e}")
        
        plt.tight_layout()
        
        # Save plot
        roc_image_path = self.results_dir / "roc_curves.png"
        plt.savefig(roc_image_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"ROC curves saved to: {roc_image_path}")
        
        # Print AUC ranking
        print("\\n=== AUC Ranking ===")
        auc_scores = [(col, result.get('auc', 0)) for col, result in roc_results.items()]
        auc_scores.sort(key=lambda x: x[1], reverse=True)
        
        for col, auc in auc_scores:
            print(f"{col:25}: AUC = {auc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="BAGEL EditScore Evaluation Tool")
    parser.add_argument("--mode", choices=["single", "batch", "multi_gpu", "analyze"], required=True,
                       help="Run mode: single=single image test, batch=batch evaluation, multi_gpu=independent GPU processing, analyze=analyze results")
    parser.add_argument("--prompt", default="Apply a cartoon style to the whole image.",
                       help="Edit prompt")
    parser.add_argument("--image_path", help="Image path for single test")
    parser.add_argument("--limit", type=int, default=1000, help="Image count limit for batch evaluation")
    parser.add_argument("--results_csv", help="Results CSV file path for analysis")
    parser.add_argument("--model_path", default="./models/BAGEL-7B-MoT", help="BAGEL model path")
    parser.add_argument("--images_dir", default="./images", help="Images directory path")
    parser.add_argument("--results_dir", default="./results", help="Results directory path")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use for parallel processing")
    parser.add_argument("--device", default="auto", help="Device to use: 'auto', 'cpu', or 'cuda:N' (e.g., 'cuda:0')")
    
    args = parser.parse_args()
    
    print("=== BAGEL EditScore Base Metrics Evaluation Tool ===")
    
    try:
        evaluator = EditScoreEvaluator(
            model_path=args.model_path,
            images_dir=args.images_dir,
            results_dir=args.results_dir,
            num_gpus=args.num_gpus,
            device=args.device
        )
        
        if args.mode == "single":
            result = evaluator.run_single_test(prompt=args.prompt, image_path=args.image_path)
            if result:
                print(f"\\n=== Base Metrics Summary ===")
                print(f"Image RLS: {result['image_rls']:.4f}")
                print(f"Image Cosine Sim: {result['image_cosine_sim']:.4f}")
                print(f"Text Similarity: {result['text_similarity']:.4f}")
                print(f"Results table: {evaluator.results_table_path}")
            
        elif args.mode == "batch":
            results_csv = evaluator.run_batch_evaluation(prompt=args.prompt, limit=args.limit)
            if results_csv:
                print(f"\\nBatch evaluation completed. Use the following command to analyze results:")
                print(f"python {__file__} --mode analyze --results_csv {results_csv}")
                print(f"Results table: {evaluator.results_table_path}")
        
        elif args.mode == "multi_gpu":
            results_csv = evaluator.run_multi_gpu_evaluation(prompt=args.prompt, limit=args.limit)
            if results_csv:
                print(f"\\nMulti-GPU evaluation completed. Use the following command to analyze results:")
                print(f"python {__file__} --mode analyze --results_csv {results_csv}")
                print(f"Results table: {evaluator.results_table_path}")
                
        elif args.mode == "analyze":
            if not args.results_csv:
                # Try using default results file
                default_csv = "results/editscore_results.csv"
                if os.path.exists(default_csv):
                    args.results_csv = default_csv
                else:
                    print("❌ Please specify --results_csv parameter")
                    return
            evaluator.analyze_results(args.results_csv)
            
    except KeyboardInterrupt:
        print("\\nUser interrupted")
    except Exception as e:
        print(f"❌ Execution error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
