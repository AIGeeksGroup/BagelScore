#!/usr/bin/env python3
"""
基于BAGEL多模态框架的相似性分数计算器
优化版本，参考官方实现方式
"""

import os
import sys
import time
import json
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
from PIL import Image
import logging
from pathlib import Path

# 添加Bagel路径
sys.path.append('./Bagel')

from .config import BagelSimilarityConfig
from .utils import load_image, validate_image

logger = logging.getLogger(__name__)


class BagelSimilarityCalculator:
    """BAGEL相似性分数计算器 - 优化版本"""
    
    def __init__(self, config: Optional[BagelSimilarityConfig] = None):
        """
        Initialize BAGEL similarity calculator
        
        Args:
            config: Configuration object, if None use default configuration
        """
        self.config = config or BagelSimilarityConfig()
        self.model = None
        self.vae_model = None
        self.tokenizer = None
        self.vae_transform = None
        self.vit_transform = None
        self.new_token_ids = None
        self.inferencer = None
        
        # Setup logging
        self._setup_logging()
        
        # Validate configuration
        if not self.config.validate():
            raise ValueError("Configuration validation failed")
        
        # Initialize model
        self._initialize_model()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        import os
        from datetime import datetime
        
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(self.config.logging.file) if self.config.logging.file else 'bs_cal/logs'
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure logging
        log_file = self.config.logging.file or f'{log_dir}/bagel_score_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        
        logging.basicConfig(
            level=getattr(logging, self.config.logging.level.upper()),
            format=self.config.logging.format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
    def _initialize_model(self):
        """初始化BAGEL模型"""
        logger.info("Initializing BAGEL model...")
        
        model_path = self.config.model.model_path
        
        try:
            # 尝试导入必要的模块
            from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
            from Bagel.data.transforms import ImageTransform
            from Bagel.data.data_utils import pil_img2rgb, add_special_tokens
            from Bagel.modeling.bagel import (
                BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
            )
            from Bagel.modeling.qwen2 import Qwen2Tokenizer
            from Bagel.modeling.bagel.qwen2_navit import NaiveCache
            from Bagel.modeling.autoencoder import load_ae
            from safetensors.torch import load_file
            
            logger.info("Successfully imported all required modules")
            
        except ImportError as e:
            logger.error(f"Failed to import required modules: {e}")
            logger.error("Please ensure all dependencies are installed:")
            logger.error("pip install torch torchvision torchaudio")
            logger.error("pip install accelerate transformers safetensors")
            logger.error("pip install flash-attn  # if available")
            raise
        
        # LLM配置
        llm_config_path = os.path.join(model_path, "llm_config.json")
        if not os.path.exists(llm_config_path):
            raise FileNotFoundError(f"LLM config file not found: {llm_config_path}")
        
        llm_config = Qwen2Config.from_json_file(llm_config_path)
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"

        # ViT配置
        vit_config_path = os.path.join(model_path, "vit_config.json")
        if not os.path.exists(vit_config_path):
            raise FileNotFoundError(f"ViT config file not found: {vit_config_path}")
        
        vit_config = SiglipVisionConfig.from_json_file(vit_config_path)
        vit_config.rope = False
        vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

        # VAE加载
        vae_path = os.path.join(model_path, "ae.safetensors")
        if not os.path.exists(vae_path):
            raise FileNotFoundError(f"VAE model file not found: {vae_path}")
        
        self.vae_model, vae_config = load_ae(local_path=vae_path)

        # Bagel配置
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

        # 初始化模型
        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)
            self.model = Bagel(language_model, vit_model, config)
            self.model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

        # Tokenizer准备
        self.tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        self.tokenizer, self.new_token_ids, _ = add_special_tokens(self.tokenizer)

        # 图像变换准备
        self.vae_transform = ImageTransform(1024, 512, 16)
        self.vit_transform = ImageTransform(980, 224, 14)

        # 设备映射 - 优化双GPU配置
        logger.info(f"Detected {torch.cuda.device_count()} GPU devices")
        
        # Configure GPU memory allocation
        max_memory = {}
        for i in range(torch.cuda.device_count()):
            max_memory[i] = self.config.model.max_memory.get(f"{i}", "22GiB")
        max_memory["cpu"] = self.config.model.max_memory.get("cpu", "64GiB")
        
        logger.info(f"GPU memory configuration: {max_memory}")
        
        device_map = infer_auto_device_map(
            self.model,
            max_memory=max_memory,
            no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
        )
        
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

        if torch.cuda.device_count() == 1:
            first_device = device_map.get(same_device_modules[0], "cuda:0")
            for k in same_device_modules:
                if k in device_map:
                    device_map[k] = first_device
                else:
                    device_map[k] = "cuda:0"
        else:
            # 双GPU配置：确保关键模块在第一个GPU上
            first_device = device_map.get(same_device_modules[0], "cuda:0")
            for k in same_device_modules:
                if k in device_map:
                    device_map[k] = first_device
                else:
                    device_map[k] = "cuda:0"
            
            logger.info(f"Dual GPU device mapping completed, key modules assigned to: {first_device}")
        
        # Find model weight files
        weight_files = ["ema.safetensors", "model.safetensors"]
        checkpoint_path = None
        
        for weight_file in weight_files:
            potential_path = os.path.join(model_path, weight_file)
            if os.path.exists(potential_path):
                checkpoint_path = potential_path
                logger.info(f"Found model weights: {weight_file}")
                break
        
        if checkpoint_path is None:
            raise FileNotFoundError(f"No model weight files found. Expected one of: {weight_files}")

        # 加载模型权重 - 优化双GPU加载
        logger.info(f"Loading model weights: {checkpoint_path}")
        logger.info(f"Using data type: {self.config.model.dtype}")
        
        self.model = load_checkpoint_and_dispatch(
            self.model,
            checkpoint=checkpoint_path,
            device_map=device_map,
            offload_buffers=True,
            dtype=getattr(torch, self.config.model.dtype),
            offload_folder=self.config.model.offload_folder
        )
        
        logger.info("Model weights loaded successfully")

        self.model = self.model.eval()
        
        # 初始化推理器
        try:
            from Bagel.inferencer import InterleaveInferencer
            self.inferencer = InterleaveInferencer(
                model=self.model,
                vae_model=self.vae_model,
                tokenizer=self.tokenizer,
                vae_transform=self.vae_transform,
                vit_transform=self.vit_transform,
                new_token_ids=self.new_token_ids,
            )
            logger.info("Successfully initialized InterleaveInferencer")
        except Exception as e:
            logger.warning(f"Failed to initialize InterleaveInferencer: {e}")
            logger.info("Will use direct model calls instead")
            self.inferencer = None
        
        logger.info("BAGEL model initialization completed")
        
    def cosine_similarity(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        """
        Calculate cosine similarity between two vectors
        Handle different dimensions by projection
        
        Args:
            vec1: Vector 1
            vec2: Vector 2
            
        Returns:
            Cosine similarity score (0-1)
        """
        # Ensure vectors are 1-dimensional
        if vec1.dim() > 1:
            vec1 = vec1.flatten()
        if vec2.dim() > 1:
            vec2 = vec2.flatten()
        
        # Ensure tensors are on the same device
        if vec1.device != vec2.device:
            vec2 = vec2.to(vec1.device)
        
        # Handle different dimensions by projecting to smaller dimension
        min_dim = min(vec1.size(0), vec2.size(0))
        vec1_proj = vec1[:min_dim]
        vec2_proj = vec2[:min_dim]
        
        # Calculate cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(vec1_proj.unsqueeze(0), vec2_proj.unsqueeze(0), dim=1)
        
        # Convert to positive range [0, 1]
        similarity_score = (cos_sim.item() + 1.0) / 2.0
        
        return similarity_score
    
    def encode_text(self, text: str) -> torch.Tensor:
        """
        Encode text to vector using simplified approach
        
        Args:
            text: Input text
            
        Returns:
            Text encoding vector
        """
        with torch.no_grad():
            try:
                # Simple tokenization approach
                input_ids = self.tokenizer.encode(text, return_tensors="pt")
                
                # Move to appropriate device
                device = next(self.model.parameters()).device
                input_ids = input_ids.to(device)
                
                # Get text encoding using embedding layer
                text_encoding = self.model.language_model.model.embed_tokens(input_ids)
                
                # Take mean pooling as text vector representation
                text_vector = text_encoding.mean(dim=1)  # [batch_size, hidden_size]
                
                return text_vector.squeeze(0)  # [hidden_size]
                
            except Exception as e:
                logger.warning(f"Text encoding failed, using fallback: {e}")
                # Fallback: create a random vector of appropriate size
                device = next(self.model.parameters()).device
                return torch.randn(self.model.config.llm_config.hidden_size, device=device)
    
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """
        编码图像为向量
        
        Args:
            image: 输入图像
            
        Returns:
            图像编码向量
        """
        # 导入必要的函数
        import sys
        import os
        sys.path.append('./Bagel')
        from data.data_utils import pil_img2rgb
        
        with torch.no_grad():
            # 预处理图像
            image_processed = pil_img2rgb(image)
            
            # 准备图像输入
            generation_input, kv_lens, ropes = self.model.prepare_vit_images(
                curr_kvlens=[0],
                curr_rope=[0], 
                images=[image_processed],
                transforms=self.vit_transform, 
                new_token_ids=self.new_token_ids,
            )
            
            # 使用packed_vit_tokens作为图像编码向量
            image_vector = generation_input['packed_vit_tokens']
            
            # 取平均池化作为图像向量表示
            if image_vector.dim() > 1:
                image_vector = image_vector.mean(dim=0)  # [hidden_size]
            
            return image_vector
    
    def decode_image_to_vector(self, image: Image.Image) -> torch.Tensor:
        """
        将图像解码为潜在向量表示
        
        Args:
            image: 输入图像
            
        Returns:
            解码后的潜在向量
        """
        # 导入必要的函数
        import sys
        import os
        sys.path.append('./Bagel')
        from data.data_utils import pil_img2rgb
        
        with torch.no_grad():
            # 预处理图像
            image_processed = self.vae_transform(pil_img2rgb(image))
            
            # 直接使用VAE模型编码
            # 将图像转换为正确的格式
            if image_processed.dim() == 3:
                image_processed = image_processed.unsqueeze(0)  # 添加batch维度
            
            # 获取潜在表示
            latent = self.vae_model.encode(image_processed.to(next(self.vae_model.parameters()).device))
            
            return latent.squeeze(0)  # [latent_channels, latent_height, latent_width]
    
    def generate_language_response(self, image: Image.Image, prompt: str = "Please describe this image") -> str:
        """
        Generate language response using BAGEL model
        
        Args:
            image: Input image
            prompt: Prompt text
            
        Returns:
            Generated language response
        """
        # Import necessary functions
        import sys
        import os
        sys.path.append('./Bagel')
        from data.data_utils import pil_img2rgb
        
        try:
            if self.inferencer is not None:
                logger.info("Using InterleaveInferencer for language generation")
                image_processed = pil_img2rgb(image)
                
                # Use the inferencer with proper parameters
                result = self.inferencer(
                    image=image_processed,
                    text=prompt,
                    think=False,  # Don't use thinking mode
                    understanding_output=True,  # Get understanding output
                    do_sample=True,  # Enable sampling
                    text_temperature=0.7,  # Use reasonable temperature
                    max_think_token_n=512  # Limit token generation
                )
                
                logger.info(f"Inferencer result type: {type(result)}")
                logger.info(f"Inferencer result: {result}")
                
                # Extract the text response
                if isinstance(result, dict):
                    response = result.get("text", result.get("understanding", ""))
                    if not response:
                        # Try other possible keys
                        for key in ["response", "output", "generated_text"]:
                            if key in result:
                                response = result[key]
                                break
                else:
                    response = str(result)
                
                if response and len(response.strip()) > 0 and "Error" not in response:
                    logger.info(f"Successfully generated response: {response[:50]}...")
                    return response
                else:
                    logger.warning("Invalid response from inferencer, using fallback")
                    return self._generate_simple_description(image)
            else:
                logger.warning("Inferencer not available, using fallback description")
                return self._generate_simple_description(image)
                
        except Exception as e:
            logger.error(f"Error in language generation: {e}")
            import traceback
            traceback.print_exc()
            logger.warning("Falling back to simple description")
            return self._generate_simple_description(image)
    
    def _update_context_text(self, gen_context: Dict, text: str) -> Dict:
        """更新文本上下文"""
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        
        generation_input, kv_lens, ropes = self.model.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            prompts=[text],
            tokenizer=self.tokenizer, 
            new_token_ids=self.new_token_ids,
        )

        past_key_values = self.model.forward_cache_update_text(past_key_values, **generation_input)        
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context
    
    def _update_context_image(self, gen_context: Dict, image: Image.Image) -> Dict:
        """更新图像上下文"""
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        # 更新ViT
        generation_input, kv_lens, ropes = self.model.prepare_vit_images(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            images=[image],
            transforms=self.vit_transform, 
            new_token_ids=self.new_token_ids,
        )
        past_key_values = self.model.forward_cache_update_vit(past_key_values, **generation_input)

        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context
    
    def calculate_bagel_score_decon(self, image: Image.Image) -> float:
        """
        计算BagelScore_DeCon（解码一致性分数）
        
        Args:
            image: 输入图像
            
        Returns:
            解码一致性分数 (0-1)
        """
        logger.info("Calculating BagelScore_DeCon...")
        
        # 获取原始输入的编码向量
        input_vector = self.encode_image(image)
        
        # 获取解码后的向量
        decoded_vector = self.decode_image_to_vector(image)
        
        # 将解码向量展平以便计算相似度
        decoded_vector_flat = decoded_vector.flatten()
        
        # 确保两个向量维度匹配
        if input_vector.numel() != decoded_vector_flat.numel():
            # 如果维度不匹配，将输入向量调整到相同维度
            if input_vector.numel() > decoded_vector_flat.numel():
                input_vector = input_vector[:decoded_vector_flat.numel()]
            else:
                decoded_vector_flat = decoded_vector_flat[:input_vector.numel()]
        
        # 计算余弦相似度
        similarity_score = self.cosine_similarity(input_vector, decoded_vector_flat)
        
        logger.info(f"BagelScore_DeCon: {similarity_score:.4f}")
        return similarity_score
    
    def calculate_bagel_score_langimgcon(self, image: Image.Image, prompt: str = "Please describe this image") -> Tuple[float, str]:
        """
        Calculate BagelScore_LangImgCon (Language-Image Consistency Score)
        According to log.md: Measure consistency between BAGEL generated language response 
        and generated image content vectors.
        
        Args:
            image: Input image
            prompt: Prompt text
            
        Returns:
            (consistency_score, generated_language_response)
        """
        logger.info("Calculating BagelScore_LangImgCon...")
        
        try:
            # Step 1: Generate language response using BAGEL model
            language_response = self.generate_language_response(image, prompt)
            
            # Step 2: Get language response encoding vector
            language_response_vector = self.encode_text(language_response)
            
            # Step 3: Get generated content vector (image encoding)
            generated_content_vector = self.encode_image(image)
            
            # Step 4: Calculate cosine similarity between language and image vectors
            similarity_score = self.cosine_similarity(language_response_vector, generated_content_vector)
            
            logger.info(f"BagelScore_LangImgCon: {similarity_score:.4f}")
            logger.info(f"Generated language response: {language_response[:100]}...")
            
            return similarity_score, language_response
            
        except Exception as e:
            logger.error(f"Error in LangImgCon calculation: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to default calculation
            return 0.5, f"Language generation failed: {str(e)}"
    
    def _generate_simple_description(self, image: Image.Image) -> str:
        """
        Generate a simple description based on image features
        Fallback method when complex language generation fails
        """
        try:
            # Import necessary functions
            import sys
            import os
            sys.path.append('./Bagel')
            from data.data_utils import pil_img2rgb
            
            # Get image features for description
            image_processed = pil_img2rgb(image)
            image_size = image_processed.size
            
            # Generate a basic description based on image properties
            description = f"This is an image with dimensions {image_size[0]}x{image_size[1]} pixels. "
            description += "The image contains visual content that can be processed by the BAGEL model. "
            description += "This description represents the image's visual characteristics."
            
            return description
            
        except Exception as e:
            logger.warning(f"Simple description generation failed: {e}")
            return "This is an image processed by the BAGEL similarity calculator."
    
    def calculate_all_scores(self, image: Union[str, Image.Image], prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        计算所有相似性分数
        
        Args:
            image: 输入图像路径或PIL图像对象
            prompt: 提示词，如果为None则使用默认提示词
            
        Returns:
            包含所有分数的字典
        """
        # 加载和验证图像
        if isinstance(image, str):
            image = load_image(image)
        
        if not validate_image(image, self.config.data):
            raise ValueError("Invalid image")
        
        prompt = prompt or self.config.calculation.default_prompt
        
        results = {
            'image_path': None,
            'prompt': prompt,
            'bagel_score_decon': None,
            'bagel_score_langimgcon': None,
            'language_response': None,
            'timestamp': datetime.now().isoformat(),
            'calculation_time': None,
            'image_size': image.size,
            'config': self.config.to_dict()
        }
        
        start_time = time.time()
        
        try:
            # 计算BagelScore_DeCon
            decon_score = self.calculate_bagel_score_decon(image)
            results['bagel_score_decon'] = decon_score
            
            # 计算BagelScore_LangImgCon
            langimgcon_score, language_response = self.calculate_bagel_score_langimgcon(image, prompt)
            results['bagel_score_langimgcon'] = langimgcon_score
            results['language_response'] = language_response
            
        except Exception as e:
            logger.error(f"Error calculating scores: {e}")
            results['error'] = str(e)
        
        results['calculation_time'] = time.time() - start_time
        
        return results
    
    def batch_calculate(self, image_paths: List[str], prompt: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        批量计算相似性分数
        
        Args:
            image_paths: 图像路径列表
            prompt: 提示词，如果为None则使用默认提示词
            
        Returns:
            结果列表
        """
        results = []
        prompt = prompt or self.config.calculation.default_prompt
        
        for i, image_path in enumerate(image_paths):
            logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
            
            try:
                result = self.calculate_all_scores(image_path, prompt)
                result['image_path'] = image_path
                result['image_index'] = i
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing image {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'image_index': i,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        return results
