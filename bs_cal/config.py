#!/usr/bin/env python3
"""
BAGEL相似性计算器配置管理
"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """模型配置"""
    model_path: str = "Bagel/BAGEL-7B-MoT"
    max_mem_per_gpu: str = "22GiB"  # 4090有24GB显存，留2GB给系统
    dtype: str = "bfloat16"
    offload_folder: str = "offload"
    device_map: str = "auto"  # 自动分配到所有可用GPU
    max_memory: Dict[str, str] = field(default_factory=lambda: {
        "0": "22GiB",  # GPU 0
        "1": "22GiB",  # GPU 1
        "cpu": "64GiB"  # CPU内存
    })


@dataclass
class DataConfig:
    """数据配置"""
    data_dir: str = "data_1000"
    supported_formats: List[str] = field(default_factory=lambda: [".jpg", ".jpeg", ".png", ".bmp", ".tiff"])
    max_image_size: int = 2048
    min_image_size: int = 64


@dataclass
class CalculationConfig:
    """计算配置"""
    default_prompt: str = "请描述这张图片"
    text_temperature: float = 0.7
    max_text_length: int = 100
    do_sample: bool = True


@dataclass
class OutputConfig:
    """输出配置"""
    output_dir_prefix: str = "bs_cal/logs/bagel_results"
    save_json: bool = True
    save_summary: bool = True
    save_images: bool = False
    include_timestamp: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    file: Optional[str] = "bs_cal/logs/bagel_score.log"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class TestConfig:
    """测试配置"""
    batch_size: int = 10


@dataclass
class BagelSimilarityConfig:
    """BAGEL相似性计算器主配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    calculation: CalculationConfig = field(default_factory=CalculationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    test: TestConfig = field(default_factory=TestConfig)
    
    def validate(self) -> bool:
        """验证配置"""
        try:
            # 检查模型路径
            if not os.path.exists(self.model.model_path):
                logger.warning(f"Model path does not exist: {self.model.model_path}")
                return False
            
            # 检查数据目录
            if not os.path.exists(self.data.data_dir):
                logger.warning(f"Data directory does not exist: {self.data.data_dir}")
                return False
            
            # 检查模型文件
            required_files = ["llm_config.json", "vit_config.json", "ae.safetensors"]
            for file in required_files:
                file_path = os.path.join(self.model.model_path, file)
                if not os.path.exists(file_path):
                    logger.warning(f"Required model file missing: {file_path}")
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BagelSimilarityConfig':
        """从字典创建配置"""
        # 处理嵌套配置
        model_config = ModelConfig(**config_dict.get('model', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        calculation_config = CalculationConfig(**config_dict.get('calculation', {}))
        output_config = OutputConfig(**config_dict.get('output', {}))
        logging_config = LoggingConfig(**config_dict.get('logging', {}))
        test_config = TestConfig(**config_dict.get('test', {}))
        
        return cls(
            model=model_config,
            data=data_config,
            calculation=calculation_config,
            output=output_config,
            logging=logging_config,
            test=test_config
        )
    
    def save_to_file(self, file_path: str):
        """保存配置到文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'BagelSimilarityConfig':
        """从文件加载配置"""
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


def create_config_from_env() -> BagelSimilarityConfig:
    """从环境变量创建配置"""
    config = BagelSimilarityConfig()
    
    # 从环境变量更新配置
    if os.getenv('BAGEL_MODEL_PATH'):
        config.model.model_path = os.getenv('BAGEL_MODEL_PATH')
    
    if os.getenv('BAGEL_DATA_DIR'):
        config.data.data_dir = os.getenv('BAGEL_DATA_DIR')
    
    if os.getenv('BAGEL_GPU_MEMORY'):
        config.model.max_mem_per_gpu = os.getenv('BAGEL_GPU_MEMORY')
    
    if os.getenv('BAGEL_DEFAULT_PROMPT'):
        config.calculation.default_prompt = os.getenv('BAGEL_DEFAULT_PROMPT')
    
    if os.getenv('BAGEL_OUTPUT_DIR'):
        config.output.output_dir_prefix = os.getenv('BAGEL_OUTPUT_DIR')
    
    return config

