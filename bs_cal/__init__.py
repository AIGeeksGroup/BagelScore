"""
BAGEL Similarity Calculator Package
基于BAGEL多模态框架的相似性分数计算器

主要功能：
1. BagelScore_DeCon（解码一致性分数）
2. BagelScore_LangImgCon（语言与图像一致性分数）
"""

from .calculator import BagelSimilarityCalculator
from .config import BagelSimilarityConfig
from .utils import create_output_directory, save_results, load_image
from .cli import main

__version__ = "1.0.0"
__author__ = "BAGEL Similarity Calculator"

__all__ = [
    "BagelSimilarityCalculator",
    "BagelSimilarityConfig", 
    "create_output_directory",
    "save_results",
    "load_image",
    "main"
]
