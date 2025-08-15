#!/usr/bin/env python3
"""
BAGEL相似性计算命令行接口
"""

import argparse
import os
import sys
import logging
import torch
from pathlib import Path
from typing import Optional

from .calculator import BagelSimilarityCalculator
from .parallel_calculator import ParallelBagelCalculator
from .config import BagelSimilarityConfig, create_config_from_env
from .utils import (
    create_output_directory, save_results, save_batch_summary,
    get_image_files, print_results_summary, print_batch_summary,
    export_batch_results
)


def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description='BAGEL Similarity Score Calculation Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 单张图像测试
  python -m bs_cal.cli --mode single --image data_1000/test.jpg
  
  # 批量测试
  python -m bs_cal.cli --mode batch --data-dir data_1000 --batch-size 5
  
  # 自定义配置
  python -m bs_cal.cli --mode single --image test.jpg --prompt "描述这张图片的内容和风格"
  
  # 使用环境变量
  export BAGEL_MODEL_PATH="path/to/model"
  python -m bs_cal.cli --mode single --image test.jpg
        """
    )
    
    # 基本参数
    parser.add_argument('--mode', choices=['single', 'batch', 'parallel'], default='single',
                       help='运行模式: single(单张图像), batch(批量测试), 或 parallel(并行处理)')
    
    # 模型配置
    parser.add_argument('--model-path', type=str, 
                       help='BAGEL模型路径 (也可通过环境变量BAGEL_MODEL_PATH设置)')
    parser.add_argument('--gpu-memory', type=str, default='40GiB',
                       help='每个GPU的最大内存使用量 (默认: 40GiB)')
    
    # 数据配置
    parser.add_argument('--image', type=str,
                       help='单张图像路径 (single模式使用)')
    parser.add_argument('--data-dir', type=str, default='data_1000',
                       help='图像数据目录 (默认: data_1000)')
    parser.add_argument('--batch-size', type=int, default=3,
                       help='批量测试的图像数量 (默认: 3)')
    parser.add_argument('--checkpoint-interval', type=int, default=100,
                       help='并行模式下每N张图像保存一次检查点 (默认: 100)')
    
    # 计算配置
    parser.add_argument('--prompt', type=str, 
                       default='Please describe the content, style and characteristics of this image in detail',
                       help='图像描述提示词')
    parser.add_argument('--text-temperature', type=float, default=0.7,
                       help='文本生成温度参数 (默认: 0.7)')
    parser.add_argument('--max-text-length', type=int, default=200,
                       help='生成文本的最大长度 (默认: 200)')
    
    # 输出配置
    parser.add_argument('--output-dir', type=str,
                       help='输出目录 (默认: 自动生成带时间戳的目录)')
    parser.add_argument('--no-timestamp', action='store_true',
                       help='输出目录不包含时间戳')
    parser.add_argument('--save-images', action='store_true',
                       help='保存处理后的图像')
    
    # 日志配置
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='日志级别 (默认: INFO)')
    parser.add_argument('--log-file', type=str,
                       help='日志文件路径')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='显示详细日志信息')
    
    # 其他选项
    parser.add_argument('--dry-run', action='store_true',
                       help='仅验证配置，不执行计算')
    parser.add_argument('--config-file', type=str,
                       help='从JSON配置文件加载配置')
    
    return parser


def load_config_from_file(config_file: str) -> BagelSimilarityConfig:
    """从JSON文件加载配置"""
    import json
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    
    return BagelSimilarityConfig.from_dict(config_dict)


def update_config_from_args(config: BagelSimilarityConfig, args: argparse.Namespace) -> BagelSimilarityConfig:
    """从命令行参数更新配置"""
    
    # 模型配置
    if args.model_path:
        config.model.model_path = args.model_path
    if args.gpu_memory:
        config.model.max_mem_per_gpu = args.gpu_memory
    
    # 数据配置
    if args.data_dir:
        config.data.data_dir = args.data_dir
    
    # 计算配置
    if args.prompt:
        config.calculation.default_prompt = args.prompt
    if args.text_temperature:
        config.calculation.text_temperature = args.text_temperature
    if args.max_text_length:
        config.calculation.max_text_length = args.max_text_length
    
    # 输出配置
    if args.output_dir:
        config.output.output_dir_prefix = args.output_dir
    if args.no_timestamp:
        config.output.include_timestamp = False
    if args.save_images:
        config.output.save_images = True
    
    # 日志配置
    if args.log_level:
        config.logging.level = args.log_level
    if args.log_file:
        config.logging.file = args.log_file
    if args.verbose:
        config.logging.level = 'DEBUG'
    
    # 测试配置
    if args.batch_size:
        config.test.batch_size = args.batch_size
    
    return config


def run_single_image_mode(args: argparse.Namespace, config: BagelSimilarityConfig):
    """运行单张图像模式"""
    if not args.image:
        raise ValueError("Single mode requires --image argument")
    
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image file does not exist: {args.image}")
    
    # 创建输出目录
    output_dir = create_output_directory(config)
    
    try:
        # 初始化计算器
        logger.info("Initializing BAGEL similarity calculator...")
        calculator = BagelSimilarityCalculator(config)
        
        # 计算相似性分数
        logger.info(f"Processing image: {args.image}")
        results = calculator.calculate_all_scores(args.image, args.prompt)
        results['image_path'] = args.image
        
        # 保存结果
        image_name = Path(args.image).stem
        save_results(results, output_dir, image_name, config)
        
        # 打印结果摘要
        print_results_summary(results, os.path.basename(args.image))
        print(f"Results saved to: {output_dir}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during single image processing: {e}")
        raise


def run_batch_mode(args: argparse.Namespace, config: BagelSimilarityConfig):
    """运行批量模式"""
    # 获取图像文件列表
    try:
        image_files = get_image_files(config.data.data_dir, config.data.supported_formats)
        if not image_files:
            raise ValueError(f"No supported image files found in directory: {config.data.data_dir}")
        
        # 限制批量大小
        image_files = image_files[:config.test.batch_size]
        logger.info(f"Found {len(image_files)} images for batch processing")
        
    except Exception as e:
        logger.error(f"Error getting image files: {e}")
        raise
    
    # 创建输出目录
    output_dir = create_output_directory(config)
    
    try:
        # 初始化计算器
        logger.info("Initializing BAGEL similarity calculator...")
        calculator = BagelSimilarityCalculator(config)
        
        # 批量计算
        logger.info(f"Starting batch processing of {len(image_files)} images...")
        results = calculator.batch_calculate(image_files, args.prompt)
        
        # 保存结果
        for i, result in enumerate(results):
            if 'image_path' in result:
                image_name = Path(result['image_path']).stem
                save_results(result, output_dir, f"{i+1:02d}_{image_name}", config)
        
        # 保存批量摘要
        save_batch_summary(results, output_dir, config)
        
        # 导出CSV和JSON格式的结果
        export_paths = export_batch_results(results, output_dir)
        
        # 打印结果摘要
        print_batch_summary(results)
        print(f"Results saved to: {output_dir}")
        
        if export_paths:
            print(f"\n📈 Analysis files:")
            if 'csv' in export_paths:
                print(f"  📊 CSV (Excel): {export_paths['csv']}")
            if 'json' in export_paths:
                print(f"  📋 JSON (Programming): {export_paths['json']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during batch processing: {e}")
        raise


def run_parallel_mode(args: argparse.Namespace, config: BagelSimilarityConfig):
    """运行并行处理模式"""
    # 获取图像文件列表
    try:
        image_files = get_image_files(config.data.data_dir, config.data.supported_formats)
        if not image_files:
            raise ValueError(f"No supported images found in {config.data.data_dir}")
        
        # 限制批量大小
        if args.batch_size and args.batch_size > 0:
            image_files = image_files[:args.batch_size]
        
        logger.info(f"Found {len(image_files)} images for parallel processing")
        
    except Exception as e:
        logger.error(f"Error getting image files: {e}")
        raise
    
    # 创建输出目录
    try:
        output_dir_prefix = getattr(config.output, 'output_dir_prefix', 'bs_cal/bagel_results')
        output_dir = create_output_directory(output_dir_prefix)
        logger.info(f"Results will be saved to: {output_dir}")
        
        # 初始化并行计算器
        logger.info("Initializing parallel BAGEL similarity calculator...")
        parallel_calculator = ParallelBagelCalculator(config)
        
        # 并行计算
        logger.info(f"Starting parallel processing of {len(image_files)} images...")
        logger.info(f"Checkpoint interval: every {args.checkpoint_interval} images")
        logger.info(f"Available GPUs: {torch.cuda.device_count()}")
        
        results = parallel_calculator.process_batch_parallel(
            image_files, 
            args.prompt,
            checkpoint_interval=args.checkpoint_interval
        )
        
        # 保存结果
        for i, result in enumerate(results):
            if 'image_path' in result:
                image_name = Path(result['image_path']).stem
                save_results(result, output_dir, f"{i+1:04d}_{image_name}", config)
        
        # 保存批量摘要
        save_batch_summary(results, output_dir, config)
        
        # 导出CSV和JSON格式的结果
        export_paths = export_batch_results(results, output_dir)
        
        # 打印结果摘要
        print_batch_summary(results)
        print(f"Results saved to: {output_dir}")
        
        if export_paths:
            print(f"\n📈 Analysis files:")
            if 'csv' in export_paths:
                print(f"  📊 CSV (Excel): {export_paths['csv']}")
            if 'json' in export_paths:
                print(f"  📋 JSON (Programming): {export_paths['json']}")
        
        # GPU performance summary
        successful_results = [r for r in results if 'error' not in r]
        if successful_results:
            gpu_stats = {}
            for result in successful_results:
                gpu_id = result.get('gpu_id', 'unknown')
                if gpu_id not in gpu_stats:
                    gpu_stats[gpu_id] = []
                gpu_stats[gpu_id].append(result.get('calculation_time', 0))
            
            print(f"\n🚀 GPU Performance Summary:")
            for gpu_id, times in gpu_stats.items():
                avg_time = sum(times) / len(times)
                print(f"  GPU {gpu_id}: {len(times)} images, avg {avg_time:.2f}s/image")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during parallel processing: {e}")
        raise


def load_config_from_file(config_file: str) -> BagelSimilarityConfig:
    """从文件加载配置"""
    return BagelSimilarityConfig.load_from_file(config_file)


def update_config_from_args(config: BagelSimilarityConfig, args: argparse.Namespace) -> BagelSimilarityConfig:
    """从命令行参数更新配置"""
    # 模型配置
    if args.model_path:
        config.model.model_path = args.model_path
    if args.gpu_memory:
        config.model.max_mem_per_gpu = args.gpu_memory
    if args.dtype:
        config.model.dtype = args.dtype
    
    # 数据配置
    if args.data_dir:
        config.data.data_dir = args.data_dir
    
    # 计算配置
    if args.prompt:
        config.calculation.default_prompt = args.prompt
    
    # 输出配置
    if args.output_dir:
        config.output.output_dir_prefix = args.output_dir
    
    # 测试配置
    if args.batch_size:
        config.test.batch_size = args.batch_size
    
    return config


def main():
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            *([logging.FileHandler(args.log_file)] if args.log_file else [])
        ]
    )
    
    global logger
    logger = logging.getLogger(__name__)
    
    try:
        # 加载配置
        if args.config_file:
            config = load_config_from_file(args.config_file)
        else:
            config = create_config_from_env()
        
        # 从命令行参数更新配置
        config = update_config_from_args(config, args)
        
        # 验证配置
        if not config.validate():
            logger.error("Configuration validation failed")
            sys.exit(1)
        
        # 干运行模式
        if args.dry_run:
            logger.info("Configuration validation passed (dry run mode)")
            return
        
        # 根据模式运行
        if args.mode == 'single':
            run_single_image_mode(args, config)
        elif args.mode == 'batch':
            run_batch_mode(args, config)
        else:  # parallel mode
            run_parallel_mode(args, config)
            
    except KeyboardInterrupt:
        logger.info("User interrupted execution")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
