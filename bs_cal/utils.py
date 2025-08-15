#!/usr/bin/env python3
"""
BAGEL相似性计算工具函数
"""

import os
import json
import csv
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from PIL import Image
import logging
from pathlib import Path

from .config import BagelSimilarityConfig, DataConfig

logger = logging.getLogger(__name__)


def load_image(image_path: str) -> Image.Image:
    """
    加载图像文件
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        PIL图像对象
    """
    try:
        image = Image.open(image_path)
        # 转换为RGB模式
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception as e:
        raise ValueError(f"Failed to load image {image_path}: {e}")


def validate_image(image: Image.Image, data_config: DataConfig) -> bool:
    """
    验证图像是否符合要求
    
    Args:
        image: PIL图像对象
        data_config: 数据配置
        
    Returns:
        是否有效
    """
    if image is None:
        return False
    
    # 检查图像尺寸
    width, height = image.size
    if width < data_config.min_image_size or height < data_config.min_image_size:
        logger.warning(f"Image too small: {width}x{height}")
        return False
    
    if width > data_config.max_image_size or height > data_config.max_image_size:
        logger.warning(f"Image too large: {width}x{height}")
        return False
    
    return True


def create_output_directory(config: BagelSimilarityConfig) -> str:
    """
    创建带时间戳的输出目录
    
    Args:
        config: 配置对象
        
    Returns:
        输出目录路径
    """
    if config.output.include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{config.output.output_dir_prefix}_{timestamp}"
    else:
        output_dir = config.output.output_dir_prefix
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")
    return output_dir


def save_results(results: Dict[str, Any], output_dir: str, image_name: str, config: BagelSimilarityConfig):
    """
    保存结果到文件
    
    Args:
        results: 结果字典
        output_dir: 输出目录
        image_name: 图像名称
        config: 配置对象
    """
    # 保存JSON结果
    if config.output.save_json:
        json_filename = f"{image_name}_results.json"
        json_path = os.path.join(output_dir, json_filename)
        
        # 处理numpy数组和tensor的序列化
        serializable_results = _make_serializable(results)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to: {json_path}")
    
    # 保存摘要到文本文件
    if config.output.save_summary:
        summary_filename = f"{image_name}_summary.txt"
        summary_path = os.path.join(output_dir, summary_filename)
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"Image: {image_name}\n")
            f.write(f"Prompt: {results.get('prompt', 'N/A')}\n")
            f.write(f"Calculation Time: {results.get('timestamp', 'N/A')}\n")
            f.write(f"Processing Time: {results.get('calculation_time', 0):.2f} seconds\n\n")
            
            if 'bagel_score_decon' in results and results['bagel_score_decon'] is not None:
                f.write(f"BagelScore_DeCon (Decoding Consistency Score): {results['bagel_score_decon']:.4f}\n")
            
            if 'bagel_score_langimgcon' in results and results['bagel_score_langimgcon'] is not None:
                f.write(f"BagelScore_LangImgCon (Language-Image Consistency Score): {results['bagel_score_langimgcon']:.4f}\n")
            
            if 'language_response' in results and results['language_response']:
                f.write(f"Generated Language Response: {results['language_response']}\n")
            
            if 'error' in results:
                f.write(f"Error: {results['error']}\n")
        
        logger.info(f"Summary saved to: {summary_path}")


def _make_serializable(obj: Any) -> Any:
    """
    将对象转换为可序列化的格式
    
    Args:
        obj: 输入对象
        
    Returns:
        可序列化的对象
    """
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):  # torch.Tensor
        return obj.item()
    elif hasattr(obj, 'size'):  # PIL.Image
        return f"PIL.Image({obj.size})"
    else:
        return obj


def save_batch_summary(results: List[Dict[str, Any]], output_dir: str, config: BagelSimilarityConfig):
    """
    保存批量处理结果摘要
    
    Args:
        results: 结果列表
        output_dir: 输出目录
        config: 配置对象
    """
    if not config.output.save_json:
        return
    
    # 计算统计信息
    successful_results = [r for r in results if 'error' not in r]
    failed_results = [r for r in results if 'error' in r]
    
    summary = {
        'total_images': len(results),
        'successful_images': len(successful_results),
        'failed_images': len(failed_results),
        'timestamp': datetime.now().isoformat(),
        'config': config.to_dict()
    }
    
    if successful_results:
        # 计算平均分数
        decon_scores = [r['bagel_score_decon'] for r in successful_results if r.get('bagel_score_decon') is not None]
        langimgcon_scores = [r['bagel_score_langimgcon'] for r in successful_results if r.get('bagel_score_langimgcon') is not None]
        
        if decon_scores:
            summary['decon_stats'] = {
                'mean': np.mean(decon_scores),
                'std': np.std(decon_scores),
                'min': np.min(decon_scores),
                'max': np.max(decon_scores)
            }
        
        if langimgcon_scores:
            summary['langimgcon_stats'] = {
                'mean': np.mean(langimgcon_scores),
                'std': np.std(langimgcon_scores),
                'min': np.min(langimgcon_scores),
                'max': np.max(langimgcon_scores)
            }
        
        # 计算总时间
        total_time = sum(r.get('calculation_time', 0) for r in successful_results)
        summary['total_time'] = total_time
        summary['average_time'] = total_time / len(successful_results)
    
    # 保存摘要
    summary_path = os.path.join(output_dir, "batch_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Batch summary saved to: {summary_path}")
    
    # 保存详细结果
    detailed_path = os.path.join(output_dir, "all_results.json")
    serializable_results = _make_serializable(results)
    with open(detailed_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Detailed results saved to: {detailed_path}")


def get_image_files(data_dir: str, supported_formats: List[str]) -> List[str]:
    """
    获取指定目录下的图像文件列表
    
    Args:
        data_dir: 数据目录
        supported_formats: 支持的图像格式
        
    Returns:
        图像文件路径列表
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    
    image_files = []
    for file in os.listdir(data_dir):
        if any(file.lower().endswith(ext) for ext in supported_formats):
            image_files.append(os.path.join(data_dir, file))
    
    return sorted(image_files)


def print_results_summary(results: Dict[str, Any], image_name: str = "Unknown"):
    """
    打印结果摘要
    
    Args:
        results: 结果字典
        image_name: 图像名称
    """
    print("\n" + "="*60)
    print("BAGEL Similarity Score Calculation Results")
    print("="*60)
    print(f"Test Image: {image_name}")
    
    if 'image_size' in results:
        print(f"Image Size: {results['image_size']}")
    
    if 'prompt' in results:
        print(f"Prompt: {results['prompt']}")
    
    print(f"Calculation Time: {results.get('timestamp', 'N/A')}")
    print(f"Processing Time: {results.get('calculation_time', 0):.2f} seconds")
    print("-" * 60)
    
    if 'bagel_score_decon' in results and results['bagel_score_decon'] is not None:
        print(f"BagelScore_DeCon (Decoding Consistency Score): {results['bagel_score_decon']:.4f}")
    
    if 'bagel_score_langimgcon' in results and results['bagel_score_langimgcon'] is not None:
        print(f"BagelScore_LangImgCon (Language-Image Consistency Score): {results['bagel_score_langimgcon']:.4f}")
    
    print("-" * 60)
    
    if 'language_response' in results and results['language_response']:
        print("Generated Language Response:")
        print(results['language_response'])
        print("-" * 60)
    
    if 'error' in results:
        print(f"Error: {results['error']}")
        print("-" * 60)


def print_batch_summary(results: List[Dict[str, Any]]):
    """
    打印批量处理结果摘要
    
    Args:
        results: 结果列表
    """
    successful_results = [r for r in results if 'error' not in r]
    failed_results = [r for r in results if 'error' in r]
    
    print("\n" + "="*60)
    print("Batch Test Results Summary")
    print("="*60)
    print(f"Total Images: {len(results)}")
    print(f"Successfully Processed: {len(successful_results)}")
    print(f"Failed: {len(failed_results)}")
    
    if successful_results:
        total_time = sum(r.get('calculation_time', 0) for r in successful_results)
        print(f"Total Time: {total_time:.2f} seconds")
        print(f"Average Time: {total_time / len(successful_results):.2f} seconds per image")
        print("-" * 60)
        
        # 计算统计信息
        decon_scores = [r['bagel_score_decon'] for r in successful_results if r.get('bagel_score_decon') is not None]
        langimgcon_scores = [r['bagel_score_langimgcon'] for r in successful_results if r.get('bagel_score_langimgcon') is not None]
        
        if decon_scores:
            print(f"BagelScore_DeCon - Average: {np.mean(decon_scores):.4f}, Std: {np.std(decon_scores):.4f}")
            print(f"  Range: [{np.min(decon_scores):.4f}, {np.max(decon_scores):.4f}]")
        
        if langimgcon_scores:
            print(f"BagelScore_LangImgCon - Average: {np.mean(langimgcon_scores):.4f}, Std: {np.std(langimgcon_scores):.4f}")
            print(f"  Range: [{np.min(langimgcon_scores):.4f}, {np.max(langimgcon_scores):.4f}]")
    
    if failed_results:
        print("-" * 60)
        print("Failed Images:")
        for result in failed_results:
            print(f"  - {result.get('image_path', 'Unknown')}: {result.get('error', 'Unknown error')}")


def export_results_to_csv(results: List[Dict[str, Any]], output_path: str) -> None:
    """
    Export batch results to CSV format for analysis
    
    Args:
        results: List of calculation results
        output_path: CSV file path
    """
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            # Define CSV headers
            fieldnames = [
                'image_path', 'image_name', 'image_size_width', 'image_size_height',
                'bagel_score_decon', 'bagel_score_langimgcon', 
                'calculation_time', 'timestamp', 'prompt',
                'language_response_length', 'status', 'error'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                # Extract image name from path
                image_name = os.path.basename(result.get('image_path', ''))
                
                # Get image size
                image_size = result.get('image_size', [0, 0])
                width = image_size[0] if len(image_size) > 0 else 0
                height = image_size[1] if len(image_size) > 1 else 0
                
                # Calculate language response length
                lang_response = result.get('language_response', '')
                lang_length = len(lang_response) if lang_response else 0
                
                # Determine status
                status = 'success' if 'error' not in result else 'failed'
                
                # Create CSV row
                csv_row = {
                    'image_path': result.get('image_path', ''),
                    'image_name': image_name,
                    'image_size_width': width,
                    'image_size_height': height,
                    'bagel_score_decon': result.get('bagel_score_decon', ''),
                    'bagel_score_langimgcon': result.get('bagel_score_langimgcon', ''),
                    'calculation_time': result.get('calculation_time', ''),
                    'timestamp': result.get('timestamp', ''),
                    'prompt': result.get('prompt', ''),
                    'language_response_length': lang_length,
                    'status': status,
                    'error': result.get('error', '')
                }
                
                writer.writerow(csv_row)
        
        logger.info(f"CSV results exported to: {output_path}")
        print(f"CSV results exported to: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to export CSV: {e}")
        print(f"Failed to export CSV: {e}")


def export_results_to_json(results: List[Dict[str, Any]], output_path: str) -> None:
    """
    Export batch results to JSON format for detailed analysis
    
    Args:
        results: List of calculation results
        output_path: JSON file path
    """
    try:
        # Calculate summary statistics
        successful_results = [r for r in results if 'error' not in r]
        failed_results = [r for r in results if 'error' in r]
        
        decon_scores = [r['bagel_score_decon'] for r in successful_results if r.get('bagel_score_decon') is not None]
        langimgcon_scores = [r['bagel_score_langimgcon'] for r in successful_results if r.get('bagel_score_langimgcon') is not None]
        
        summary_stats = {
            'total_images': len(results),
            'successful_count': len(successful_results),
            'failed_count': len(failed_results),
            'bagel_score_decon_stats': {
                'count': len(decon_scores),
                'mean': float(np.mean(decon_scores)) if decon_scores else None,
                'std': float(np.std(decon_scores)) if decon_scores else None,
                'min': float(np.min(decon_scores)) if decon_scores else None,
                'max': float(np.max(decon_scores)) if decon_scores else None,
            },
            'bagel_score_langimgcon_stats': {
                'count': len(langimgcon_scores),
                'mean': float(np.mean(langimgcon_scores)) if langimgcon_scores else None,
                'std': float(np.std(langimgcon_scores)) if langimgcon_scores else None,
                'min': float(np.min(langimgcon_scores)) if langimgcon_scores else None,
                'max': float(np.max(langimgcon_scores)) if langimgcon_scores else None,
            },
            'processing_info': {
                'total_time': sum(r.get('calculation_time', 0) for r in successful_results),
                'average_time': np.mean([r.get('calculation_time', 0) for r in successful_results]) if successful_results else 0,
                'export_timestamp': datetime.now().isoformat()
            }
        }
        
        # Create comprehensive export data
        export_data = {
            'summary': summary_stats,
            'results': results,
            'metadata': {
                'export_version': '1.0',
                'bagel_model': 'BAGEL-7B-MoT',
                'calculation_types': ['BagelScore_DeCon', 'BagelScore_LangImgCon']
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(export_data, jsonfile, indent=2, ensure_ascii=False)
        
        logger.info(f"JSON results exported to: {output_path}")
        print(f"JSON results exported to: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to export JSON: {e}")
        print(f"Failed to export JSON: {e}")


def export_batch_results(results: List[Dict[str, Any]], output_dir: str) -> Dict[str, str]:
    """
    Export batch results to both CSV and JSON formats
    
    Args:
        results: List of calculation results
        output_dir: Output directory path
        
    Returns:
        Dictionary with export file paths
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Define export file paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(output_dir, f"batch_summary_{timestamp}.csv")
        json_path = os.path.join(output_dir, f"all_results_{timestamp}.json")
        
        # Export to both formats
        export_results_to_csv(results, csv_path)
        export_results_to_json(results, json_path)
        
        export_paths = {
            'csv': csv_path,
            'json': json_path
        }
        
        print(f"\n📊 Batch results exported:")
        print(f"  📋 CSV Summary: {csv_path}")
        print(f"  📄 JSON Details: {json_path}")
        
        return export_paths
        
    except Exception as e:
        logger.error(f"Failed to export batch results: {e}")
        print(f"Failed to export batch results: {e}")
        return {}
