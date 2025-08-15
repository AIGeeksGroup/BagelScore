#!/usr/bin/env python3
"""
增强版双GPU并行BagelScore计算器
包含完整的数据保存和分析功能
"""

import os
import sys
import json
import csv
import subprocess
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
from datetime import datetime, timedelta
import numpy as np
import threading
import queue

def get_image_files(data_dir, num_images=100):
    """获取指定数量的图像文件"""
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = []
    
    for file_path in Path(data_dir).iterdir():
        if file_path.suffix.lower() in supported_formats:
            image_files.append(str(file_path))
            if len(image_files) >= num_images:
                break
    
    return image_files[:num_images]

def run_gpu_batch(gpu_id, image_batch, batch_id, output_base_dir):
    """在指定GPU上运行一批图像的计算"""
    print(f"🚀 GPU {gpu_id}: Starting batch {batch_id} with {len(image_batch)} images")
    
    # 创建临时数据目录
    temp_dir = f"temp_gpu{gpu_id}_batch{batch_id}"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # 复制图像到临时目录
        for i, img_path in enumerate(image_batch):
            src = img_path
            dst = os.path.join(temp_dir, f"img_{i:03d}_{Path(img_path).name}")
            shutil.copy2(src, dst)
        
        # 设置CUDA_VISIBLE_DEVICES环境变量
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # 运行BagelScore计算
        cmd = [
            sys.executable, '-m', 'bs_cal.cli',
            '--mode', 'batch',
            '--data-dir', temp_dir,
            '--batch-size', str(len(image_batch))
        ]
        
        print(f"💻 GPU {gpu_id}: Running command: {' '.join(cmd)}")
        start_time = time.time()
        
        # 使用实时输出而不是capture_output
        print(f"🔄 GPU {gpu_id}: Starting real-time processing...")
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd=os.getcwd()
        )
        
        # 实时输出处理进度
        output_lines = []
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                line = output.strip()
                output_lines.append(line)
                # 显示重要的进度信息
                if any(keyword in line for keyword in ['Processing image', 'BagelScore_DeCon:', 'BagelScore_LangImgCon:', 'Results saved', 'ERROR', 'Failed']):
                    print(f"📊 GPU {gpu_id}: {line}")
        
        result = process
        result.stdout_text = '\n'.join(output_lines)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ GPU {gpu_id}: Batch {batch_id} completed in {processing_time:.1f}s")
            
            # 查找生成的结果目录
            result_dirs = [d for d in os.listdir('.') if d.startswith('bs_cal/bagel_results_')]
            if result_dirs:
                latest_result_dir = max(result_dirs, key=lambda x: os.path.getctime(x))
                
                # 重命名结果目录
                new_result_dir = f"{output_base_dir}_gpu{gpu_id}_batch{batch_id}"
                if os.path.exists(latest_result_dir):
                    shutil.move(latest_result_dir, new_result_dir)
                    print(f"📁 GPU {gpu_id}: Results moved to {new_result_dir}")
                    return new_result_dir, processing_time, len(image_batch)
        else:
            print(f"❌ GPU {gpu_id}: Batch {batch_id} failed")
            print(f"Error output: {result.stdout_text[-500:] if hasattr(result, 'stdout_text') else 'No output captured'}")
            return None, processing_time, 0
            
    except Exception as e:
        print(f"❌ GPU {gpu_id}: Exception in batch {batch_id}: {e}")
        return None, 0, 0
    finally:
        # 清理临时目录
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            
    return None, 0, 0

def create_comprehensive_csv(results, output_path, gpu_stats, total_stats):
    """创建包含所有重要数据的综合CSV文件"""
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            # 定义详细的CSV字段
            fieldnames = [
                # 基本信息
                'image_id', 'image_name', 'image_path', 'image_size_width', 'image_size_height',
                # BagelScore结果
                'bagel_score_decon', 'bagel_score_langimgcon', 
                # 处理信息
                'calculation_time', 'timestamp', 'processed_by_gpu', 'gpu_batch_id',
                # 语言生成信息
                'prompt', 'language_response', 'language_response_length',
                # 状态信息
                'status', 'error',
                # 统计信息
                'decon_score_rank', 'langimgcon_score_rank',
                'decon_score_percentile', 'langimgcon_score_percentile'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # 计算排名和百分位数
            successful_results = [r for r in results if r.get('bagel_score_decon') is not None]
            decon_scores = [r['bagel_score_decon'] for r in successful_results]
            langimgcon_scores = [r.get('bagel_score_langimgcon') for r in successful_results if r.get('bagel_score_langimgcon') is not None]
            
            decon_ranks = np.argsort(np.argsort(decon_scores)[::-1]) + 1 if decon_scores else []
            langimgcon_ranks = np.argsort(np.argsort(langimgcon_scores)[::-1]) + 1 if langimgcon_scores else []
            
            # 写入每行数据
            for i, result in enumerate(results):
                image_name = os.path.basename(result.get('image_path', ''))
                image_size = result.get('image_size', [0, 0])
                
                # 计算排名和百分位数
                decon_rank = decon_ranks[i] if i < len(decon_ranks) else ''
                langimgcon_rank = langimgcon_ranks[i] if i < len(langimgcon_ranks) else ''
                
                decon_percentile = (1 - (decon_rank - 1) / len(decon_scores)) * 100 if decon_rank else ''
                langimgcon_percentile = (1 - (langimgcon_rank - 1) / len(langimgcon_scores)) * 100 if langimgcon_rank else ''
                
                csv_row = {
                    'image_id': i + 1,
                    'image_name': image_name,
                    'image_path': result.get('image_path', ''),
                    'image_size_width': image_size[0] if len(image_size) > 0 else 0,
                    'image_size_height': image_size[1] if len(image_size) > 1 else 0,
                    'bagel_score_decon': result.get('bagel_score_decon', ''),
                    'bagel_score_langimgcon': result.get('bagel_score_langimgcon', ''),
                    'calculation_time': result.get('calculation_time', ''),
                    'timestamp': result.get('timestamp', ''),
                    'processed_by_gpu': result.get('processed_by_gpu', ''),
                    'gpu_batch_id': result.get('gpu_batch_id', ''),
                    'prompt': result.get('prompt', ''),
                    'language_response': result.get('language_response', ''),
                    'language_response_length': len(result.get('language_response', '')),
                    'status': 'success' if 'error' not in result else 'failed',
                    'error': result.get('error', ''),
                    'decon_score_rank': decon_rank,
                    'langimgcon_score_rank': langimgcon_rank,
                    'decon_score_percentile': f"{decon_percentile:.1f}%" if decon_percentile else '',
                    'langimgcon_score_percentile': f"{langimgcon_percentile:.1f}%" if langimgcon_percentile else ''
                }
                
                writer.writerow(csv_row)
        
        print(f"✅ 综合CSV分析文件已生成: {output_path}")
        
    except Exception as e:
        print(f"❌ 生成综合CSV文件失败: {e}")

def create_comprehensive_json(results, output_path, gpu_stats, total_stats):
    """创建包含所有重要数据的综合JSON文件"""
    try:
        # 计算详细统计信息
        successful_results = [r for r in results if 'error' not in r]
        decon_scores = [r['bagel_score_decon'] for r in successful_results if r.get('bagel_score_decon') is not None]
        langimgcon_scores = [r['bagel_score_langimgcon'] for r in successful_results if r.get('bagel_score_langimgcon') is not None]
        
        # 创建综合分析数据
        comprehensive_data = {
            'metadata': {
                'analysis_version': '2.0',
                'generation_time': datetime.now().isoformat(),
                'bagel_model': 'BAGEL-7B-MoT',
                'processing_mode': 'dual_gpu_parallel',
                'calculation_types': ['BagelScore_DeCon', 'BagelScore_LangImgCon']
            },
            'summary_statistics': {
                'overall': total_stats,
                'by_gpu': gpu_stats,
                'bagel_score_decon': {
                    'count': len(decon_scores),
                    'mean': float(np.mean(decon_scores)) if decon_scores else None,
                    'median': float(np.median(decon_scores)) if decon_scores else None,
                    'std': float(np.std(decon_scores)) if decon_scores else None,
                    'min': float(np.min(decon_scores)) if decon_scores else None,
                    'max': float(np.max(decon_scores)) if decon_scores else None,
                    'q25': float(np.percentile(decon_scores, 25)) if decon_scores else None,
                    'q75': float(np.percentile(decon_scores, 75)) if decon_scores else None,
                },
                'bagel_score_langimgcon': {
                    'count': len(langimgcon_scores),
                    'mean': float(np.mean(langimgcon_scores)) if langimgcon_scores else None,
                    'median': float(np.median(langimgcon_scores)) if langimgcon_scores else None,
                    'std': float(np.std(langimgcon_scores)) if langimgcon_scores else None,
                    'min': float(np.min(langimgcon_scores)) if langimgcon_scores else None,
                    'max': float(np.max(langimgcon_scores)) if langimgcon_scores else None,
                    'q25': float(np.percentile(langimgcon_scores, 25)) if langimgcon_scores else None,
                    'q75': float(np.percentile(langimgcon_scores, 75)) if langimgcon_scores else None,
                },
                'performance_metrics': {
                    'total_processing_time': total_stats['total_time'],
                    'average_time_per_image': total_stats['total_time'] / max(total_stats['successful_count'], 1),
                    'throughput_images_per_second': total_stats['successful_count'] / max(total_stats['total_time'], 1),
                    'success_rate': total_stats['successful_count'] / max(total_stats['total_images'], 1) * 100
                }
            },
            'detailed_results': results
        }
        
        with open(output_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(comprehensive_data, jsonfile, indent=2, ensure_ascii=False)
        
        print(f"✅ 综合JSON分析文件已生成: {output_path}")
        
    except Exception as e:
        print(f"❌ 生成综合JSON文件失败: {e}")

def create_gpu_performance_analysis(gpu_stats, output_path):
    """创建GPU性能对比分析CSV"""
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'gpu_id', 'images_processed', 'successful_count', 'failed_count',
                'total_time_seconds', 'avg_time_per_image', 'throughput_images_per_sec',
                'success_rate_percent', 'efficiency_score'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for gpu_id, stats in gpu_stats.items():
                success_rate = (stats['success'] / max(stats['images'], 1)) * 100
                avg_time = stats['time'] / max(stats['success'], 1)
                throughput = stats['success'] / max(stats['time'], 1)
                efficiency = (stats['success'] / max(stats['time'], 1)) * (success_rate / 100)
                
                writer.writerow({
                    'gpu_id': gpu_id,
                    'images_processed': stats['images'],
                    'successful_count': stats['success'],
                    'failed_count': stats['failed'],
                    'total_time_seconds': f"{stats['time']:.2f}",
                    'avg_time_per_image': f"{avg_time:.2f}",
                    'throughput_images_per_sec': f"{throughput:.3f}",
                    'success_rate_percent': f"{success_rate:.1f}%",
                    'efficiency_score': f"{efficiency:.3f}"
                })
        
        print(f"✅ GPU性能分析文件已生成: {output_path}")
        
    except Exception as e:
        print(f"❌ 生成GPU性能分析文件失败: {e}")

def merge_results(result_dirs, output_dir):
    """合并多个GPU的计算结果并生成统一分析文件"""
    print(f"🔗 Merging results from {len(result_dirs)} directories...")
    
    all_results = []
    gpu_stats = {}  # 记录每个GPU的统计信息
    total_stats = {
        'total_images': 0,
        'successful_count': 0,
        'failed_count': 0,
        'total_time': 0
    }
    
    # 收集所有结果
    for i, result_dir in enumerate(result_dirs):
        if not result_dir or not os.path.exists(result_dir):
            continue
            
        gpu_id = f"GPU_{i}"
        gpu_stats[gpu_id] = {'images': 0, 'success': 0, 'failed': 0, 'time': 0}
            
        # 查找JSON结果文件
        json_files = [f for f in os.listdir(result_dir) if f.startswith('all_results_') and f.endswith('.json')]
        if json_files:
            json_file = os.path.join(result_dir, json_files[0])
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'results' in data:
                        # 为每个结果添加GPU标识
                        for result in data['results']:
                            result['processed_by_gpu'] = gpu_id
                            result['gpu_batch_id'] = i + 1
                        all_results.extend(data['results'])
                        
                    if 'summary' in data:
                        summary = data['summary']
                        gpu_images = summary.get('total_images', 0)
                        gpu_success = summary.get('successful_count', 0)
                        gpu_failed = summary.get('failed_count', 0)
                        gpu_time = summary.get('processing_info', {}).get('total_time', 0)
                        
                        gpu_stats[gpu_id] = {
                            'images': gpu_images,
                            'success': gpu_success,
                            'failed': gpu_failed,
                            'time': gpu_time
                        }
                        
                        total_stats['total_images'] += gpu_images
                        total_stats['successful_count'] += gpu_success
                        total_stats['failed_count'] += gpu_failed
                        total_stats['total_time'] += gpu_time
                        
            except Exception as e:
                print(f"⚠️ Failed to read {json_file}: {e}")
    
    # 创建合并后的输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    if not all_results:
        print("❌ No results to merge")
        return None
    
    # 生成详细的CSV分析文件
    csv_analysis_path = os.path.join(output_dir, 'comprehensive_analysis.csv')
    create_comprehensive_csv(all_results, csv_analysis_path, gpu_stats, total_stats)
    
    # 生成详细的JSON分析文件
    json_analysis_path = os.path.join(output_dir, 'comprehensive_analysis.json')
    create_comprehensive_json(all_results, json_analysis_path, gpu_stats, total_stats)
    
    # 生成GPU性能对比分析
    performance_analysis_path = os.path.join(output_dir, 'gpu_performance_analysis.csv')
    create_gpu_performance_analysis(gpu_stats, performance_analysis_path)
    
    # 使用原有的导出功能
    from bs_cal.utils import export_batch_results
    export_paths = export_batch_results(all_results, output_dir)
    
    print(f"📊 合并完成! 结果保存到: {output_dir}")
    print(f"📈 主要分析文件:")
    print(f"  🔍 综合分析CSV: {csv_analysis_path}")
    print(f"  📄 综合分析JSON: {json_analysis_path}")
    print(f"  ⚡ GPU性能分析: {performance_analysis_path}")
    
    return {
        'comprehensive_csv': csv_analysis_path,
        'comprehensive_json': json_analysis_path,
        'performance_analysis': performance_analysis_path,
        **export_paths
    }

def create_progress_monitor(output_dir, total_images):
    """创建进度监控文件"""
    os.makedirs(output_dir, exist_ok=True)
    progress_file = os.path.join(output_dir, 'real_time_progress.json')
    
    progress_data = {
        'start_time': datetime.now().isoformat(),
        'total_images': total_images,
        'completed_images': 0,
        'gpu_0_completed': 0,
        'gpu_1_completed': 0,
        'current_status': 'Initializing...',
        'last_update': datetime.now().isoformat(),
        'estimated_completion': None,
        'results_summary': []
    }
    
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress_data, f, indent=2, ensure_ascii=False)
    
    return progress_file

def update_progress_monitor(progress_file, gpu_id, status, image_name=None, scores=None):
    """更新进度监控文件"""
    try:
        with open(progress_file, 'r', encoding='utf-8') as f:
            progress_data = json.load(f)
        
        progress_data['last_update'] = datetime.now().isoformat()
        progress_data['current_status'] = status
        
        if gpu_id is not None:
            if gpu_id == 0:
                progress_data['gpu_0_completed'] += 1
            else:
                progress_data['gpu_1_completed'] += 1
            
            progress_data['completed_images'] = progress_data['gpu_0_completed'] + progress_data['gpu_1_completed']
            
            # 计算预计完成时间
            if progress_data['completed_images'] > 0:
                start_time = datetime.fromisoformat(progress_data['start_time'])
                elapsed = (datetime.now() - start_time).total_seconds()
                avg_time_per_image = elapsed / progress_data['completed_images']
                remaining_images = progress_data['total_images'] - progress_data['completed_images']
                estimated_remaining = remaining_images * avg_time_per_image
                estimated_completion = datetime.now() + timedelta(seconds=estimated_remaining)
                progress_data['estimated_completion'] = estimated_completion.isoformat()
        
        if image_name and scores:
            progress_data['results_summary'].append({
                'image_name': image_name,
                'gpu_id': gpu_id,
                'bagel_score_decon': scores.get('decon'),
                'bagel_score_langimgcon': scores.get('langimgcon'),
                'timestamp': datetime.now().isoformat()
            })
        
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, indent=2, ensure_ascii=False)
            
        # 打印进度信息
        completion_rate = (progress_data['completed_images'] / progress_data['total_images']) * 100
        print(f"📈 总进度: {progress_data['completed_images']}/{progress_data['total_images']} ({completion_rate:.1f}%) - GPU0: {progress_data['gpu_0_completed']}, GPU1: {progress_data['gpu_1_completed']}")
        
    except Exception as e:
        print(f"⚠️ 更新进度监控失败: {e}")

def main():
    """主函数：双GPU并行处理"""
    print("🚀 启动增强版双GPU并行BagelScore计算器")
    
    # 配置参数
    DATA_DIR = "data_1000"
    NUM_IMAGES = 50  # 修改为50张图片测试
    OUTPUT_BASE = f"dual_gpu_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 获取图像文件
    print(f"📁 从 {DATA_DIR} 获取 {NUM_IMAGES} 张图片...")
    image_files = get_image_files(DATA_DIR, NUM_IMAGES)
    
    if len(image_files) < NUM_IMAGES:
        print(f"⚠️ 只找到 {len(image_files)} 张图片，将处理所有可用图片")
        NUM_IMAGES = len(image_files)
    
    # 分配图片到两个GPU
    mid_point = NUM_IMAGES // 2
    gpu0_images = image_files[:mid_point]
    gpu1_images = image_files[mid_point:]
    
    print(f"🔀 图片分配:")
    print(f"  GPU 0: {len(gpu0_images)} 张图片")
    print(f"  GPU 1: {len(gpu1_images)} 张图片")
    
    # 创建进度监控
    progress_output_dir = f"progress_{OUTPUT_BASE}"
    progress_file = create_progress_monitor(progress_output_dir, NUM_IMAGES)
    print(f"📊 进度监控文件: {progress_file}")
    
    # 并行执行
    start_time = time.time()
    result_dirs = []
    processing_times = []
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        # 提交任务到两个GPU
        futures = []
        futures.append(executor.submit(run_gpu_batch, 0, gpu0_images, 1, OUTPUT_BASE))
        futures.append(executor.submit(run_gpu_batch, 1, gpu1_images, 2, OUTPUT_BASE))
        
        # 等待完成
        for future in as_completed(futures):
            result_dir, proc_time, img_count = future.result()
            if result_dir:
                result_dirs.append(result_dir)
            processing_times.append(proc_time)
    
    total_time = time.time() - start_time
    
    print(f"\n⏱️ 并行处理完成! 总耗时: {total_time:.1f}秒")
    print(f"📊 处理效率: {NUM_IMAGES/total_time:.2f} 张/秒")
    
    # 合并结果
    if result_dirs:
        final_output_dir = f"final_{OUTPUT_BASE}"
        export_paths = merge_results(result_dirs, final_output_dir)
        
        if export_paths:
            print(f"\n🎉 双GPU并行处理完成!")
            print(f"📈 最终分析文件:")
            print(f"  🔍 综合分析CSV: {export_paths.get('comprehensive_csv')}")
            print(f"  📄 综合分析JSON: {export_paths.get('comprehensive_json')}")
            print(f"  ⚡ GPU性能分析: {export_paths.get('performance_analysis')}")
            print(f"  📊 批量摘要CSV: {export_paths.get('csv')}")
            print(f"  📋 批量详情JSON: {export_paths.get('json')}")
            print(f"  📁 结果目录: {final_output_dir}")
            
            print(f"\n💡 分析建议:")
            print(f"  - 使用 comprehensive_analysis.csv 进行Excel数据透视表分析")
            print(f"  - 使用 comprehensive_analysis.json 进行编程分析")
            print(f"  - 查看 gpu_performance_analysis.csv 了解GPU性能对比")
    else:
        print("❌ 没有成功的计算结果")

if __name__ == "__main__":
    main()
