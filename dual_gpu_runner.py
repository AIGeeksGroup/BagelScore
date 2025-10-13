#!/usr/bin/env python3
"""
双GPU并行BagelScore计算器
将图片分配到两个GPU上并行处理，提升计算效率
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
from datetime import datetime

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
        
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
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
            print(f"Error: {result.stderr}")
            return None, processing_time, 0
            
    except Exception as e:
        print(f"❌ GPU {gpu_id}: Exception in batch {batch_id}: {e}")
        return None, 0, 0
    finally:
        # 清理临时目录
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            
    return None, 0, 0

def merge_results(result_dirs, output_dir):
    """合并多个GPU的计算结果"""
    print(f"🔗 Merging results from {len(result_dirs)} directories...")
    
    all_results = []
    total_stats = {
        'total_images': 0,
        'successful_count': 0,
        'failed_count': 0,
        'total_time': 0
    }
    
    # 收集所有结果
    for result_dir in result_dirs:
        if not result_dir or not os.path.exists(result_dir):
            continue
            
        # 查找JSON结果文件
        json_files = [f for f in os.listdir(result_dir) if f.startswith('all_results_') and f.endswith('.json')]
        if json_files:
            json_file = os.path.join(result_dir, json_files[0])
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'results' in data:
                        all_results.extend(data['results'])
                        
                    if 'summary' in data:
                        summary = data['summary']
                        total_stats['total_images'] += summary.get('total_images', 0)
                        total_stats['successful_count'] += summary.get('successful_count', 0)
                        total_stats['failed_count'] += summary.get('failed_count', 0)
                        total_stats['total_time'] += summary.get('processing_info', {}).get('total_time', 0)
                        
            except Exception as e:
                print(f"⚠️ Failed to read {json_file}: {e}")
    
    # 创建合并后的输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 导出合并后的结果
    if all_results:
        from bs_cal.utils import export_batch_results
        export_paths = export_batch_results(all_results, output_dir)
        
        # 创建汇总报告
        summary_report = f"""
# 双GPU并行处理汇总报告

## 处理统计
- 总图片数: {total_stats['total_images']}
- 成功处理: {total_stats['successful_count']}
- 失败数量: {total_stats['failed_count']}
- 总处理时间: {total_stats['total_time']:.1f}秒
- 平均每张: {total_stats['total_time']/max(total_stats['successful_count'], 1):.1f}秒

## 分数统计
"""
        
        if total_stats['successful_count'] > 0:
            decon_scores = [r.get('bagel_score_decon') for r in all_results if r.get('bagel_score_decon') is not None]
            langimg_scores = [r.get('bagel_score_langimgcon') for r in all_results if r.get('bagel_score_langimgcon') is not None]
            
            if decon_scores:
                import numpy as np
                summary_report += f"""
### BagelScore_DeCon
- 平均值: {np.mean(decon_scores):.4f}
- 标准差: {np.std(decon_scores):.4f}
- 范围: [{np.min(decon_scores):.4f}, {np.max(decon_scores):.4f}]
"""
            
            if langimg_scores:
                summary_report += f"""
### BagelScore_LangImgCon  
- 平均值: {np.mean(langimg_scores):.4f}
- 标准差: {np.std(langimg_scores):.4f}
- 范围: [{np.min(langimg_scores):.4f}, {np.max(langimg_scores):.4f}]
"""
        
        summary_report += f"""
## 输出文件
- CSV统计: {export_paths.get('csv', 'N/A')}
- JSON详情: {export_paths.get('json', 'N/A')}

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        # 保存汇总报告
        report_path = os.path.join(output_dir, 'dual_gpu_summary.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(summary_report)
        
        print(f"📊 合并完成! 结果保存到: {output_dir}")
        print(f"📋 汇总报告: {report_path}")
        return export_paths
    
    return None

def main():
    """主函数：双GPU并行处理"""
    print("🚀 启动双GPU并行BagelScore计算器")
    
    # 配置参数
    DATA_DIR = "data_1000"
    NUM_IMAGES = 100
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
            print(f"📈 最终结果:")
            print(f"  📊 CSV文件: {export_paths.get('csv')}")
            print(f"  📄 JSON文件: {export_paths.get('json')}")
            print(f"  📁 结果目录: {final_output_dir}")
    else:
        print("❌ 没有成功的计算结果")

if __name__ == "__main__":
    main()
