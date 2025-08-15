#!/usr/bin/env python3
"""
并行BAGEL相似性计算器 - 充分利用多GPU资源
"""

import os
import json
import time
import logging
import multiprocessing as mp
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
from PIL import Image
import torch

from .calculator import BagelSimilarityCalculator
from .config import BagelSimilarityConfig
from .utils import load_image, export_batch_results

logger = logging.getLogger(__name__)


class GPUWorker:
    """单GPU工作进程"""
    
    def __init__(self, gpu_id: int, config: BagelSimilarityConfig):
        self.gpu_id = gpu_id
        self.config = config
        self.calculator = None
        
    def initialize(self):
        """初始化GPU工作器"""
        try:
            # 设置GPU设备
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
            
            # 修改配置以使用单GPU
            single_gpu_config = self.config
            single_gpu_config.model.max_memory = {0: "22GiB", "cpu": "32GiB"}
            single_gpu_config.model.device_map = "auto"
            
            # 初始化计算器
            self.calculator = BagelSimilarityCalculator(single_gpu_config)
            logger.info(f"GPU {self.gpu_id} worker initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize GPU {self.gpu_id} worker: {e}")
            return False
    
    def process_image(self, image_path: str, prompt: str) -> Dict[str, Any]:
        """处理单张图像"""
        try:
            start_time = time.time()
            
            # 计算相似性分数
            results = self.calculator.calculate_all_scores(image_path, prompt)
            
            # 添加GPU信息和处理时间
            results['gpu_id'] = self.gpu_id
            results['calculation_time'] = time.time() - start_time
            
            logger.info(f"GPU {self.gpu_id} processed {os.path.basename(image_path)} in {results['calculation_time']:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"GPU {self.gpu_id} failed to process {image_path}: {e}")
            return {
                'image_path': image_path,
                'gpu_id': self.gpu_id,
                'error': str(e),
                'calculation_time': time.time() - start_time if 'start_time' in locals() else 0
            }


def gpu_worker_process(gpu_id: int, config_dict: Dict, task_queue: mp.Queue, result_queue: mp.Queue):
    """GPU工作进程函数"""
    try:
        # 重建配置对象
        config = BagelSimilarityConfig.from_dict(config_dict)
        
        # 创建并初始化工作器
        worker = GPUWorker(gpu_id, config)
        if not worker.initialize():
            result_queue.put(('error', f'Failed to initialize GPU {gpu_id}'))
            return
        
        result_queue.put(('ready', f'GPU {gpu_id} ready'))
        
        # 处理任务
        while True:
            try:
                task = task_queue.get(timeout=5)
                if task is None:  # 结束信号
                    break
                    
                image_path, prompt = task
                result = worker.process_image(image_path, prompt)
                result_queue.put(('result', result))
                
            except Exception as e:
                logger.error(f"GPU {gpu_id} worker error: {e}")
                result_queue.put(('error', f'GPU {gpu_id} error: {str(e)}'))
                
    except Exception as e:
        logger.error(f"GPU {gpu_id} process failed: {e}")
        result_queue.put(('error', f'GPU {gpu_id} process failed: {str(e)}'))


class ParallelBagelCalculator:
    """并行BAGEL相似性计算器"""
    
    def __init__(self, config: BagelSimilarityConfig):
        self.config = config
        self.gpu_count = torch.cuda.device_count()
        self.processes = []
        self.task_queue = None
        self.result_queue = None
        
        logger.info(f"Initializing parallel calculator with {self.gpu_count} GPUs")
        
    def _start_workers(self):
        """启动GPU工作进程"""
        try:
            # 创建队列
            self.task_queue = mp.Queue()
            self.result_queue = mp.Queue()
            
            # 启动工作进程
            for gpu_id in range(self.gpu_count):
                config_dict = self.config.to_dict()
                process = mp.Process(
                    target=gpu_worker_process,
                    args=(gpu_id, config_dict, self.task_queue, self.result_queue)
                )
                process.start()
                self.processes.append(process)
                logger.info(f"Started worker process for GPU {gpu_id}")
            
            # 等待所有GPU就绪
            ready_count = 0
            while ready_count < self.gpu_count:
                try:
                    msg_type, msg_data = self.result_queue.get(timeout=60)
                    if msg_type == 'ready':
                        ready_count += 1
                        logger.info(msg_data)
                    elif msg_type == 'error':
                        logger.error(msg_data)
                        raise RuntimeError(f"GPU initialization failed: {msg_data}")
                except Exception as e:
                    logger.error(f"Failed to start workers: {e}")
                    self.shutdown()
                    raise
                    
            logger.info(f"All {self.gpu_count} GPU workers are ready")
            
        except Exception as e:
            logger.error(f"Failed to start workers: {e}")
            self.shutdown()
            raise
    
    def process_batch_parallel(self, image_paths: List[str], prompt: str, 
                             checkpoint_interval: int = 100) -> List[Dict[str, Any]]:
        """并行处理批量图像"""
        try:
            # 启动工作进程
            self._start_workers()
            
            total_images = len(image_paths)
            results = []
            processed_count = 0
            
            logger.info(f"Starting parallel processing of {total_images} images")
            logger.info(f"Checkpoint interval: every {checkpoint_interval} images")
            
            # 分发任务
            for image_path in image_paths:
                self.task_queue.put((image_path, prompt))
            
            # 收集结果
            while processed_count < total_images:
                try:
                    msg_type, msg_data = self.result_queue.get(timeout=300)  # 5分钟超时
                    
                    if msg_type == 'result':
                        results.append(msg_data)
                        processed_count += 1
                        
                        # 打印进度
                        if processed_count % 10 == 0 or processed_count == total_images:
                            logger.info(f"Progress: {processed_count}/{total_images} images processed")
                        
                        # 定期存档
                        if processed_count % checkpoint_interval == 0:
                            self._save_checkpoint(results, processed_count)
                            
                    elif msg_type == 'error':
                        logger.error(f"Worker error: {msg_data}")
                        
                except Exception as e:
                    logger.error(f"Error collecting results: {e}")
                    break
            
            # 最终存档
            if results:
                self._save_checkpoint(results, processed_count, final=True)
            
            logger.info(f"Parallel processing completed: {processed_count}/{total_images} images")
            return results
            
        except Exception as e:
            logger.error(f"Parallel processing failed: {e}")
            raise
        finally:
            self.shutdown()
    
    def _save_checkpoint(self, results: List[Dict[str, Any]], count: int, final: bool = False):
        """保存检查点"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_type = "final" if final else "checkpoint"
            
            # 创建检查点目录
            checkpoint_dir = f"bs_cal/checkpoints_{timestamp}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # 保存结果
            checkpoint_file = os.path.join(checkpoint_dir, f"{checkpoint_type}_{count}_results.json")
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': timestamp,
                    'processed_count': count,
                    'checkpoint_type': checkpoint_type,
                    'results': results
                }, f, indent=2, ensure_ascii=False)
            
            # 导出分析文件
            if final or count % 500 == 0:  # 每500个或最终结果导出分析文件
                export_batch_results(results, checkpoint_dir)
            
            logger.info(f"💾 Checkpoint saved: {checkpoint_file} ({count} images)")
            print(f"💾 Checkpoint saved: {count} images processed")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def shutdown(self):
        """关闭并行计算器"""
        try:
            # 发送结束信号
            if self.task_queue:
                for _ in range(self.gpu_count):
                    self.task_queue.put(None)
            
            # 等待进程结束
            for process in self.processes:
                if process.is_alive():
                    process.join(timeout=10)
                    if process.is_alive():
                        process.terminate()
                        process.join()
            
            logger.info("All GPU workers shut down successfully")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        finally:
            self.processes = []
            self.task_queue = None
            self.result_queue = None


def test_parallel_calculator():
    """测试并行计算器"""
    from .config import BagelSimilarityConfig
    
    # 创建测试配置
    config = BagelSimilarityConfig()
    
    # 创建并行计算器
    parallel_calc = ParallelBagelCalculator(config)
    
    # 测试图像路径
    test_images = [
        "data_1000/classic_0001.jpg",
        "data_1000/classic_0002.jpg", 
        "data_1000/classic_0003.jpg"
    ]
    
    try:
        # 并行处理
        results = parallel_calc.process_batch_parallel(
            test_images, 
            "Please describe this image in detail",
            checkpoint_interval=2
        )
        
        print(f"✅ Parallel processing completed: {len(results)} results")
        for result in results:
            print(f"  GPU {result.get('gpu_id', 'unknown')}: {os.path.basename(result.get('image_path', 'unknown'))}")
            
    except Exception as e:
        print(f"❌ Parallel processing failed: {e}")
    finally:
        parallel_calc.shutdown()


if __name__ == "__main__":
    test_parallel_calculator()
