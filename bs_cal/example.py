#!/usr/bin/env python3
"""
BAGEL相似性计算器使用示例
"""

import os
import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.append('.')

from bs_cal import BagelSimilarityCalculator, BagelSimilarityConfig
from bs_cal.utils import print_results_summary, print_batch_summary


def example_single_image():
    """单张图像计算示例"""
    print("="*60)
    print("单张图像相似性计算示例")
    print("="*60)
    
    # 创建配置
    config = BagelSimilarityConfig()
    config.model.model_path = "Bagel/models/BAGEL-7B-MoT"
    config.data.data_dir = "data_1000"
    
    # 验证配置
    if not config.validate():
        print("配置验证失败，请检查模型路径和数据目录")
        return
    
    try:
        # 初始化计算器
        print("初始化BAGEL相似性计算器...")
        calculator = BagelSimilarityCalculator(config)
        
        # 获取测试图像
        image_files = [f for f in os.listdir(config.data.data_dir) 
                      if any(f.lower().endswith(ext) for ext in config.data.supported_formats)]
        
        if not image_files:
            print(f"在目录 {config.data.data_dir} 中未找到图像文件")
            return
        
        # 选择第一张图像进行测试
        test_image_path = os.path.join(config.data.data_dir, image_files[0])
        print(f"测试图像: {test_image_path}")
        
        # 计算相似性分数
        prompt = "请详细描述这张图片的内容、风格和特征"
        results = calculator.calculate_all_scores(test_image_path, prompt)
        
        # 打印结果
        print_results_summary(results, image_files[0])
        
    except Exception as e:
        print(f"计算过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


def example_batch_processing():
    """批量处理示例"""
    print("="*60)
    print("批量图像相似性计算示例")
    print("="*60)
    
    # 创建配置
    config = BagelSimilarityConfig()
    config.model.model_path = "Bagel/models/BAGEL-7B-MoT"
    config.data.data_dir = "data_1000"
    config.test.batch_size = 3  # 只处理3张图像
    
    # 验证配置
    if not config.validate():
        print("配置验证失败，请检查模型路径和数据目录")
        return
    
    try:
        # 初始化计算器
        print("初始化BAGEL相似性计算器...")
        calculator = BagelSimilarityCalculator(config)
        
        # 获取图像文件列表
        image_files = [f for f in os.listdir(config.data.data_dir) 
                      if any(f.lower().endswith(ext) for ext in config.data.supported_formats)]
        
        if not image_files:
            print(f"在目录 {config.data.data_dir} 中未找到图像文件")
            return
        
        # 限制处理数量
        image_files = image_files[:config.test.batch_size]
        image_paths = [os.path.join(config.data.data_dir, f) for f in image_files]
        
        print(f"将处理 {len(image_paths)} 张图像")
        
        # 批量计算
        prompt = "请详细描述这张图片的内容、风格和特征"
        results = calculator.batch_calculate(image_paths, prompt)
        
        # 打印结果摘要
        print_batch_summary(results)
        
    except Exception as e:
        print(f"批量处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


def example_custom_config():
    """自定义配置示例"""
    print("="*60)
    print("自定义配置示例")
    print("="*60)
    
    # 创建自定义配置
    config = BagelSimilarityConfig()
    
    # 模型配置
    config.model.model_path = "Bagel/models/BAGEL-7B-MoT"
    config.model.max_mem_per_gpu = "40GiB"
    
    # 数据配置
    config.data.data_dir = "data_1000"
    config.data.supported_formats = [".jpg", ".jpeg", ".png"]
    
    # 计算配置
    config.calculation.default_prompt = "请分析这张图片的视觉特征和内容"
    config.calculation.text_temperature = 0.8
    config.calculation.max_text_length = 150
    
    # 输出配置
    config.output.output_dir_prefix = "custom_results"
    config.output.save_json = True
    config.output.save_summary = True
    
    # 日志配置
    config.logging.level = "INFO"
    
    # 验证配置
    if not config.validate():
        print("配置验证失败")
        return
    
    try:
        # 初始化计算器
        print("使用自定义配置初始化计算器...")
        calculator = BagelSimilarityCalculator(config)
        
        # 获取测试图像
        image_files = [f for f in os.listdir(config.data.data_dir) 
                      if any(f.lower().endswith(ext) for ext in config.data.supported_formats)]
        
        if not image_files:
            print(f"在目录 {config.data.data_dir} 中未找到图像文件")
            return
        
        # 测试单张图像
        test_image_path = os.path.join(config.data.data_dir, image_files[0])
        print(f"测试图像: {test_image_path}")
        
        results = calculator.calculate_all_scores(test_image_path)
        print_results_summary(results, image_files[0])
        
    except Exception as e:
        print(f"自定义配置示例中出现错误: {e}")
        import traceback
        traceback.print_exc()


def example_environment_variables():
    """环境变量配置示例"""
    print("="*60)
    print("环境变量配置示例")
    print("="*60)
    
    # 设置环境变量
    os.environ['BAGEL_MODEL_PATH'] = "Bagel/models/BAGEL-7B-MoT"
    os.environ['BAGEL_DATA_DIR'] = "data_1000"
    os.environ['BAGEL_GPU_MEMORY'] = "40GiB"
    os.environ['BAGEL_LOG_LEVEL'] = "INFO"
    
    try:
        # 从环境变量创建配置
        from bs_cal.config import create_config_from_env
        config = create_config_from_env()
        
        # 验证配置
        if not config.validate():
            print("配置验证失败")
            return
        
        print("从环境变量成功创建配置:")
        print(f"  模型路径: {config.model.model_path}")
        print(f"  数据目录: {config.data.data_dir}")
        print(f"  GPU内存: {config.model.max_mem_per_gpu}")
        print(f"  日志级别: {config.logging.level}")
        
        # 初始化计算器
        print("\n初始化BAGEL相似性计算器...")
        calculator = BagelSimilarityCalculator(config)
        
        # 获取测试图像
        image_files = [f for f in os.listdir(config.data.data_dir) 
                      if any(f.lower().endswith(ext) for ext in config.data.supported_formats)]
        
        if not image_files:
            print(f"在目录 {config.data.data_dir} 中未找到图像文件")
            return
        
        # 测试单张图像
        test_image_path = os.path.join(config.data.data_dir, image_files[0])
        print(f"测试图像: {test_image_path}")
        
        results = calculator.calculate_all_scores(test_image_path)
        print_results_summary(results, image_files[0])
        
    except Exception as e:
        print(f"环境变量配置示例中出现错误: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    print("BAGEL相似性计算器使用示例")
    print("="*60)
    
    # 检查必要的目录和文件
    if not os.path.exists("Bagel/models/BAGEL-7B-MoT"):
        print("错误: 未找到BAGEL模型，请确保模型已下载到正确位置")
        print("模型路径: Bagel/models/BAGEL-7B-MoT")
        return
    
    if not os.path.exists("data_1000"):
        print("错误: 未找到数据目录，请确保data_1000目录存在")
        return
    
    # 运行示例
    examples = [
        ("单张图像计算", example_single_image),
        ("批量处理", example_batch_processing),
        ("自定义配置", example_custom_config),
        ("环境变量配置", example_environment_variables),
    ]
    
    for i, (name, func) in enumerate(examples, 1):
        print(f"\n{i}. {name}")
        try:
            func()
        except KeyboardInterrupt:
            print("\n用户中断执行")
            break
        except Exception as e:
            print(f"示例 {name} 执行失败: {e}")
        
        if i < len(examples):
            input("\n按回车键继续下一个示例...")


if __name__ == "__main__":
    main()
