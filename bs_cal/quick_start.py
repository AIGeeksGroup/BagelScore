#!/usr/bin/env python3
"""
BAGEL相似性计算器快速开始脚本
最简单的使用方式
"""

import os
import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.append('.')

def quick_test():
    """快速测试函数"""
    try:
        from bs_cal import BagelSimilarityCalculator, BagelSimilarityConfig
        
        print("🚀 BAGEL相似性计算器快速开始")
        print("="*50)
        
        # 检查模型路径
        model_path = "Bagel/models/BAGEL-7B-MoT"
        if not os.path.exists(model_path):
            print(f"❌ 错误: 未找到BAGEL模型")
            print(f"   请确保模型已下载到: {model_path}")
            print("   下载地址: https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT")
            return False
        
        # 检查数据目录
        data_dir = "data_1000"
        if not os.path.exists(data_dir):
            print(f"❌ 错误: 未找到数据目录")
            print(f"   请确保数据目录存在: {data_dir}")
            return False
        
        # 获取测试图像
        image_files = [f for f in os.listdir(data_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print(f"❌ 错误: 在目录 {data_dir} 中未找到图像文件")
            return False
        
        # 创建配置
        print("📋 创建配置...")
        config = BagelSimilarityConfig()
        config.model.model_path = model_path
        config.data.data_dir = data_dir
        
        # 验证配置
        if not config.validate():
            print("❌ 配置验证失败")
            return False
        
        # 初始化计算器
        print("🔧 初始化BAGEL相似性计算器...")
        calculator = BagelSimilarityCalculator(config)
        
        # 选择测试图像
        test_image = image_files[0]
        test_image_path = os.path.join(data_dir, test_image)
        
        print(f"🖼️  测试图像: {test_image}")
        print(f"📝 提示词: {config.calculation.default_prompt}")
        print("⏳ 开始计算...")
        
        # 计算相似性分数
        results = calculator.calculate_all_scores(test_image_path)
        
        # 显示结果
        print("\n" + "="*50)
        print("📊 计算结果")
        print("="*50)
        print(f"图像: {test_image}")
        print(f"图像尺寸: {results['image_size']}")
        print(f"处理时间: {results['calculation_time']:.2f} 秒")
        print("-" * 50)
        print(f"BagelScore_DeCon (解码一致性分数): {results['bagel_score_decon']:.4f}")
        print(f"BagelScore_LangImgCon (语言图像一致性分数): {results['bagel_score_langimgcon']:.4f}")
        print("-" * 50)
        print("生成的语言描述:")
        print(results['language_response'])
        print("="*50)
        
        print("✅ 快速测试完成!")
        return True
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保已安装所有依赖包")
        return False
    except Exception as e:
        print(f"❌ 运行错误: {e}")
        return False


def quick_batch_test():
    """快速批量测试"""
    try:
        from bs_cal import BagelSimilarityCalculator, BagelSimilarityConfig
        
        print("🚀 BAGEL相似性计算器批量测试")
        print("="*50)
        
        # 检查路径
        model_path = "Bagel/models/BAGEL-7B-MoT"
        data_dir = "data_1000"
        
        if not os.path.exists(model_path):
            print(f"❌ 错误: 未找到BAGEL模型: {model_path}")
            return False
        
        if not os.path.exists(data_dir):
            print(f"❌ 错误: 未找到数据目录: {data_dir}")
            return False
        
        # 创建配置
        config = BagelSimilarityConfig()
        config.model.model_path = model_path
        config.data.data_dir = data_dir
        config.test.batch_size = 3  # 只处理3张图像
        
        if not config.validate():
            print("❌ 配置验证失败")
            return False
        
        # 初始化计算器
        print("🔧 初始化BAGEL相似性计算器...")
        calculator = BagelSimilarityCalculator(config)
        
        # 获取图像文件
        image_files = [f for f in os.listdir(data_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print(f"❌ 错误: 在目录 {data_dir} 中未找到图像文件")
            return False
        
        # 限制处理数量
        image_files = image_files[:config.test.batch_size]
        image_paths = [os.path.join(data_dir, f) for f in image_files]
        
        print(f"🖼️  将处理 {len(image_paths)} 张图像")
        print("⏳ 开始批量计算...")
        
        # 批量计算
        results = calculator.batch_calculate(image_paths)
        
        # 显示结果摘要
        successful_results = [r for r in results if 'error' not in r]
        failed_results = [r for r in results if 'error' in r]
        
        print("\n" + "="*50)
        print("📊 批量计算结果摘要")
        print("="*50)
        print(f"总图像数: {len(results)}")
        print(f"成功处理: {len(successful_results)}")
        print(f"处理失败: {len(failed_results)}")
        
        if successful_results:
            total_time = sum(r.get('calculation_time', 0) for r in successful_results)
            print(f"总处理时间: {total_time:.2f} 秒")
            print(f"平均处理时间: {total_time / len(successful_results):.2f} 秒/图像")
            
            # 计算统计信息
            decon_scores = [r['bagel_score_decon'] for r in successful_results if r.get('bagel_score_decon') is not None]
            langimgcon_scores = [r['bagel_score_langimgcon'] for r in successful_results if r.get('bagel_score_langimgcon') is not None]
            
            if decon_scores:
                print(f"DeCon分数 - 平均: {sum(decon_scores)/len(decon_scores):.4f}")
            if langimgcon_scores:
                print(f"LangImgCon分数 - 平均: {sum(langimgcon_scores)/len(langimgcon_scores):.4f}")
        
        if failed_results:
            print("\n❌ 失败的图像:")
            for result in failed_results:
                print(f"  - {result.get('image_path', 'Unknown')}: {result.get('error', 'Unknown error')}")
        
        print("="*50)
        print("✅ 批量测试完成!")
        return True
        
    except Exception as e:
        print(f"❌ 批量测试错误: {e}")
        return False


def main():
    """主函数"""
    print("🎯 BAGEL相似性计算器快速开始")
    print("="*60)
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        print("❌ 错误: 需要Python 3.8或更高版本")
        return
    
    print("✅ Python版本检查通过")
    
    # 检查CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA可用: {torch.cuda.get_device_name(0)}")
            print(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB")
        else:
            print("⚠️  CUDA不可用，将使用CPU模式（速度较慢）")
    except ImportError:
        print("⚠️  未安装PyTorch，请先安装: pip install torch")
        return
    
    print("\n选择测试模式:")
    print("1. 单张图像快速测试")
    print("2. 批量图像快速测试")
    print("3. 退出")
    
    while True:
        try:
            choice = input("\n请输入选择 (1-3): ").strip()
            
            if choice == '1':
                print("\n开始单张图像快速测试...")
                success = quick_test()
                break
            elif choice == '2':
                print("\n开始批量图像快速测试...")
                success = quick_batch_test()
                break
            elif choice == '3':
                print("👋 再见!")
                return
            else:
                print("❌ 无效选择，请输入1-3")
                
        except KeyboardInterrupt:
            print("\n👋 用户中断，再见!")
            return
        except Exception as e:
            print(f"❌ 输入错误: {e}")
    
    if success:
        print("\n🎉 测试成功完成!")
        print("\n💡 提示:")
        print("  - 使用 'python -m bs_cal.cli --help' 查看命令行选项")
        print("  - 使用 'python -m bs_cal.example' 查看更多示例")
        print("  - 查看 bs_cal/README.md 获取详细文档")
    else:
        print("\n❌ 测试失败，请检查错误信息")


if __name__ == "__main__":
    main()
