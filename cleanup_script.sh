#!/bin/bash
# 移动测试CSV文件
mv flickr8k_*_test.csv delete/ 2>/dev/null
# 移动运行脚本（保留核心的）
mv run_flickr8k_*.sh delete/ 2>/dev/null
mv run_on_supercomputer.* delete/ 2>/dev/null
mv run_simple_scoring.sh run_precise_scoring.sh run_memory_fixed.sh run_gpu_specific.sh run_batch_processing.sh delete/ 2>/dev/null
# 移动临时文件和分析文件
mv analyze_*.py delete/ 2>/dev/null
mv normalization_comparison.csv delete/ 2>/dev/null
mv final_model_summary.md delete/ 2>/dev/null
# 移动空文件
find . -maxdepth 1 -type f -size 0 -exec mv {} delete/ \; 2>/dev/null
# 移动其他README
mv README_image_downloader.md README_BAGEL.md delete/ 2>/dev/null
echo "清理完成！"
