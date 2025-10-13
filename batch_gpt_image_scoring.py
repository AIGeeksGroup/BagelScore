#!/usr/bin/env python3
"""
批量 GPT 图像编辑质量评分脚本
基于 deal.ipynb 的 API 和思路，对 EditScore 结果中的图像对进行独立评分

Usage:
python batch_gpt_scoring.py --limit 5 --output results/gpt_scores.csv
"""

import os
import sys
import argparse
import csv
import json
import base64
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import os
import time
import re
from tqdm import tqdm
from openai import OpenAI

# 设置 OpenAI API
# 请设置环境变量 OPENAI_API_KEY 或在此处替换为您的API密钥
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here"))

def encode_image(image_path: str) -> str:
    """将本地图片转为 base64 编码字符串"""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def get_image_pairs_from_editscore(csv_path: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
    """从 EditScore 结果中提取图像对信息"""
    pairs = []
    
    try:
        df = pd.read_csv(csv_path)
        
        # 限制处理数量
        if limit:
            df = df.head(limit)
        
        for _, row in df.iterrows():
            # 从图像路径推断原图和编辑图路径
            image_path = row['image_path']  # 例如: images_2/0008.jpg
            image_name = Path(image_path).stem  # 0008
            
            # 原图路径：在 images_2/ 目录中
            original_path = image_path  # images_2/0008.jpg
            
            # 编辑图路径：在 results_2/ 目录中
            edited_path = f"results_2/{image_name}_edited.jpg"  # results_2/0008_edited.jpg
            
            # 检查文件是否存在
            if os.path.exists(original_path) and os.path.exists(edited_path):
                pairs.append({
                    'image_id': row['image_id'],
                    'original_path': original_path,
                    'edited_path': edited_path,
                    'prompt': row['prompt'],
                    'image_rls': row['image_rls'],
                    'image_cosine_sim': row['image_cosine_sim'],
                    'text_similarity': row['text_similarity']
                })
            else:
                print(f"Warning: Missing files for {image_name}")
                print(f"  Original: {original_path} - {'✓' if os.path.exists(original_path) else '✗'}")
                print(f"  Edited: {edited_path} - {'✓' if os.path.exists(edited_path) else '✗'}")
                
    except Exception as e:
        print(f"Error reading CSV file: {e}")
    
    return pairs

def gpt_score_image_editing(original_path: str, edited_path: str, prompt: str) -> Dict[str, Any]:
    """
    使用 GPT-4o-mini 对图像编辑质量进行评分
    
    Args:
        original_path: 原图路径
        edited_path: 编辑后图像路径
        prompt: 编辑提示词
        
    Returns:
        包含评分的字典
    """
    
    # 编码图像
    original_b64 = encode_image(original_path)
    edited_b64 = encode_image(edited_path)
    
    if not original_b64 or not edited_b64:
        return {
            'error': 'Failed to encode images',
            'scores': None
        }
    
    # 构建评分提示词
    scoring_prompt = f"""
You are an expert image editing evaluator. I will provide you with two images: the first is the original image, and the second is the edited version. The editing goal was: "{prompt}"

Please analyze both images and provide numerical scores for the editing quality. You must give specific scores for each dimension:

1. Editing Accuracy (0-20 points): How well does the edited image achieve the prompt requirements?
2. Visual Quality (0-20 points): Is the edited image clear, natural, and free of artifacts?
3. Content Preservation (0-20 points): Are the important elements and structure of the original image preserved?
4. Style Consistency (0-20 points): Is the editing style harmonious and consistent?
5. Overall Effect (0-20 points): How is the overall visual impact and appeal?

You MUST respond in exactly this format with numerical scores:
Editing Accuracy: [number]/20
Visual Quality: [number]/20  
Content Preservation: [number]/20
Style Consistency: [number]/20
Overall Effect: [number]/20
Total Score: [number]/100

Detailed Evaluation: [Your analysis of what works well and what could be improved]
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": scoring_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{original_b64}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{edited_b64}"}}
                    ]
                }
            ],
            max_tokens=1000,
            temperature=0.3
        )
        
        content = response.choices[0].message.content
        
        # 解析评分结果
        scores = parse_gpt_scores(content)
        
        return {
            'error': None,
            'raw_response': content,
            'scores': scores
        }
        
    except Exception as e:
        return {
            'error': f"API call failed: {e}",
            'scores': None
        }

def parse_gpt_scores(content: str) -> Dict[str, Any]:
    """解析 GPT 返回的评分结果"""
    scores = {
        'editing_accuracy': None,
        'visual_quality': None,
        'content_preservation': None,
        'style_consistency': None,
        'overall_effect': None,
        'total_score': None,
        'detailed_evaluation': None
    }
    
    lines = content.split('\n')
    detailed_eval = []
    
    for line in lines:
        line = line.strip()
        
        # 解析各项评分
        if 'Editing Accuracy' in line:
            scores['editing_accuracy'] = extract_score(line)
        elif 'Visual Quality' in line:
            scores['visual_quality'] = extract_score(line)
        elif 'Content Preservation' in line:
            scores['content_preservation'] = extract_score(line)
        elif 'Style Consistency' in line:
            scores['style_consistency'] = extract_score(line)
        elif 'Overall Effect' in line:
            scores['overall_effect'] = extract_score(line)
        elif 'Total Score' in line:
            scores['total_score'] = extract_score(line)
        elif 'Detailed Evaluation' in line:
            # 收集详细评价
            continue
        elif line and not any(keyword in line for keyword in ['Editing Accuracy', 'Visual Quality', 'Content Preservation', 'Style Consistency', 'Overall Effect', 'Total Score', 'Detailed Evaluation']):
            detailed_eval.append(line)
    
    scores['detailed_evaluation'] = ' '.join(detailed_eval)
    
    return scores

def extract_score(line: str) -> Optional[float]:
    """从文本中提取分数"""
    # 查找 X/20 或 X/100 格式的分数
    match = re.search(r'(\d+)/(\d+)', line)
    if match:
        numerator = int(match.group(1))
        denominator = int(match.group(2))
        return numerator / denominator if denominator > 0 else None
    
    # 查找单独的分数
    match = re.search(r'(\d+(?:\.\d+)?)', line)
    if match:
        return float(match.group(1))
    
    return None

def save_results(results: List[Dict[str, Any]], output_path: str):
    """保存评分结果到 CSV 文件"""
    
    # 准备 CSV 数据
    csv_data = []
    
    for result in results:
        # 从图像路径提取序号
        image_name = Path(result['edited_path']).stem  # 例如: 0001_0
        sequence_number = image_name.split('_')[0] if '_' in image_name else image_name
        
        row = {
            'sequence_number': sequence_number,
            'image_id': result['image_id'],
            'image_name': image_name,
            'original_path': result['original_path'],
            'edited_path': result['edited_path'],
            'prompt': result['prompt'],
            'image_rls': result['image_rls'],
            'image_cosine_sim': result['image_cosine_sim'], 
            'text_similarity': result['text_similarity'],
            'gpt_error': result['gpt_result']['error'],
            'gpt_editing_accuracy': result['gpt_result']['scores']['editing_accuracy'] if result['gpt_result']['scores'] else None,
            'gpt_visual_quality': result['gpt_result']['scores']['visual_quality'] if result['gpt_result']['scores'] else None,
            'gpt_content_preservation': result['gpt_result']['scores']['content_preservation'] if result['gpt_result']['scores'] else None,
            'gpt_style_consistency': result['gpt_result']['scores']['style_consistency'] if result['gpt_result']['scores'] else None,
            'gpt_overall_effect': result['gpt_result']['scores']['overall_effect'] if result['gpt_result']['scores'] else None,
            'gpt_total_score': result['gpt_result']['scores']['total_score'] if result['gpt_result']['scores'] else None,
            'gpt_detailed_evaluation': result['gpt_result']['scores']['detailed_evaluation'] if result['gpt_result']['scores'] else None,
            'timestamp': result['timestamp']
        }
        csv_data.append(row)
    
    # 按序号排序
    df = pd.DataFrame(csv_data)
    df = df.sort_values('sequence_number')
    
    # 保存到 CSV
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="批量 GPT-4o-mini 图像编辑质量评分")
    parser.add_argument("--editscore_csv", type=str, default="results_2/editscore_results.csv",
                       help="EditScore 结果 CSV 文件路径")
    parser.add_argument("--limit", type=int, default=1000,
                       help="限制处理的图像对数量")
    parser.add_argument("--output", type=str, default="gpt_scores_results_2.csv",
                       help="输出 CSV 文件路径")
    parser.add_argument("--delay", type=float, default=0.5,
                       help="API 调用间隔（秒）")
    
    args = parser.parse_args()
    
    # 创建输出目录（如果需要）
    output_dir = Path(args.output).parent
    if output_dir != Path('.'):
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print("开始批量 GPT 图像编辑质量评分...")
    print(f"EditScore CSV: {args.editscore_csv}")
    print(f"处理限制: {args.limit}")
    print(f"输出文件: {args.output}")
    print(f"API 调用间隔: {args.delay}秒")
    print("-" * 50)
    
    # 获取图像对
    image_pairs = get_image_pairs_from_editscore(args.editscore_csv, args.limit)
    
    if not image_pairs:
        print("未找到有效的图像对，请检查文件路径和格式")
        return
    
    print(f"找到 {len(image_pairs)} 个图像对")
    
    # 处理每个图像对
    results = []
    
    try:
        for i, pair in enumerate(tqdm(image_pairs, desc="处理图像对")):
            print(f"\n处理 {i+1}/{len(image_pairs)}: {pair['image_id']}")
            print(f"原图: {pair['original_path']}")
            print(f"编辑图: {pair['edited_path']}")
            print(f"提示词: {pair['prompt']}")
            print(f"Image RLS: {pair['image_rls']:.4f}")
            print(f"Image Cosine Sim: {pair['image_cosine_sim']:.4f}")
            print(f"Text Similarity: {pair['text_similarity']:.4f}")
            
            # GPT 评分
            gpt_result = gpt_score_image_editing(
                pair['original_path'],
                pair['edited_path'],
                pair['prompt']
            )
            
            result = {
                'image_id': pair['image_id'],
                'original_path': pair['original_path'],
                'edited_path': pair['edited_path'],
                'prompt': pair['prompt'],
                'image_rls': pair['image_rls'],
                'image_cosine_sim': pair['image_cosine_sim'],
                'text_similarity': pair['text_similarity'],
                'gpt_result': gpt_result,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            results.append(result)
            
            # 显示结果
            if gpt_result['error']:
                print(f"错误: {gpt_result['error']}")
            else:
                scores = gpt_result['scores']
                print(f"GPT 评分:")
                print(f"  编辑准确性: {scores['editing_accuracy']}")
                print(f"  视觉质量: {scores['visual_quality']}")
                print(f"  内容保持: {scores['content_preservation']}")
                print(f"  风格一致性: {scores['style_consistency']}")
                print(f"  整体效果: {scores['overall_effect']}")
                print(f"  总分: {scores['total_score']}")
            
            # 每处理10个图像对保存一次中间结果
            if (i + 1) % 10 == 0:
                temp_output = f"{args.output}.temp"
                save_results(results, temp_output)
                print(f"中间结果已保存到 {temp_output}")
            
            # API 调用间隔
            if i < len(image_pairs) - 1:
                time.sleep(args.delay)
                
    except KeyboardInterrupt:
        print(f"\n用户中断处理，已处理 {len(results)} 个图像对")
        print("正在保存已处理的结果...")
    
    # 保存结果
    save_results(results, args.output)
    
    # 统计信息
    successful_results = [r for r in results if not r['gpt_result']['error']]
    print(f"\n处理完成!")
    print(f"总图像对: {len(results)}")
    print(f"成功评分: {len(successful_results)}")
    print(f"失败数量: {len(results) - len(successful_results)}")
    
    if successful_results:
        total_scores = [r['gpt_result']['scores']['total_score'] for r in successful_results if r['gpt_result']['scores']['total_score']]
        if total_scores:
            avg_score = sum(total_scores) / len(total_scores)
            print(f"平均总分: {avg_score:.2f}")
            
            # 显示 EditScore vs GPT Score 的对比
            print(f"\nEditScore vs GPT Score 对比:")
            for r in successful_results:
                if r['gpt_result']['scores']['total_score']:
                    print(f"  {r['image_id']}: RLS={r['image_rls']:.4f}, Cosine={r['image_cosine_sim']:.4f}, Text={r['text_similarity']:.4f}, GPT={r['gpt_result']['scores']['total_score']:.2f}")

if __name__ == "__main__":
    main()
