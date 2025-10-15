# <img src="https://github.com/AIGeeksGroup/BagelScore/blob/master/bagelscore_logo.png" alt="logo" width="50"/> BagelScore: Visual-Language Evaluation Made Easy

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)

This is the code repository for the paper:
> **BagelScore: Visual-Language Evaluation Made Easy**
>
> [Shuo Yin](https://www.linkedin.com/in/shuoyin/)\*, [Zeyu Zhang](https://steve-zeyu-zhang.github.io/)\*<sup>†</sup>, Huacan Wang, Qizhen Lan, Ronghao Chen, and Hao Tang<sup>#</sup>
>
> \*Equal contribution. <sup>†</sup>Project lead. <sup>#</sup>Corresponding author.
>
> [Paper](placeholder) 

## Citation

If you use any content of this repo for your work, please cite the following our paper:
```
placeholder
```

## 🌟 Overview

BagelScore is a reference-free evaluation metric that leverages the BAGEL multimodal model to assess:
- **Image-Text Matching**: Semantic alignment between images and captions
- **Image Editing Quality**: Quality of AI-generated image edits

Unlike traditional embedding-based metrics (e.g., CLIPScore), BagelScore uses **inference-based semantic judgment** to capture fine-grained semantic mismatches like negations and substitutions.

 <img src="https://github.com/AIGeeksGroup/BagelScore/blob/master/framework.png" alt="framework"/> 

### Key Features

✅ **Reference-Free**: No need for ground-truth images  
✅ **Semantic Understanding**: Captures complex semantic relationships  
✅ **Multi-Task**: Supports both matching and editing evaluation  
✅ **High Correlation**: Strong alignment with human judgments  

---

## 📦 Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU support)
- 80GB+ GPU memory (for BAGEL-7B model)

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_ORG/BAGELSCORE.git
cd BAGELSCORE

# Install dependencies
pip install -r requirements.txt

# Install flash-attention (required)
pip install flash_attn==2.5.8 --no-build-isolation
```

### Download BAGEL Model

```python
from huggingface_hub import snapshot_download

save_dir = "models/BAGEL-7B-MoT"
repo_id = "ByteDance-Seed/BAGEL-7B-MoT"

snapshot_download(
    cache_dir=save_dir + "/cache",
    local_dir=save_dir,
    repo_id=repo_id,
    local_dir_use_symlinks=False,
    resume_download=True,
    allow_patterns=["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt"],
)
```

---

## 🚀 Quick Start

### 1. BagelScore for Image-Text Matching

```python
from bagelscore import BagelScorer
from PIL import Image

# Initialize scorer
scorer = BagelScorer(
    model_path="./models/BAGEL-7B-MoT",
    device_id=0
)

# Load image and caption
image = Image.open("example.jpg")
caption = "A cat sitting on a couch"

# Calculate BagelScore
score, info = scorer.calculate_bagelscore(image, caption)
print(f"BagelScore: {score:.3f}")
```

### 2. EditScore for Image Editing Quality

```python
from edit_score_calculator import EditScoreCalculator
from inferencer import InterleaveInferencer

# Initialize components (see evaluate_editscore_metrics.py for full setup)
calculator = EditScoreCalculator()

# Run inference with editing
output = inferencer(
    image=original_image,
    text="Apply a cartoon style to the whole image.",
    think=True,
    return_edit_score_data=True
)

# Calculate EditScore metrics
scores = calculator.compute_base_metrics(
    original_vae_latent=output['edit_score_data']['original_vae_latent'],
    generated_latent=output['edit_score_data']['generated_latent'],
    input_text_emb=output['edit_score_data']['input_text_emb'],
    think_text_emb=output['edit_score_data']['think_text_emb']
)

print(f"Image RLS: {scores['image_rls']:.4f}")
print(f"Image Cosine Sim: {scores['image_cosine_sim']:.4f}")
print(f"Text Similarity: {scores['text_similarity']:.4f}")
```

### 3. Batch Evaluation

```bash
# Evaluate BagelScore on a dataset
python bagelscore.py \
    --model_path ./models/BAGEL-7B-MoT \
    --data_file dataset.json \
    --images_dir ./images \
    --output_file results/bagelscore_results.csv \
    --device_id 0

# Evaluate EditScore metrics
python evaluate_editscore_metrics.py \
    --mode batch \
    --model_path ./models/BAGEL-7B-MoT \
    --images_dir ./images \
    --results_dir ./results \
    --prompt "Apply a cartoon style to the whole image." \
    --limit 100
```

---

## 📊 Evaluation Metrics

### BagelScore

BagelScore uses a binary query approach:
1. Asks the model: "Are the IMAGE and TEXT describing the same content?"
2. Extracts logits for "Yes" tokens
3. Applies sigmoid function to get final score: `S(x,y) = σ(ℓ_yes)`

**Score Range**: [0, 1]  
- **1.0**: Perfect semantic match
- **0.0**: Complete mismatch

### EditScore Base Metrics

EditScore provides three fundamental metrics:

1. **image_rls** (Relative Latent Shift): Measures editing magnitude
   - `RLS = ||generated - original||₂ / ||original||₂`
   
2. **image_cosine_sim** (Cosine Similarity): Measures content preservation
   - Cosine similarity between original and edited image latents
   
3. **text_similarity**: Measures instruction consistency
   - Cosine similarity between input prompt and model's "think" text

---

## 📁 Project Structure

```
BAGELSCORE/
├── bagelscore.py                    # Main BagelScore implementation
├── edit_score_calculator.py         # EditScore base metrics calculator
├── inferencer.py                    # BAGEL model inference wrapper
├── evaluate_editscore_metrics.py    # EditScore evaluation script
├── batch_gpt_image_scoring.py       # GPT-4 scoring for comparison
├── modeling/                        # BAGEL model architecture
├── data/                            # Data loading utilities
├── eval/                            # Evaluation benchmarks
├── train/                           # Training scripts
├── requirements.txt                 # Python dependencies
└── LICENSE                          # Apache 2.0 License
```

---

## 🔬 Experiments & Benchmarks

### Datasets

We evaluate BagelScore on:
- **Flickr8k-Expert**: Expert-annotated image-caption pairs (1-4 scale)
- **Flickr8k-CF**: CrowdFlower-annotated pairs (0-1 scale)
- **[Edit-1K](https://huggingface.co/datasets/yinelon/Edit_1K)**: Image editing quality dataset

### Results


| Metric                | Flickr8K-Expert | Flickr8K-CF | Composite |
| --------------------- | --------------- | ----------- | --------- |
| **BAGELScore**        | **53.2**        | **38.0**    | **55.9**  |
| CLIPScore             | 51.2            | 34.4        | 53.8      |
| RefCLIPScore          | 53.0            | 36.4        | 55.4      |
| ViLBERTScore-F        | 50.1            | N/A         | 52.4      |
| SPICE                 | 44.9            | 24.4        | 40.3      |
| CIDEr                 | 43.9            | 24.6        | 37.7      |
| METEOR                | 41.8            | 22.2        | 38.9      |
| ROUGE-L               | 32.3            | 19.9        | 32.4      |
| BLEU-1                | 32.3            | N/A         | 31.3      |
| BLEU-4                | 30.8            | 16.9        | 30.6      |
| BERTScore (RoBERTa-F) | 39.2            | 22.8        | 30.1      |
| TIGEr                 | N/A             | N/A         | 45.4      |
| BERTScore++           | N/A             | N/A         | 44.9      |
| LEIC\*                | N/A             | 29.5        | N/A       |


 <img src="https://github.com/AIGeeksGroup/BagelScore/blob/master/comparison.png" alt="comparison"/> 


| Metric            | EditScore | Image RLS | Image Cosine | Text Sim. | Human Score |
| ----------------- | --------- | --------- | ------------ | --------- | ----------- |
| **EditScore**     | 1.00      | -0.78     | 0.78         | 0.05      | **0.14**    |
| Image RLS         | -0.78     | 1.00      | -0.74        | 0.00      | -0.12       |
| Image Cosine Sim. | 0.78      | -0.74     | 1.00         | 0.01      | 0.09        |
| Text Similarity   | 0.05      | 0.00      | 0.01         | 1.00      | 0.05        |
| **Human Score**   | **0.14**  | -0.12     | 0.09         | 0.05      | 1.00        |



| Metric          | Kendall Tau-b | Kendall Tau-c |
| --------------- | ------------- | ------------- |
| **Human Score** | 1.000         | 1.000         |
| **EditScore**   | **0.259**     | **0.253**     |
| GPT-based Score | 0.192         | 0.189         |


<p align="center">
  <img src="https://github.com/AIGeeksGroup/BagelScore/blob/master/editsocre_consistency.png" width="48%" alt="Rank Consistency between Edit Score and Human Score"/>
  <img src="https://github.com/AIGeeksGroup/BagelScore/blob/master/3d_visualization.png" width="48%" alt="3D Visualization of Edit Score with Image Cosine Similarity, Text Similarity, and Image RLS"/>
</p>
---

## 🛠️ Advanced Usage

### Multi-GPU Evaluation

```bash
python evaluate_editscore_metrics.py \
    --mode multi_gpu \
    --num_gpus 4 \
    --images_dir ./images \
    --results_dir ./results \
    --limit 1000
```

### Custom Prompts

```python
# Custom editing prompt
prompt = "Transform the image into a watercolor painting style"

output = inferencer(
    image=image,
    text=prompt,
    think=True,
    cfg_text_scale=4.0,
    cfg_img_scale=1.5,
    num_timesteps=50
)
```

### Memory Optimization

For limited GPU memory:
- Use batch processing with `--batch_size 1`
- Enable memory cleanup with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- Process data incrementally with `--resume_from` flag

---



## 📝 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **BAGEL Model**: Based on [ByteDance-Seed/BAGEL-7B-MoT](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT)
- **Datasets**: Flickr8k, SEED-Data-Edit
- **Inspiration**: CLIPScore, PAC-S, and other vision-language metrics

---

## 📁 Project Organization

### Core Files
This repository contains the essential components for BagelScore evaluation:
- **Core implementation files** in the root directory
- **Model architecture** in `modeling/`
- **Evaluation tools** in `eval/`
- **Training scripts** in `train/`

---

## 📧 Contact

For questions and feedback:
- **Issues**: [GitHub Issues](https://github.com/AIGeeksGroup/BAGELSCORE/issues)
- **Email**: yins25@tsinghua.mails.edu.cn
- **Paper**: [arXiv:XXXX.XXXXX](palceholder)


---

**⭐ Star us on GitHub if you find this project helpful!**
