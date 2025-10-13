# BagelScore: Visual-Language Evaluation Made Easy

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)

**Official implementation of BagelScore v1.0** - A novel visual-language evaluation metric for assessing image-text alignment and image editing quality.

📄 **Paper**: BAGELSCORE: VISUAL-LANGUAGE EVALUATION MADE EASY (ICLR 2025)

---

## 🌟 Overview

BagelScore is a reference-free evaluation metric that leverages the BAGEL multimodal model to assess:
- **Image-Text Matching**: Semantic alignment between images and captions
- **Image Editing Quality**: Quality of AI-generated image edits

Unlike traditional embedding-based metrics (e.g., CLIPScore), BagelScore uses **inference-based semantic judgment** to capture fine-grained semantic mismatches like negations and substitutions.

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
├── EVAL.md                          # Evaluation guide
├── TRAIN.md                         # Training guide
└── LICENSE                          # Apache 2.0 License
```

---

## 🔬 Experiments & Benchmarks

### Datasets

We evaluate BagelScore on:
- **Flickr8k-Expert**: Expert-annotated image-caption pairs (1-4 scale)
- **Flickr8k-CF**: CrowdFlower-annotated pairs (0-1 scale)
- **Edit-1K**: Image editing quality dataset

### Results

| Metric | Flickr8k-Expert | Flickr8k-CF | Edit-1K |
|--------|----------------|-------------|---------|
| **BagelScore** | **0.XXX** | **0.XXX** | **0.XXX** |
| CLIPScore | 0.XXX | 0.XXX | 0.XXX |
| PAC-S | 0.XXX | 0.XXX | 0.XXX |

*Correlation with human ratings (Kendall's τ)*

For detailed evaluation results, see [EVAL.md](EVAL.md).

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

## 📖 Citation

If you find BagelScore useful in your research, please cite:

```bibtex
@inproceedings{bagelscore2025,
  title={BagelScore: Visual-Language Evaluation Made Easy},
  author={Your Name and Others},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
pytest tests/

# Format code
black .
```

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

### Backup Files
**Note**: This is a simplified organization. Many experimental implementations, detailed calculation processes, and development iterations are preserved in the `delete/` folder for reference:

```bash
# View backup files
ls delete/

# Restore specific files if needed
cp delete/filename.py ./
```

The `delete/` folder contains:
- **147 backup files** including experimental versions
- **Alternative implementations** of scoring algorithms
- **Detailed analysis scripts** and calculation processes
- **Development iterations** and testing files
- **Flickr8k evaluation scripts** in various versions

If you cannot find specific functionality in the main codebase, please check the `delete/` folder as it contains comprehensive backup of all development work.

---

## 📧 Contact

For questions and feedback:
- **Issues**: [GitHub Issues](https://github.com/YOUR_ORG/BAGELSCORE/issues)
- **Email**: your.email@example.com
- **Paper**: [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)

---

## 🔗 Related Projects

- [BAGEL](https://github.com/ByteDance/BAGEL) - The base multimodal model
- [CLIPScore](https://github.com/jmhessel/clipscore) - CLIP-based evaluation
- [PAC-S](https://github.com/aimagelab/pacscore) - Perceptual similarity metric

---

**⭐ Star us on GitHub if you find this project helpful!**
