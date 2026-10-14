<div align="center">

# PARROT: Heterogeneous PTM Fusion for Speech Emotion Recognition

## Synergizing Mamba and Attention-based SSL Pre-Trained Models via Parallel Branch Hadamard Optimal Transport

[![Paper](https://img.shields.io/badge/Paper-Interspeech%202025-blue)](https://www.isca-archive.org/interspeech_2025/phukan25e_interspeech.html)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-ee4c2c)](https://pytorch.org/)

</div>

---

## Paper Details

**Title**: PARROT: Synergizing Mamba and Attention-based SSL Pre-Trained Models via Parallel Branch Hadamard Optimal Transport for Speech Emotion Recognition

**Authors**: Orchid Chetia Phukan, Mohd Mujtaba Akhtar, Girish, Swarup Ranjan Behera, Jaya Sai Kiran Patibandla, Arun Balaji Buduru, Rajesh Sharma

**Conference**: Interspeech 2025, pp. 4468–4472

---

## Abstract

Mamba-based self-supervised learning (SSL) pre-trained models (PTMs) are emerging as alternatives to attention-based models for speech processing. Recent studies report that they perform comparably to or better than attention-based PTMs on speech emotion recognition (SER). In this work, we hypothesize that fusing heterogeneous pre-trained models—combining Mamba-based and attention-based PTMs—will outperform fusing attention-based PTMs alone.

To this end, we propose **PARROT**, a novel framework that employs parallel-branch fusion combining Optimal Transport and the Hadamard product. Through PARROT with heterogeneous PTM fusion, we achieve state-of-the-art results in comparison to individual PTMs, homogeneous attention-based fusion, and baseline fusion techniques.

---

## Overview

**PARROT** introduces a novel approach to Speech Emotion Recognition by fusing heterogeneous pre-trained models:
- **Mamba-based Models** - State-space models with linear complexity
- **Attention-based Models** - Transformer-based speech encoders

By combining these complementary architectures using **Optimal Transport** and **Hadamard Product**, PARROT captures both the efficiency of Mamba and the expressiveness of attention mechanisms, achieving **state-of-the-art** performance.

---

## Key Innovation: PARROT Framework

### What is Optimal Transport?

Optimal Transport measures the minimum cost to transform one probability distribution into another. It provides a mathematically principled way to compare and align distributions.

### Why Optimal Transport for Fusion?

1. **Distribution Alignment** - Optimally aligns feature distributions from different models
2. **Cost-Aware Fusion** - Minimizes the cost of transporting features between models
3. **Information Preservation** - Maintains geometric structure of embeddings
4. **Heterogeneous Integration** - Naturally handles models with different architectures

### Hadamard Product

Element-wise multiplication that preserves complementary information from both branches while enhancing shared characteristics.

---

## Pre-Trained Models

### Mamba-based Models (Novel)
- **Mamba** - State-space model with linear complexity
- **Mamba-Speech** - Speech-optimized variant
- Efficient temporal modeling with linear memory

### Attention-based Models (SOTA)
- **HuBERT** - Self-supervised speech representation
- **WavLM** - Speech-specific foundation model
- **XLSR** - Multilingual speech representation
- **Wav2Vec2** - Contrastive speech pretraining

---

## Quick Start

### Installation

```bash
git clone https://github.com/Helix-IIIT-Delhi/PARROT-SER.git
cd PARROT-SER
pip install -r requirements.txt
```

### Basic Usage

```python
from parrot import PARROTFusion
from models import load_ptm_models

# Load heterogeneous PTM models
mamba_model = load_ptm_models('mamba-speech')
attention_model = load_ptm_models('hubert')

# Initialize PARROT fusion
fusion = PARROTFusion(
    mamba_model=mamba_model,
    attention_model=attention_model,
    num_emotions=7
)

# Extract speech features
audio_features = extract_features(audio_path)

# Predict emotion
emotion_pred = fusion.predict(audio_features)
print(f"Predicted Emotion: {emotion_pred}")
```

---

## Methodology

### Optimal Transport Distance (Wasserstein)

For distributions P and Q:

```
W(P, Q) = inf E[||X - Y||]
          π ∈ Π(P,Q)
```

PARROT uses Optimal Transport to measure and minimize the divergence between Mamba and Attention model representations.

### Hadamard Product Fusion

For feature matrices F_mamba and F_attention:

```
F_fused = F_mamba ⊙ F_attention
```

This preserves complementary information while enhancing shared characteristics.

### Why Heterogeneous Fusion Works

**Mamba-based Models:**
- ✓ Linear complexity in sequence length
- ✓ Efficient long-range dependencies

**Attention-based Models:**
- ✓ Proven expressiveness
- ✓ Many pre-trained variants

**Fusion Result:**
- ✓ Efficiency of Mamba + Expressiveness of Attention
- ✓ State-of-the-art SER performance

---

## Results

PARROT achieves **state-of-the-art** speech emotion recognition:

| Method | Accuracy | Key Insight |
|--------|----------|------------|
| Individual Mamba PTM | Baseline | Efficient but limited |
| Individual Attention PTM | Baseline | Expressive but slower |
| Homogeneous Attention Fusion | Lower | Limited architectural diversity |
| **PARROT (Heterogeneous)** | **SOTA** | **Mamba + Attention + OT = Best** |

---

## Repository Structure

```
PARROT-SER/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
│
├── parrot/
│   ├── __init__.py
│   ├── fusion.py            # PARROT fusion framework
│   ├── transport.py         # Optimal Transport computation
│   ├── hadamard.py          # Hadamard product operations
│   ├── utils.py            # Utility functions
│   └── metrics.py          # Emotion recognition metrics
│
├── models/
│   ├── __init__.py
│   ├── loaders.py          # Load PTM models
│   ├── mamba.py            # Mamba model wrappers
│   ├── attention.py        # Attention model wrappers
│   └── pretrained.py       # Model definitions
│
├── configs/
│   ├── parrot_config.yaml
│   └── models_config.yaml
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
└── tests/
    └── test_fusion.py
```

---

## Key Features

🔄 **Heterogeneous Fusion** - Combines Mamba and Attention architectures  
📊 **Optimal Transport** - Mathematically principled distribution alignment  
⚡ **Parallel Branches** - Efficient dual-path processing  
🎯 **State-of-the-Art** - Best SER performance  
🧠 **Complementary** - Captures both efficiency and expressiveness  
🎤 **Emotion Recognition** - 7-class emotion classification  

---

## Citation

If you find this work useful, please consider citing us:

```bibtex
@inproceedings{phukan2025parrot,
  title={PARROT: Synergizing Mamba and Attention-based SSL Pre-Trained Models via Parallel Branch Hadamard Optimal Transport for Speech Emotion Recognition},
  author={Phukan, Orchid Chetia and Akhtar, Mohd Mujtaba and Girish and Behera, Swarup Ranjan and Patibandla, Jaya Sai Kiran and Buduru, Arun Balaji and Sharma, Rajesh},
  booktitle={Proceedings of Interspeech 2025},
  pages={4468--4472},
  year={2025},
  doi={10.21437/Interspeech.2025-2233}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">

**Advancing Speech Emotion Recognition through Heterogeneous Model Fusion and Optimal Transport**

</div>
