# Image Classification and Representation Analysis
A collection of small PyTorch projects exploring image classification and the behavior of learned representations under realistic input variation.

## Projects

### 1. CIFAR-10 Simple ANN
- **Dataset:** CIFAR-10 (10 classes, 32×32 RGB images)
- **Architecture:** 3-layer fully connected neural network
- **Performance:** ~55–60% training accuracy
- **Purpose:** Baseline model for understanding optimization, training dynamics, and the limitations of non-convolutional architectures on image data.

### 2. CIFAR-10 ResNet-50 (in progress)
- **Dataset:** CIFAR-10
- **Architecture:** ResNet-50 (convolutional encoder with linear classification head)
- **Purpose:** Learn structured image representations suitable for analyzing embedding stability under realistic perturbations such as illumination shifts, blur, noise, and partial occlusion.

## Planned Analysis
- Extraction of intermediate embeddings from trained models
- Measurement of representation stability using cosine similarity and nearest-neighbor consistency
- Evaluation of how small input perturbations affect downstream predictions and latent geometry

## Requirements
```
torch
torchvision
matplotlib
```

## Installation
```bash
pip install torch torchvision matplotlib
```

