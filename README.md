# Image and Tabular Classification Projects

A collection of PyTorch projects exploring classification across different data modalities and architectures, from simple fully-connected networks to deep convolutional models.

## Projects

### 1. Iris Simple ANN
**Dataset:** Iris flower dataset (150 samples, 4 features, 3 species classes)

**Architecture:** 2-layer fully connected neural network (4 → 8 → 8 → 3)

**Performance:** ~96.67% test accuracy

**Purpose:** Foundational implementation demonstrating neural network basics on tabular data, including forward/backward propagation, loss tracking, and prediction visualization.

---

### 2. CIFAR-10 Simple ANN
**Dataset:** CIFAR-10 (10 classes, 32×32 RGB images)

**Architecture:** 3-layer fully connected neural network

**Performance:** ~55–60% training accuracy

**Purpose:** Baseline model for understanding optimization, training dynamics, and the limitations of non-convolutional architectures on image data.

---

### 3. CIFAR-10 ResNet-50
**Dataset:** CIFAR-10 (10 classes, 32×32 RGB images)

**Architecture:** ResNet-50 adapted for CIFAR-10 (convolutional encoder with residual connections + linear classification head)

**Performance:** ~85–88% validation accuracy (20 epochs with data augmentation)

**Purpose:** Learn structured image representations through deep residual networks, with planned analysis of embedding stability under realistic perturbations (illumination shifts, blur, noise, partial occlusion).

---

## Planned Analysis

- Extraction of intermediate embeddings from trained models
- Measurement of representation stability using cosine similarity and nearest-neighbor consistency
- Evaluation of how small input perturbations affect downstream predictions and latent geometry

## Requirements
```
torch
torchvision
matplotlib
pandas
scikit-learn
```

## Installation
```bash
pip install torch torchvision matplotlib pandas scikit-learn
```
