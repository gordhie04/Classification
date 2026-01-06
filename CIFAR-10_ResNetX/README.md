# ResNet-50 on CIFAR-10

A PyTorch implementation of ResNet-50 for image classification on the CIFAR-10 dataset, achieving **87.82% test accuracy** with GPU acceleration.

## Results

- **Final Test Accuracy**: 87.82%
- **Training Time**: 20 epochs on Tesla T4 GPU
- **Architecture**: ResNet-50 (3-4-6-3 configuration)

### Training Output

![Terminal Output](Terminal_Output_ResNet-CIFAR10_.png)

Training executed on Tesla T4 GPU with consistent convergence across all epochs.

### Training Metrics

![Training Metrics](CIFAR-10_ResNet-Output.png)

The model shows steady convergence with training loss decreasing from ~1.62 to ~0.28 and validation accuracy improving from ~52% to ~88% over 20 epochs.

### Sample Predictions

![Sample Predictions](Image_Prediction.png)

The model demonstrates strong performance across all 10 CIFAR-10 classes, with particularly high accuracy on vehicle categories.

## Architecture

Custom ResNet-50 architecture adapted for CIFAR-10's 32×32 images:

- Initial convolution (3→64 channels)
- Layer 0: 3 Residual Blocks (64 channels)
- Layer 1: 4 Residual Blocks (128 channels, stride=2)
- Layer 2: 6 Residual Blocks (256 channels, stride=2)
- Layer 3: 3 Residual Blocks (512 channels, stride=2)
- Adaptive Global Average Pooling
- Fully Connected (512→10 classes)

**Key Features:**
- Residual blocks with skip connections
- Batch normalization after each convolution
- No MaxPooling layer (preserves spatial information)

## Requirements
```bash
torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.5.0
numpy>=1.21.0
```

## Installation
```bash
git clone https://github.com/yourusername/resnet-cifar10.git
cd resnet-cifar10
pip install torch torchvision matplotlib numpy
python resnet_cifar10.py
```

## Training Configuration

**Hyperparameters:**
- Optimizer: SGD with momentum (0.9)
- Learning rate: 0.01
- Weight decay: 0.001
- Batch size: 64
- Epochs: 20

**Data Augmentation (Training only):**
- Random crop (32×32, padding=4)
- Random horizontal flip
- Normalization: mean/std = (0.5, 0.5, 0.5)

## Performance Timeline

| Epoch | Train Loss | Val Accuracy |
|-------|-----------|--------------|
| 1     | 1.6241    | 51.90%       |
| 5     | 0.6224    | 78.12%       |
| 10    | 0.4254    | 82.48%       |
| 15    | 0.3436    | 84.95%       |
| 20    | 0.2844    | 87.33%       |

**Final Test Accuracy: 87.82%**

## CIFAR-10 Classes

Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck

## Output Files

- `training_metrics.png`: Training loss and validation accuracy curves
- `sample_predictions.png`: 4×4 grid of test predictions with ground truth labels

## References

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
