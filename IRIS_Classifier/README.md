# Iris Species Classification with Neural Network

A simple 3-layer feedforward neural network for classifying iris flowers into three species using PyTorch.

## Dataset

**Iris Dataset:** 150 samples of iris flowers with 4 features each
- Features: sepal length, sepal width, petal length, petal width
- Classes: Setosa (0), Versicolor (1), Virginica (2)
- Split: 80% training (120 samples), 20% testing (30 samples)

## Architecture
```
Input (4 features)
    ↓
FC Layer (4 → 8) + ReLU
    ↓
FC Layer (8 → 8) + ReLU
    ↓
FC Layer (8 → 3)
    ↓
Output (3 class logits)
```

**Model Specifications:**
- 3 fully connected layers
- Hidden layers: 8 → 8 neurons
- Activation: ReLU
- Loss: Cross-Entropy Loss
- Optimizer: Adam (lr = 0.01)
- Epochs: 100

## Usage
```bash
python classifier.py
```

The script will:
1. Download the Iris dataset from the provided URL
2. Split data into training and test sets
3. Train for 100 epochs
4. Display training progress every 10 epochs
5. Evaluate on test set
6. Generate loss curve and prediction comparison plots

## Results

**Performance:**
- **Test Accuracy: 93.33% (28/30 correct)**
- Test Loss: 0.0468

### Training Loss

![Training Loss](training_loss00.png)

The loss decreases smoothly from ~1.15 to ~0.05 over 100 epochs, indicating successful learning with stable convergence.

### Prediction Results

![Actual vs Predicted](actual_vs_predicted.png)

The model correctly predicts 28 out of 30 test samples. Blue circles represent actual labels, and red X's represent predictions. Overlapping markers indicate correct predictions.

**Analysis:**
- 2 misclassifications out of 30 test samples
- Errors occur at boundary cases between similar species
- Strong performance demonstrates the model learned meaningful feature representations
- The Iris dataset's clear class separation makes it well-suited for simple neural networks

## Requirements
```bash
pip install torch pandas scikit-learn matplotlib
```

## Key Takeaways

- Simple feedforward networks perform well on small, well-separated datasets like Iris
- 93.33% test accuracy demonstrates good generalization despite the simple architecture
- Smooth training loss curve indicates appropriate learning rate and stable optimization
- Most classification errors occur between visually similar species (likely Versicolor vs Virginica)
