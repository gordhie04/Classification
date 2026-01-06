import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader
from torchvision.utils import make_grid
import torch.nn as nn
import numpy as np

# Check if GPU is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Verify GPU
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# Training transforms (with augmentation)
train_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5), inplace=True)
])

# Test transforms (NO augmentation)
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5), inplace=True)
])

# Datasets
train_dataset = torchvision.datasets.CIFAR10(root="./cifer", train=True, transform=train_transforms, download=True)
test_dataset = torchvision.datasets.CIFAR10(root="./cifer", train=False, transform=test_transforms, download=True)

val_ratio = 0.2
train_dataset, val_dataset = random_split(train_dataset, [int((1-val_ratio) * len(train_dataset)), int(val_ratio*len(train_dataset))])

# DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True)

class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channel

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResnetX(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.maxpool = nn.Identity()
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Hyperparameters
num_classes = 10
num_epochs = 20
batch_size = 64
learning_rate = 0.01

# Create model and move to GPU
model = ResnetX(ResidualBlock, [3, 4, 6, 3]).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.001, momentum=0.9)

# Initialize tracking lists
train_losses = []
val_accuracies = []
epochs_list = []

# Train the model
total_step = len(train_dataloader)
print(f"Starting training for {num_epochs} epochs...")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for i, (images, labels) in enumerate(train_dataloader):
        # Move data to GPU
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Print progress every 100 batches
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')
    
    avg_loss = running_loss / total_step
    train_losses.append(avg_loss)
    epochs_list.append(epoch + 1)
    
    print(f'Epoch [{epoch+1}/{num_epochs}] Complete, Average Loss: {avg_loss:.4f}')
    
    # Validation
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)
        print(f'Validation Accuracy: {val_accuracy:.2f}%\n')

print("Training complete!")

# Final test evaluation
print("Evaluating on test set...")
model.eval()
test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_accuracy = 100 * test_correct / test_total
print(f'Final Test Accuracy: {test_accuracy:.2f}%')

# Plot training metrics
plt.figure(figsize=(12, 4))

# Plot 1: Training Loss
plt.subplot(1, 2, 1)
plt.plot(epochs_list, train_losses, 'b-o', linewidth=2, markersize=6)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Plot 2: Validation Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs_list, val_accuracies, 'g-o', linewidth=2, markersize=6)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Validation Accuracy Over Epochs', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
plt.show()

# Sample Predictions Visualization
print("Generating sample predictions...")

# CIFAR10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# Get a batch of test images
dataiter = iter(test_dataloader)
images, labels = next(dataiter)
images = images.to(device)
labels = labels.to(device)

# Get predictions
model.eval()
with torch.no_grad():
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

# Move back to CPU and denormalize
images = images.cpu()
images = images * 0.5 + 0.5

# Plot 16 samples
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for idx, ax in enumerate(axes.flat):
    if idx < len(images):
        # Convert from CHW to HWC format
        img = images[idx].numpy().transpose(1, 2, 0)
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        
        pred_label = class_names[predicted[idx]]
        true_label = class_names[labels[idx]]
        
        # Color: green if correct, red if wrong
        color = 'green' if predicted[idx] == labels[idx] else 'red'
        ax.set_title(f'Pred: {pred_label}\nTrue: {true_label}', 
                     color=color, fontsize=10, fontweight='bold')
        ax.axis('off')

plt.suptitle('Sample Predictions (Green=Correct, Red=Wrong)', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('sample_predictions.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nAll visualizations saved!")
print(f"- training_metrics.png")
print(f"- sample_predictions.png")
