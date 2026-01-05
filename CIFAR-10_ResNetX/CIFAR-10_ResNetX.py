import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader
from torchvision.utils import make_grid
import torch.nn as nn

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
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=False)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=False)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=False)

def show_image(dl):
    for image, label in dl:
        image = image * 0.5 + 0.5
        fig, ax = plt.subplots(figsize=(10,10))
        ax.imshow(make_grid(image, 10).permute(1, 2, 0))
        break
show_image(train_dataloader)
plt.show()

class ResidualBlock(nn.Module):
    def __init__(self,  in_channel, out_channel, stride = 1, downsample = None):
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
    def __init__(self, block, layers, num_classes = 10):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.Identity()
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
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

num_classes = 10
num_epochs = 20
batch_size = 16
learning_rate = 0.01

model = ResnetX(ResidualBlock, [3, 4, 6, 3])

#Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.001, momentum = 0.9)


#Train the model
total_step = len(train_dataloader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_dataloader):
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

    # Validation
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Accuracy: {} %'.format(100 * correct / total))
