import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

dataset = torchvision.datasets.CIFAR10(root = "./cifer", train = True, transform = transform, download = True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle = True)

class SimpleANN(nn.Module):
    def __init__(self):
        super(SimpleANN, self).__init__()
        self.fc1 = nn.Linear(3*32*32, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1,3*32*32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleANN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

train_losses = []
train_accuracies = []
epoch = 50
for each in range(epoch):
    running_loss = 0
    correct = 0
    total = 0
    epoch_loss = 0

    for i, data in enumerate(dataloader,0):
        inputs, labels = data
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        epoch_loss += loss.item()

        ##Calculate the accuracy
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if i % 100 == 99:
            print(f' epoch {each + 1}, Mini_batch {i + 1}, loss: {running_loss / 100:.3f}, Accuracy: {100 * correct / total: .2f}%')
            running_loss = 0.0

    avg_loss = epoch_loss / len(dataloader)
    accuracy = 100 * correct / total
    train_losses.append(avg_loss)
    train_accuracies.append(accuracy)
print('Finished Training')


plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(range(1, epoch + 1), train_losses, 'b-', label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Time')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, epoch + 1), train_accuracies, 'r-', label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training Accuracy over Time')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
