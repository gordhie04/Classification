import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

class SimpleNN(nn.Module):
    def __init__(self, in_features=4, hl1 = 8, hl2 = 8, out_features=3): ##out_features, 3 because we have 3 classes or outputs
        super().__init__()
        self.fc1 = nn.Linear(in_features, hl1)
        self.fc2 = nn.Linear(hl1, hl2)
        self.output = nn.Linear(hl2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x)) # x is now the output of layer 1 (e.g., [64, 8]) # We keep reusing the variable name 'x' for the transformed data
        x = F.relu(self.fc2(x))  # x is now the output of layer 2
        x = self.output(x) # x is now the final output

        return x

torch.manual_seed(4)
SimpleNN = SimpleNN()

url = "https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv"
df = pd.read_csv(url)
df['variety'] = df['variety'].map({'Setosa': 0, 'Versicolor': 1, 'Virginica': 2})
# print(df)

X = df.drop(columns=['variety'])
X = X.values
y = df['variety']
y = y.values


## Train Test Split
X_train, X_test, y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=4)
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

##Error measurement
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(SimpleNN.parameters(), lr=0.01)

## epochs
epoch = 100
losses = []
for i in range(epoch):
    ## Forward prediction
    y_pred_train = SimpleNN(X_train)
    ## Measure Loss
    loss = criterion(y_pred_train, y_train)
    ##Keep track of loss
    losses.append(loss.detach().item())

    ##print every 10 epoch
    if i % 10 == 0:
        print(f'epoch: {i}, loss: {loss}')

    ##Back propagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(range(epoch), losses)
plt.ylabel("loss/error")
plt.xlabel('Epoch')
plt.savefig('training_loss00.png')


##Evaluate Model & Test Data
SimpleNN.eval()
with torch.no_grad():   ##turn off back prop
    y_pred_test = SimpleNN(X_test) ##getting y predictions from test
    _, predicted_classes = torch.max(y_pred_test, 1)  # Get predicted class labels
    loss = criterion(y_pred_test, y_test)
    print(f'Test loss: {loss:.4f}')

correct = (predicted_classes == y_test).sum().item()
accuracy = 100 * correct / len(y_test)
print(f'We got {correct} correct out of {len(y_test)} ({accuracy:.2f}%)')

plt.figure(figsize=(10, 6))
plt.plot(y_test.numpy(), 'bo-', label='Actual', markersize=10, linewidth=2)
plt.plot(predicted_classes.numpy(), 'rx-', label='Predicted', markersize=10, linewidth=2)
plt.xlabel('Sample Index')
plt.ylabel('Class (0=Setosa, 1=Versicolor, 2=Virginica)')
plt.title('Actual vs Predicted Classes on Test Set')
plt.legend()
plt.grid(True)
plt.yticks([0, 1, 2], ['Setosa (0)', 'Versicolor (1)', 'Virginica (2)'])
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
plt.show()
plt.show()
