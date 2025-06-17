
```markdown
# Fashion-MNIST CNN Classifier

## Overview
This project implements a **Convolutional Neural Network (CNN)** for classifying images in the **Fashion-MNIST** dataset using **PyTorch**. The dataset consists of **10 fashion categories**, and the goal is to train a deep learning model for accurate classification.

## Dataset
The **Fashion-MNIST** dataset is a collection of **grayscale images (28x28 pixels)** representing different clothing items. The 10 classes include:
- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

## Setup Instructions
### Install Dependencies
Ensure you have the required libraries installed:

```bash
pip install torch torchvision matplotlib
```

### Load the Dataset
The dataset is automatically downloaded through torchvision. The transformations applied include **resizing** and **normalization**:

```python
import torchvision.datasets as dsets
import torchvision.transforms as transforms

# Define transformations
IMAGE_SIZE = 16
composed = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

# Create dataset objects
dataset_train = dsets.FashionMNIST(root='.fashion/data', train=True, transform=composed, download=True)
dataset_val = dsets.FashionMNIST(root='.fashion/data', train=False, transform=composed, download=True)
```

## Model Architecture
### CNN Structure:
- **2 Convolutional Layers:** Extract visual features from images.
- **Batch Normalization & Max Pooling:** Improve model performance.
- **Fully Connected Layer:** Final classification step.
- **Activation Function:** ReLU for non-linearity.
- **Optimizer:** Stochastic Gradient Descent (SGD).
- **Loss Function:** Cross-Entropy Loss.

### Defining the Model:
```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define CNN model
class CNN(nn.Module):
    def __init__(self, out_1=16, out_2=32, number_of_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, out_1, kernel_size=5, padding=2)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(out_1, out_2, kernel_size=5, padding=2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(out_2 * 4 * 4, number_of_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = torch.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# Initialize model
model = CNN(out_1=16, out_2=32, number_of_classes=10)
```

### Defining the Loss Function & Optimizer:
```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
```

## Training Process
### Training Loop:
- Iterate over multiple epochs to optimize the model.
- Compute **loss** and update weights using backpropagation.
- Track **accuracy** across validation data.

```python
import time

start_time = time.time()
cost_list = []
accuracy_list = []
n_epochs = 5
N_test = len(dataset_val)

for epoch in range(n_epochs):
    cost = 0
    model.train()
    for x, y in train_loader:
        optimizer.zero_grad()
        z = model(x)
        loss = criterion(z, y)
        loss.backward()
        optimizer.step()
        cost += loss.item()

    correct = 0
    model.eval()
    for x_test, y_test in test_loader:
        z = model(x_test)
        _, yhat = torch.max(z.data, 1)
        correct += (yhat == y_test).sum().item()
    
    accuracy = correct / N_test
    accuracy_list.append(accuracy)
    cost_list.append(cost)
```

## Results & Visualization
### Plot Training Cost & Accuracy:
```python
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.plot(cost_list, color=color)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Cost', color=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.plot(accuracy_list, color=color)
ax2.set_ylabel('Accuracy', color=color)

fig.tight_layout()
plt.show()
```

## Sample Predictions
Include screenshots of sample predictions showcasing classification results.

## Contributors
- **Santhosh** - Developed and trained the model.

## License
This project is released under the **MIT License**.

