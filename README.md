# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

The objective of this experiment is to develop a Convolutional Neural Network (CNN) using PyTorch to perform multi-class image classification. The model takes grayscale images as input and classifies them into one of ten categories. The aim is to extract meaningful features using convolutional layers and accurately predict the corresponding class label.
The dataset used in this experiment consists of grayscale images of size 28 × 28 pixels, designed for multi-class image classification. Each image belongs to one of 10 different classes, represented by numeric labels ranging from 0 to 9.

Each image is stored as a tensor of shape:

(1 × 28 × 28)


Where:

1 represents the grayscale channel

28 × 28 represents the image dimensions

Labels represent the corresponding class category

For implementation and testing purposes, the dataset was loaded using PyTorch utilities such as TensorDataset and DataLoader, enabling efficient batching and training of the CNN model.

## Neural Network Model

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/450a1337-b45c-42cc-bb6a-df30da49bce3" />


## DESIGN STEPS

## STEP 1:

Import required libraries such as PyTorch, torchvision, NumPy, and Matplotlib. Load and preprocess the image dataset using transformations.

## STEP 2:

Design and implement a Convolutional Neural Network using convolutional layers, pooling layers, and fully connected layers.

## STEP 3:

Train the CNN model using a suitable loss function and optimizer. Evaluate the model using test data and generate performance metrics.


## PROGRAM

### Name:JAGANNIVASH U M
### Register Number: 212224240059
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from torchsummary import summary

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.FashionMNIST(
    root="./data", train=True, transform=transform, download=True
)
test_dataset = torchvision.datasets.FashionMNIST(
    root="./data", train=False, transform=transform, download=True
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f'Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}')


class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNClassifier().to(device)

print('Name: JAGANNIVASH U M')
print('Register Number: 212224240059')
summary(model, input_size=(1, 28, 28))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, num_epochs=3):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('Name: JAGANNIVASH U M')
        print('Register Number: 212224240059')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

train_model(model, train_loader)

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print('Name: JAGANNIVASH U M')
    print('Register Number: 212224240059')
    print(f'Test Accuracy: {accuracy:.4f}')

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=test_dataset.classes,
                yticklabels=test_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    print('Name: JAGANNIVASH U M')
    print('Register Number: 212224240059')
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

test_model(model, test_loader)


def predict_image(model, image_index, dataset):
    model.eval()
    image, label = dataset[image_index]
    with torch.no_grad():
        image_tensor = image.unsqueeze(0).to(device)
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)

    class_names = dataset.classes
    print('Name: JAGANNIVASH U M')
    print('Register Number: 212224240059')
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f'Actual: {class_names[label]}\nPredicted: {class_names[predicted.item()]}')
    plt.axis("off")
    plt.show()
    print(f'Actual: {class_names[label]}, Predicted: {class_names[predicted.item()]}')

# Example: Predict image at index 80
predict_image(model, image_index=80, dataset=test_dataset)
```

## OUTPUT
### Training Loss per Epoch

<img width="414" height="346" alt="image" src="https://github.com/user-attachments/assets/be5bbf14-6480-4fe5-89ee-c320bbc1bc58" />


### Confusion Matrix

<img width="1022" height="754" alt="image" src="https://github.com/user-attachments/assets/d4b02a4c-1d73-4a87-bc89-86c309f9fb6f" />


### Classification Report

<img width="697" height="441" alt="image" src="https://github.com/user-attachments/assets/de7ede47-1f01-4f0a-9df0-a5140b65feb2" />


### New Sample Data Prediction

<img width="627" height="547" alt="image" src="https://github.com/user-attachments/assets/9b1d031a-d7fb-4741-8d1c-7bd57e0d88ee" />


## RESULT
Thus, a convolutional neural network for image classification was successfully implemented and verified using an Excel-based dataset
