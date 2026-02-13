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

<img width="962" height="468" alt="image" src="https://github.com/user-attachments/assets/4fc6bd9a-604c-49c7-b3a6-328330a7112a" />


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
```
```py
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNClassifier().to(device)

print('Name: JAGANNIVASH U M')
print('Register Number: 212224240059')
summary(model, input_size=(1, 28, 28))
```
```py
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
