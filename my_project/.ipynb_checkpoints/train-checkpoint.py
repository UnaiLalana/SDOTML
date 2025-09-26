import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import dataset
from torch.utils.data import random_split
import numpy as np



class ImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 61 * 61, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(path):
    net = Net()

    X_tensor, y_tensor = dataset.read_dataset(path)

    dataset_full = ImageDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset_full))
    test_size = len(dataset_full) - train_size
    train_dataset, test_dataset = random_split(dataset_full, [train_size, test_size])
            

    batch_size = 32

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    train_losses = []
    train_accs = []

    for epoch in range(10):
        running_loss = 0.0
        correct, total = 0, 0
        net.train()
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(trainloader)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")


    y_true, y_pred = [], []
    net.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(predicted.numpy())

    losses = []
    images = []
    labels_list = []
    preds = []
    net.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = net(inputs)
            loss_batch = F.cross_entropy(outputs, labels, reduction="none")
            losses.extend(loss_batch.numpy())
            images.extend(inputs.numpy())
            labels_list.extend(labels.numpy())
            preds.extend(outputs.argmax(1).numpy())

    probs = []
    true_labels = []
    net.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = net(inputs)
            probs_batch = F.softmax(outputs, dim=1)[:,1]
            probs.extend(probs_batch.numpy())
            true_labels.extend(labels.numpy())
            
    return train_losses, train_accs, y_true, y_pred, losses, images, labels_list, preds, probs, true_labels
















