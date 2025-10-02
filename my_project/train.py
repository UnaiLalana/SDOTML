import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import dataset
from torch.utils.data import random_split
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.net import Net



class ImageDataset(Dataset):
    """
    Custom PyTorch Dataset for handling images and their corresponding labels.

    This dataset wrapper stores image tensors and their labels, providing
    indexing and length operations compatible with PyTorch's ``DataLoader``.

    Args:
        images (array-like or torch.Tensor): Collection of images. Each element 
            is expected to be a tensor or array representing an image.
        labels (array-like or torch.Tensor): Collection of labels corresponding 
            to each image.

    Attributes:
        images (array-like or torch.Tensor): Stored image data.
        labels (array-like or torch.Tensor): Stored labels for the images.

    Example:
        >>> import torch
        >>> from torch.utils.data import DataLoader
        >>> from dataset import ImageDataset
        >>> images = torch.randn(100, 3, 64, 64)  # 100 RGB images of size 64x64
        >>> labels = torch.randint(0, 2, (100,))  # binary labels
        >>> dataset = ImageDataset(images, labels)
        >>> dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        >>> for batch_images, batch_labels in dataloader:
        ...     print(batch_images.shape, batch_labels.shape)
        torch.Size([16, 3, 64, 64]) torch.Size([16])
    """
    
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]




def train(path):
    """
    Trains a neural network model on image data from a given directory path.

    This function loads the dataset from the specified path, splits it into
    training and testing sets, and trains a neural network model using
    cross-entropy loss and stochastic gradient descent (SGD). It also evaluates
    the model on the test set, collecting metrics such as accuracy, predicted
    labels, probabilities, and per-sample losses.

    Args:
        path (str): Path of the directory where the dataset is stored. The path
            should be compatible with ``dataset.read_dataset(path)`` to return
            tensors for features (X) and labels (y).

    Returns:
        tuple:
            - train_losses (list[float]): Training loss per epoch.
            - train_accs (list[float]): Training accuracy per epoch.
            - y_true (list[int]): True labels from the test dataset.
            - y_pred (list[int]): Predicted labels from the test dataset.
            - losses (list[float]): Per-sample loss values on the test set.
            - images (list[numpy.ndarray]): Input images from the test dataset.
            - labels_list (list[int]): True labels from the test dataset (redundant with y_true).
            - preds (list[int]): Predicted labels from the test dataset (redundant with y_pred).
            - probs (list[float]): Predicted probabilities for the positive class (class index 1).
            - true_labels (list[int]): True labels corresponding to ``probs``.

    Example:
        >>> from myproject.training import train
        >>> results = train(model, "./data/images")
        >>> train_losses, train_accs, y_true, y_pred, *_ = results
        >>> print(f"Final accuracy: {train_accs[-1]:.2f}")
    """
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
















