"""
train.py
--------

Module responsible for training a neural network classifier that distinguishes
between AI-generated and human-created images.

This module defines the main ``train`` function, which loads image data, splits
it into training and test sets, trains a convolutional neural network (defined
in ``net.py``), and evaluates its performance. It returns key training metrics,
predictions, and probability scores for further analysis and visualization.

Typical usage example:
    >>> from train import train
    >>> results = train("./data/interim/initial_data", epochs=5, lr=0.001, batch_size=32)
    >>> train_losses, train_accs, y_true, y_pred, *_ = results
    >>> print(f"Final accuracy: {train_accs[-1]:.2f}")
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(__file__)) 
import dataset
from net import Net


def train(path, epochs=2, lr=0.001, batch_size=32):
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

    dataset_full = dataset.ImageDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset_full))
    test_size = len(dataset_full) - train_size
    train_dataset, test_dataset = random_split(dataset_full, [train_size, test_size])
            

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    train_losses = []
    train_accs = []

    for epoch in range(epochs):
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
            
    return train_losses, train_accs, y_true, y_pred, losses, images, labels_list, preds, probs, true_labels, net
















