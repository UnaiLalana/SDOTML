import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import dataset
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve

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


net = Net()

X_tensor, y_tensor = dataset.read_dataset("./data/interim/initial_data")

class ImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

dataset_full = ImageDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset_full))
test_size = len(dataset_full) - train_size
train_dataset, test_dataset = random_split(dataset_full, [train_size, test_size])
        

batch_size = 32

trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


labels_np = y_tensor.numpy()
classes, counts = np.unique(labels_np, return_counts=True)

plt.figure(figsize=(6,4))
plt.bar(classes, counts)
plt.xticks(classes)
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

print("Image tensor shape:", X_tensor.shape)

def imshow(img, title=""):
    img = img.numpy().transpose((1, 2, 0))
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")

plt.figure(figsize=(8,4))
for cls in classes:
    idx = np.where(labels_np == cls)[0][0]
    plt.subplot(1, len(classes), cls+1)
    imshow(X_tensor[idx], f"Class {cls}")
plt.show()

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

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(train_losses, label="Loss")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(train_accs, label="Accuracy")
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


y_true, y_pred = [], []
net.eval()
with torch.no_grad():
    for inputs, labels in testloader:
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.numpy())
        y_pred.extend(predicted.numpy())

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (Test Set)")
plt.show()


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

idx_sorted = np.argsort(losses)[-5:]
plt.figure(figsize=(12,6))
for i, idx in enumerate(idx_sorted):
    plt.subplot(1,5,i+1)
    imshow(torch.tensor(images[idx]), f"T:{labels_list[idx]} P:{preds[idx]}\nLoss:{losses[idx]:.2f}")
plt.show()

probs = []
true_labels = []
net.eval()
with torch.no_grad():
    for inputs, labels in testloader:
        outputs = net(inputs)
        probs_batch = F.softmax(outputs, dim=1)[:,1]
        probs.extend(probs_batch.numpy())
        true_labels.extend(labels.numpy())

prob_true, prob_pred = calibration_curve(true_labels, probs, n_bins=10)

plt.figure(figsize=(6,6))
plt.plot(prob_pred, prob_true, marker='o', label="Model")
plt.plot([0,1],[0,1], linestyle="--", label="Perfectly calibrated")
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives")
plt.title("Calibration Curve")
plt.legend()
plt.show()

