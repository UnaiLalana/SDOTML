import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """
    Convolutional Neural Network (CNN) for binary image classification.

    The network consists of:
        - Two convolutional layers (with ReLU + MaxPooling).
        - Three fully connected layers.
        - A final linear layer that outputs logits for 2 classes.

    Architecture:
        Input (3 channels) → Conv2d(3, 6, 5) → ReLU → MaxPool(2,2) →
        Conv2d(6, 16, 5) → ReLU → MaxPool(2,2) →
        Flatten →
        Linear(16 * 61 * 61 → 120) → ReLU →
        Linear(120 → 84) → ReLU →
        Linear(84 → 2).

    Args:
        nn.Module (torch.nn.Module): Base class for PyTorch models.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer (3 input channels, 6 output channels, 5x5 kernel).
        pool (nn.MaxPool2d): Max pooling layer with 2x2 kernel and stride 2.
        conv2 (nn.Conv2d): Second convolutional layer (6 input channels, 16 output channels, 5x5 kernel).
        fc1 (nn.Linear): Fully connected layer mapping 16 * 61 * 61 features to 120.
        fc2 (nn.Linear): Fully connected layer mapping 120 to 84.
        fc3 (nn.Linear): Final fully connected layer mapping 84 to 2 output classes.

    Example:
        >>> import torch
        >>> from models.net import Net
        >>> model = Net()
        >>> dummy_input = torch.randn(1, 3, 256, 256)  # batch of 1, 3 channels, 256x256 image
        >>> output = model(dummy_input)
        >>> print(output.shape)
        torch.Size([1, 2])
    """
    
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