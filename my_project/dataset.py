"""
dataset.py
----------

Provides utilities for loading, preprocessing, and managing image datasets used in
the AI vs. Human image classification project.

This module includes:
    - Functions for reading datasets from directories or ZIP archives.
    - Data preprocessing (resizing, normalization, and tensor conversion).
    - A custom PyTorch ``Dataset`` class for convenient integration with ``DataLoader``.

It expects each dataset to include:
    - A CSV file containing image paths and labels (e.g., ``train.csv``).
    - A directory containing the corresponding image files (e.g., ``train/``).
    - Optionally, a ZIP archive (e.g., ``train.zip``) that will be extracted automatically
      if the dataset directory is missing.

Typical usage example:
    >>> from dataset import read_dataset, ImageDataset
    >>> X, y = read_dataset("./data/interim/initial_data")
    >>> dataset = ImageDataset(X, y)
    >>> print(len(dataset))
    1000
"""
import pandas as pd
import sys
import os
import shutil
import zipfile
import cv2
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
from PIL import Image
from torch.utils.data import Dataset


def load_data_from_zip(output_dir):
    """
    Extracts the contents of a ZIP file into a specified output directory.

    This function takes a path (without the `.zip` extension), opens the corresponding 
    ZIP file, and extracts all its contents into the given output directory.

    Args:
        output_dir (str): Path to the directory where the ZIP file will be extracted.
            The function expects a file named ``<output_dir>.zip`` to exist.

    Returns:
        None

    Example:
        >>> # Suppose you have 'data.zip' in your working directory
        >>> load_data_from_zip("data")
        >>> # This extracts the contents of 'data.zip' into the folder './data'
    """
    with zipfile.ZipFile(output_dir+'.zip', 'r') as zipf:
            zipf.extractall(output_dir)

def read_dataset(dataset_dir):
    """
    Loads an image dataset from a directory and its corresponding CSV file.

    This function expects a dataset directory and a CSV file with the same name
    (e.g., ``dataset_dir/`` and ``dataset_dir.csv``). The CSV file should contain
    at least two columns: ``file_name`` (the relative path of each image) and 
    ``label`` (the class label). 

    If the directory does not exist, the function attempts to extract it from a
    ZIP file with the same base name (e.g., ``dataset_dir.zip``).

    Each image is read with OpenCV, converted to RGB, transformed into a 
    PyTorch tensor, resized to (256, 256), and normalized to the range [-1, 1].

    Args:
        dataset_dir (str): Path to the dataset directory (without `.csv` or `.zip` extension). 
            Example: ``"./data/train"`` expects:
                - ``./data/train.csv``
                - ``./data/train/`` directory with images
                - optionally ``./data/train.zip`` if the directory is missing.

    Returns:
        tuple:
            - X (torch.Tensor): Tensor of shape (N, 3, 256, 256) containing N images.
            - y (torch.Tensor): Tensor of shape (N,) containing integer labels.

    Example:
        >>> X, y = read_dataset("./data/train")
        >>> print(X.shape, y.shape)
        torch.Size([1000, 3, 256, 256]) torch.Size([1000])
    """

    if not os.path.exists(dataset_dir):
        print("No existe el directorio con las imagenes:", dataset_dir)
        load_data_from_zip(dataset_dir)
    

    df = pd.read_csv(os.path.join(dataset_dir + ".csv"))
    X = []
    y = []

    transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    
    for _, row in df.iterrows():
        img = os.path.join(dataset_dir, row['file_name'].replace('train_data/', ''))
        if img is not None:
            img_cv = cv2.imread(img)       
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

            img_pil = Image.fromarray(img_cv)

            img_tensor = transform(img_pil)

            X.append(img_tensor)
            y.append(row['label'])
            
    X = torch.stack(X)
    y = torch.tensor(y, dtype=torch.long)
    return X, y


if __name__ == "__main__":

    output_dir = "data/interim/initial_data"
    if os.path.exists(output_dir + ".zip"):
        load_data_from_zip(output_dir)

    else:
        data = pd.read_csv("data/raw/train.csv")

        initial_data = data[:1000]
        initial_data.to_csv("data/interim/initial_data.csv", index=False)

        for picture in initial_data['file_name']:
            output_dir = "data/interim/initial_data"
            os.makedirs(output_dir, exist_ok=True)
            src_path = os.path.join("data/raw/", picture)
            dst_path = os.path.join(output_dir, picture.replace('train_data/', ''))
            shutil.copy2(src_path, dst_path)

        zip_path = output_dir + ".zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, output_dir)
                    zipf.write(file_path, arcname)


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



