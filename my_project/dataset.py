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




