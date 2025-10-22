"""
gradio_app.py
-------------

Interactive Gradio interface for exploring, training, and evaluating an
AI-vs-human image classification model.

This module provides:
- A data exploration interface to visualize sample images from the dataset.
- A training interface to configure and train a neural network model.
- An evaluation interface to display performance metrics such as accuracy and
  confusion matrix.

It uses the ``train`` module for model training and ``dataset`` for data loading.

Example:
    Run this script directly to launch the interactive Gradio demo:
        >>> python gradio_app.py
"""

import gradio as gr
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import sys
import os

sys.path.append(os.path.dirname(__file__))

import dataset
import train

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data/interim/initial_data")


def show_random_images(num_images):
    """
    Displays a set of random images from the dataset with their labels.

    Randomly selects a specified number of samples from the dataset and
    visualizes them using Matplotlib. Each image is labeled as either
    "Human-created" (label 0) or "AI-generated" (label 1).

    Args:
        num_images (int): Number of images to display.

    Returns:
        matplotlib.figure.Figure: A Matplotlib figure containing the displayed images.
    """
    X, y = dataset.read_dataset(DATA_DIR) 
    idx = np.random.choice(len(X), num_images, replace=False)
    
    fig, axes = plt.subplots(1, num_images, figsize=(num_images*3, 3))
    if num_images == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        img = X[idx[i]].permute(1, 2, 0).numpy()
        img = (img * 0.5 + 0.5)  
        ax.imshow(img)
        if y[idx[i]].item() == 0:
            ax.set_title(f"Label: {y[idx[i]].item()}, Human-created")
        else:
            ax.set_title(f"Label: {y[idx[i]].item()}, AI-generated")
        ax.axis("off")
    plt.tight_layout()
    return fig

trained_model = None  
y_true_test = None
y_pred_test = None

def train_model(epochs, lr, batch_size):
    """
    Trains the neural network model using the provided hyperparameters.

    This function calls ``train.train()`` to execute the training loop and
    stores the trained model and evaluation results for later use in the
    evaluation tab.

    Args:
        epochs (int): Number of epochs to train the model.
        lr (float): Learning rate for the optimizer.
        batch_size (int): Batch size used during training.

    Returns:
        str: Summary text containing the final loss and accuracy from training.
    """
    global trained_model, y_true_test, y_pred_test
    train_losses, train_accs, y_true, y_pred, _, _, _, _, _, _, net = train.train(DATA_DIR, epochs=epochs, lr=lr, batch_size=batch_size)
    
    trained_model = True  
    y_pred_test = y_pred
    y_true_test = y_true
    return f"Training completed. Last loss: {train_losses[-1]:.4f}, last accuracy: {train_accs[-1]:.4f}"

def evaluate_model():
    """
    Evaluates the performance of the trained model on the test dataset.

    Displays a confusion matrix heatmap and computes overall test accuracy.

    Returns:
        tuple:
            - fig (matplotlib.figure.Figure): Confusion matrix visualization.
            - eval_text (str): Text summary containing the test accuracy.
    """
    if trained_model is None:
        return None, "No trained model yet"

    # Usa las métricas de test guardadas
    cm = confusion_matrix(y_true_test, y_pred_test)
    fig, ax = plt.subplots(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    
    acc = accuracy_score(y_true_test, y_pred_test)
    eval_text = f"Accuracy on test: {acc:.4f}"
    
    return fig, eval_text




with gr.Blocks() as demo:
    gr.Markdown("# Image Classification Interactive Demo")

    with gr.Tab("Data Exploration"):
        gr.Markdown("Visualize images from the dataset")
        num_images = gr.Slider(1, 6, value=3, step=1, label="Number of images")
        img_btn = gr.Button("Show images")
        img_plot = gr.Plot()
        img_btn.click(show_random_images, inputs=num_images, outputs=img_plot)

    with gr.Tab("Training Interface"):
        gr.Markdown("Adjust the models parameters and train.")
        epochs = gr.Slider(1, 10, value=2, step=1, label="Epochs")
        lr = gr.Slider(1e-5, 1e-2, value=1e-3, step=1e-5, label="Learning Rate")
        batch_size = gr.Slider(8, 64, value=16, step=8, label="Batch Size")
        train_btn = gr.Button("Train model")
        train_output = gr.Textbox()
        train_btn.click(train_model, inputs=[epochs, lr, batch_size], outputs=train_output)

    with gr.Tab("Model Evaluation"):
        gr.Markdown("Evaluates the trained model")
        eval_btn = gr.Button("Evaluate the model")
        eval_plot = gr.Plot()
        eval_text = gr.Textbox()
        eval_btn.click(evaluate_model, outputs=[eval_plot, eval_text])

demo.launch()

