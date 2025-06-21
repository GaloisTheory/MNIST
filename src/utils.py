"""
Utility functions for MNIST classification project.

Contains common helper functions for data manipulation, evaluation metrics,
and visualization utilities used across different models.
"""

from typing import Tuple, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.figure
from sklearn.metrics import confusion_matrix
import seaborn as sns


def one_hot_encode(labels: Union[np.ndarray, list], num_classes: int = 10) -> np.ndarray:
    """
    Convert integer labels to one-hot encoded vectors.
    
    Args:
        labels: Array of integer labels
        num_classes: Number of classes (default: 10 for digits)
        
    Returns:
        One-hot encoded labels of shape (n_samples, num_classes)
    """
    if isinstance(labels, list):
        labels = np.array(labels)
    
    # Handle string labels by converting to int
    if labels.dtype.kind in ['U', 'S']:  # Unicode or byte string
        labels = labels.astype(int)
    
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels] = 1.0
    return one_hot


def decode_one_hot(one_hot_labels: np.ndarray) -> np.ndarray:
    """
    Convert one-hot encoded labels back to integer labels.
    
    Args:
        one_hot_labels: One-hot encoded labels
        
    Returns:
        Integer labels
    """
    return np.argmax(one_hot_labels, axis=1)


def calculate_accuracy(predictions: np.ndarray, true_labels: np.ndarray) -> float:
    """
    Calculate classification accuracy.
    
    Args:
        predictions: Predicted labels
        true_labels: True labels
        
    Returns:
        Accuracy as a float between 0 and 1
    """
    # Handle one-hot encoded labels
    if true_labels.ndim > 1 and true_labels.shape[1] > 1:
        true_labels = decode_one_hot(true_labels)
    if predictions.ndim > 1 and predictions.shape[1] > 1:
        predictions = decode_one_hot(predictions)
        
    return np.mean(predictions == true_labels)


def plot_digit(
    data: np.ndarray, 
    title: Optional[str] = None, 
    ax: Optional[matplotlib.axes.Axes] = None
) -> matplotlib.axes.Axes:
    """
    Plot a single MNIST digit.
    
    Args:
        data: Flattened digit data (784 elements) or 28x28 image
        title: Optional title for the plot
        ax: Optional matplotlib axes to plot on
        
    Returns:
        The matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    
    # Reshape if needed
    if data.shape == (784,):
        image = data.reshape(28, 28)
    else:
        image = data
    
    ax.imshow(image, cmap='binary')
    ax.axis('off')
    if title:
        ax.set_title(title)
    
    return ax


def plot_confusion_matrix(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot a confusion matrix for classification results.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels  
        title: Title for the plot
        figsize: Figure size tuple
        
    Returns:
        The matplotlib figure object
    """
    # Handle one-hot encoded labels
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true = decode_one_hot(y_true)
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred = decode_one_hot(y_pred)
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10), ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)
    
    return fig


def plot_sample_errors(
    X: np.ndarray, 
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    n_samples: int = 8,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot sample misclassified digits.
    
    Args:
        X: Input images (flattened or 28x28)
        y_true: True labels
        y_pred: Predicted labels
        n_samples: Number of error samples to show
        figsize: Figure size tuple
        
    Returns:
        The matplotlib figure object
    """
    # Handle one-hot encoded labels
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true = decode_one_hot(y_true)
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred = decode_one_hot(y_pred)
    
    # Find misclassified samples
    errors = np.where(y_true != y_pred)[0]
    
    if len(errors) == 0:
        print("No misclassifications found!")
        return plt.figure(figsize=figsize)
    
    # Sample random errors
    n_show = min(n_samples, len(errors))
    error_indices = np.random.choice(errors, n_show, replace=False)
    
    fig, axes = plt.subplots(2, n_show // 2, figsize=figsize)
    axes = axes.flatten() if n_show > 1 else [axes]
    
    for i, idx in enumerate(error_indices):
        if i >= len(axes):
            break
        plot_digit(X[idx], 
                  title=f"True: {y_true[idx]}, Pred: {y_pred[idx]}", 
                  ax=axes[i])
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.suptitle("Sample Misclassifications")
    plt.tight_layout()
    
    return fig


def normalize_data(X: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    Normalize input data.
    
    Args:
        X: Input data
        method: Normalization method ("minmax" or "zscore")
        
    Returns:
        Normalized data
    """
    if method == "minmax":
        return X / 255.0
    elif method == "zscore":
        return (X - np.mean(X)) / np.std(X)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed) 