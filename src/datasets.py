"""
MNIST Dataset Loading and Preprocessing Utilities.

This module provides standardized functions for loading and preprocessing 
MNIST data across different model implementations. It handles data splitting,
normalization, and label encoding consistently.
"""

from typing import Tuple, Optional, Union
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from .utils import one_hot_encode, normalize_data, set_random_seed


def load_mnist_data(
    normalize: bool = True,
    flatten: bool = True,
    one_hot_labels: bool = False,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess MNIST dataset with train/validation/test splits.
    
    Args:
        normalize: Whether to normalize pixel values to [0,1]
        flatten: Whether to flatten images to 784-dimensional vectors
        one_hot_labels: Whether to one-hot encode labels
        random_state: Random seed for reproducible splits
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        
    Note:
        - Training set: 42,000 samples (60% of 70,000)
        - Validation set: 14,000 samples (20% of 70,000) 
        - Test set: 14,000 samples (20% of 70,000)
    """
    set_random_seed(random_state)
    
    # Load MNIST data from scikit-learn
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data.astype(np.float32), mnist.target
    
    # Split into train/temp (60%/40%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state, stratify=y
    )
    
    # Split temp into validation/test (20%/20% of original)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
    )
    
    # Normalize pixel values if requested
    if normalize:
        X_train = normalize_data(X_train, method="minmax")
        X_val = normalize_data(X_val, method="minmax")
        X_test = normalize_data(X_test, method="minmax")
    
    # Reshape data if not flattened
    if not flatten:
        X_train = X_train.reshape(-1, 28, 28)
        X_val = X_val.reshape(-1, 28, 28)
        X_test = X_test.reshape(-1, 28, 28)
    
    # Encode labels if requested
    if one_hot_labels:
        y_train = one_hot_encode(y_train)
        y_val = one_hot_encode(y_val) 
        y_test = one_hot_encode(y_test)
    else:
        # Convert string labels to integers
        y_train = y_train.astype(int)
        y_val = y_val.astype(int)
        y_test = y_test.astype(int)
    
    print(f"Dataset loaded successfully:")
    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples") 
    print(f"  Test: {X_test.shape[0]} samples")
    print(f"  Image shape: {X_train.shape[1:]}")
    print(f"  Label format: {'one-hot' if one_hot_labels else 'integer'}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_mnist_for_torch(
    batch_size: int = 64,
    normalize: bool = True
) -> Tuple:
    """
    Load MNIST data formatted for PyTorch training.
    
    Args:
        batch_size: Batch size for data loaders
        normalize: Whether to normalize pixel values
        
    Returns:
        Tuple of (train_loader, test_loader, input_size, num_classes)
    """
    try:
        import torch
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms
    except ImportError:
        raise ImportError("PyTorch not available. Install with: pip install torch torchvision")
    
    # Define transforms
    transform_list = [transforms.ToTensor()]
    if normalize:
        # Standard MNIST normalization
        transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
    
    # Add flattening transform
    transform_list.append(transforms.Lambda(lambda x: x.view(-1)))
    
    transform = transforms.Compose(transform_list)
    
    # Load datasets
    train_dataset = datasets.MNIST(
        root='data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root='data', train=False, download=True, transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, test_loader, 784, 10


def prepare_sequential_data(
    X_train: np.ndarray, 
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    normalize: bool = True
) -> Tuple[list, list]:
    """
    Prepare data in the format expected by the sequential neural network.
    
    Args:
        X_train: Training images
        y_train: Training labels  
        X_test: Test images
        y_test: Test labels
        normalize: Whether to normalize pixel values
        
    Returns:
        Tuple of (training_data, test_data) as lists of (x, y) tuples
    """
    if normalize:
        X_train = normalize_data(X_train, method="minmax")
        X_test = normalize_data(X_test, method="minmax")
    
    # Ensure labels are one-hot encoded
    if y_train.ndim == 1 or (y_train.ndim == 2 and y_train.shape[1] == 1):
        y_train = one_hot_encode(y_train)
    if y_test.ndim == 1 or (y_test.ndim == 2 and y_test.shape[1] == 1):
        y_test = one_hot_encode(y_test)
    
    # Convert to list of tuples format
    training_data = list(zip(X_train, y_train))
    test_data = list(zip(X_test, y_test))
    
    return training_data, test_data


def create_small_dataset(
    X: np.ndarray, 
    y: np.ndarray, 
    n_samples: int = 1000,
    stratify: bool = True,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a smaller subset of the data for quick testing/prototyping.
    
    Args:
        X: Input data
        y: Labels
        n_samples: Number of samples to select
        stratify: Whether to maintain class distribution
        random_state: Random seed
        
    Returns:
        Tuple of (X_subset, y_subset)
    """
    set_random_seed(random_state)
    
    if n_samples >= len(X):
        return X, y
    
    if stratify and y.ndim == 1:  # Only stratify for non-one-hot labels
        from sklearn.model_selection import train_test_split
        X_subset, _, y_subset, _ = train_test_split(
            X, y, train_size=n_samples, random_state=random_state, stratify=y
        )
    else:
        # Random sampling
        indices = np.random.choice(len(X), size=n_samples, replace=False)
        X_subset, y_subset = X[indices], y[indices]
    
    return X_subset, y_subset 