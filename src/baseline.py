"""
Baseline MNIST Classifier using Average Pixel Values and Cosine Similarity.

This module implements a simple baseline classifier that computes digit-wise
centroids (average pixel values) and classifies new samples using cosine similarity.
The approach achieves approximately 75% accuracy on MNIST.
"""

from typing import Tuple, Optional
import numpy as np
from sklearn.metrics import confusion_matrix

from .utils import calculate_accuracy, one_hot_encode, decode_one_hot, set_random_seed


class BaselineClassifier:
    """
    Baseline classifier using average pixel values and cosine similarity.
    
    This classifier works by:
    1. Computing the average pixel values (centroid) for each digit class
    2. L2-normalizing these centroids 
    3. Classifying new samples by finding the centroid with highest cosine similarity
    
    Mathematical foundation:
    - Cosine similarity: cos(θ) = (a·b) / (||a|| ||b||)
    - For normalized vectors: cos(θ) = a·b
    """
    
    def __init__(self, normalize_centroids: bool = True):
        """
        Initialize the baseline classifier.
        
        Args:
            normalize_centroids: Whether to L2-normalize the centroids
        """
        self.normalize_centroids = normalize_centroids
        self.centroids = None
        self.is_trained = False
        
    def _compute_centroids(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute digit-wise centroids from training data.
        
        Args:
            X: Training images, shape (n_samples, 784)
            y: Training labels (one-hot or integer)
            
        Returns:
            Centroids array of shape (10, 784)
        """
        # Handle label format conversion
        if y.ndim > 1 and y.shape[1] == 10:  # One-hot encoded
            labels = decode_one_hot(y)
        else:  # Raw labels (strings or ints)
            labels = np.array([int(label) for label in y])
        
        # Compute centroids for each digit 0-9
        centroids = np.zeros((10, X.shape[1]))
        for digit in range(10):
            digit_mask = labels == digit
            if np.any(digit_mask):
                centroids[digit] = np.mean(X[digit_mask], axis=0)
        
        # L2-normalize each centroid if specified
        if self.normalize_centroids:
            norms = np.linalg.norm(centroids, axis=1, keepdims=True)
            # Avoid division by zero
            norms = np.where(norms == 0, 1, norms)
            centroids = centroids / norms
            
        return centroids
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> 'BaselineClassifier':
        """
        Train the baseline classifier by computing digit centroids.
        
        Args:
            X_train: Training images, shape (n_samples, 784)
            y_train: Training labels 
            
        Returns:
            Self for method chaining
        """
        set_random_seed(42)  # For reproducibility
        
        self.centroids = self._compute_centroids(X_train, y_train)
        self.is_trained = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels using cosine similarity to centroids.
        
        Args:
            X: Input images, shape (n_samples, 784)
            
        Returns:
            Predicted integer labels, shape (n_samples,)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Handle empty input
        if len(X) == 0:
            return np.array([])
        
        # Normalize input if centroids are normalized
        if self.normalize_centroids:
            X_norms = np.linalg.norm(X, axis=1, keepdims=True)
            # Handle zero-norm vectors (all-zero images)
            X_normalized = np.where(X_norms == 0, 0, X / X_norms)
        else:
            X_normalized = X
        
        # Compute cosine similarities: (n_samples, 784) @ (784, 10) -> (n_samples, 10)
        similarities = np.dot(X_normalized, self.centroids.T)
        
        # Handle zero-norm inputs by setting their similarities to very low values
        if self.normalize_centroids:
            zero_norm_mask = np.linalg.norm(X, axis=1) == 0
            similarities[zero_norm_mask] = -1e9
        
        # Return class with highest similarity
        return np.argmax(similarities, axis=1).astype(int)
    
    def evaluate(
        self, 
        X_test: np.ndarray, 
        y_test: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Evaluate the classifier on test data.
        
        Args:
            X_test: Test images, shape (n_samples, 784)
            y_test: Test labels
            
        Returns:
            Tuple of (accuracy, confusion_matrix)
        """
        predictions = self.predict(X_test)
        
        # Handle label format conversion for true labels
        if y_test.ndim > 1 and y_test.shape[1] == 10:  # One-hot encoded
            true_labels = decode_one_hot(y_test)
        else:  # Raw labels
            true_labels = np.array([int(label) for label in y_test])
        
        accuracy = calculate_accuracy(predictions, true_labels)
        cm = confusion_matrix(true_labels, predictions)
        
        return accuracy, cm
    
    def get_centroids(self) -> Optional[np.ndarray]:
        """
        Get the computed centroids.
        
        Returns:
            Centroids array of shape (10, 784) or None if not trained
        """
        return self.centroids if self.is_trained else None
    
    def visualize_centroids(self) -> np.ndarray:
        """
        Get centroids reshaped for visualization.
        
        Returns:
            Centroids reshaped to (10, 28, 28) for plotting
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before visualization")
        
        return self.centroids.reshape(10, 28, 28) 