"""
PyTorch Neural Network Implementation for MNIST Classification.

This module provides a clean PyTorch implementation of a fully-connected
neural network for MNIST digit classification, with modern best practices
including proper device handling, reproducible training, and comprehensive evaluation.
"""

from typing import Tuple, Optional, Union
import numpy as np
import time

from .utils import calculate_accuracy, set_random_seed

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix


class MNISTNet(nn.Module):
    """
    PyTorch neural network for MNIST classification.
    
    Architecture:
    - Input: 784 (28x28 flattened)
    - Hidden 1: 392 neurons + ReLU
    - Hidden 2: 196 neurons + ReLU  
    - Output: 10 neurons (logits)
    
    Uses modern PyTorch practices:
    - Proper weight initialization
    - Dropout for regularization (optional)
    - Batch normalization (optional)
    """
    
    def __init__(self, dropout_rate: float = 0.0, use_batch_norm: bool = False):
        super(MNISTNet, self).__init__()
        
        self.use_batch_norm = use_batch_norm
        
        # Define layers
        self.fc1 = nn.Linear(784, 392)
        self.fc2 = nn.Linear(392, 196)
        self.fc3 = nn.Linear(196, 10)
        
        # Optional batch normalization
        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(392)
            self.bn2 = nn.BatchNorm1d(196)
        
        # Optional dropout
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/Glorot initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Flatten input if needed
        x = x.view(x.size(0), -1)
        
        # First hidden layer
        x = self.fc1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = F.relu(x)
        if self.dropout:
            x = self.dropout(x)
        
        # Second hidden layer
        x = self.fc2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = F.relu(x)
        if self.dropout:
            x = self.dropout(x)
        
        # Output layer (no activation - raw logits)
        x = self.fc3(x)
        
        return x


class TorchNNClassifier:
    """
    PyTorch Neural Network Classifier for MNIST.
    
    Provides a scikit-learn-like interface for training and evaluating
    PyTorch models with proper device handling and reproducible results.
    """
    
    def __init__(
        self,
        hidden_sizes: Tuple[int, int] = (392, 196),
        dropout_rate: float = 0.0,
        use_batch_norm: bool = False,
        random_state: int = 42
    ):      
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.random_state = random_state
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = None
        self.is_trained = False
        
        # Set random seeds for reproducibility
        self._set_seeds()
    
    def _set_seeds(self):
        """Set random seeds for reproducible results."""
        set_random_seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_state)
            torch.cuda.manual_seed_all(self.random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _create_model(self) -> MNISTNet:
        """Create and initialize the neural network model."""
        model = MNISTNet(
            dropout_rate=self.dropout_rate,
            use_batch_norm=self.use_batch_norm
        )
        return model.to(self.device)
    
    def _create_data_loaders(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        batch_size: int = 64
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Create PyTorch data loaders from numpy arrays."""
        from torch.utils.data import TensorDataset, DataLoader
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val)  
            y_val_tensor = torch.LongTensor(y_val)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 10,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> 'TorchNNClassifier':
        """
        Train the neural network.
        
        Args:
            X_train: Training data, shape (n_samples, 784)
            y_train: Training labels (integer format)
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            X_val: Validation data (optional)
            y_val: Validation labels (optional)
            verbose: Whether to print training progress
            
        Returns:
            Self for method chaining
        """
        # Ensure labels are integers
        if y_train.ndim > 1:  # One-hot encoded
            y_train = np.argmax(y_train, axis=1)
        if y_val is not None and y_val.ndim > 1:  # One-hot encoded
            y_val = np.argmax(y_val, axis=1)
        
        # Create model and data loaders
        self.model = self._create_model()
        train_loader, val_loader = self._create_data_loaders(
            X_train, y_train, X_val, y_val, batch_size
        )
        
        # Set up optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        if verbose:
            print(f"Training on device: {self.device}")
            print(f"Training samples: {len(X_train)}, Epochs: {epochs}")
        
        # Training loop
        start_time = time.time()
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                epoch_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            # Calculate training metrics
            train_loss = epoch_loss / total
            train_acc = correct / total
            
            # Validation phase
            val_acc = None
            if val_loader is not None:
                val_acc = self._evaluate_loader(val_loader)
            
            # Print progress
            if verbose:
                if val_acc is not None:
                    print(f"Epoch {epoch+1:2d}/{epochs}: "
                          f"Loss: {train_loss:.4f}, "
                          f"Train Acc: {train_acc:.4f}, "
                          f"Val Acc: {val_acc:.4f}")
                else:
                    print(f"Epoch {epoch+1:2d}/{epochs}: "
                          f"Loss: {train_loss:.4f}, "
                          f"Train Acc: {train_acc:.4f}")
        
        if verbose:
            elapsed = time.time() - start_time
            print(f"Training completed in {elapsed:.2f} seconds")
        
        self.is_trained = True
        return self
    
    def _evaluate_loader(self, data_loader: DataLoader) -> float:
        """Evaluate model on a data loader."""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        return correct / total
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predictions = torch.max(outputs, 1)
        
        return predictions.cpu().numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = F.softmax(outputs, dim=1)
        
        return probabilities.cpu().numpy()
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test data
            y_test: Test labels
            
        Returns:
            Tuple of (accuracy, confusion_matrix)
        """
        predictions = self.predict(X_test)
        
        # Handle label format conversion
        if y_test.ndim > 1 and y_test.shape[1] == 10:  # One-hot encoded
            true_labels = np.argmax(y_test, axis=1)
        else:  # Integer labels
            true_labels = y_test.astype(int)
        
        accuracy = calculate_accuracy(predictions, true_labels)
        cm = confusion_matrix(true_labels, predictions)
        
        return accuracy, cm
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'hidden_sizes': self.hidden_sizes,
                'dropout_rate': self.dropout_rate, 
                'use_batch_norm': self.use_batch_norm
            }
        }, filepath)
    
    def load_model(self, filepath: str) -> 'TorchNNClassifier':
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Update config if needed
        config = checkpoint['model_config']
        self.hidden_sizes = config['hidden_sizes']
        self.dropout_rate = config['dropout_rate']
        self.use_batch_norm = config['use_batch_norm']
        
        # Create and load model
        self.model = self._create_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.is_trained = True
        
        return self 