"""
Sequential Neural Network Implementation from Scratch.

This module implements a fully-connected neural network using only NumPy,
with modular layers, multiple activation functions, and modern training techniques.
Mathematical foundation includes backpropagation, gradient descent, and cross-entropy loss.
"""

from typing import List, Tuple, Optional, Union
import numpy as np
import random
from sklearn.metrics import confusion_matrix

from .utils import calculate_accuracy, decode_one_hot, set_random_seed


class Layer:
    """Base layer class for building neural networks."""
    
    def __init__(self):
        self.params = {}
        self.previous = None
        self.next = None
        self.input_data = None
        self.output_data = None
        self.input_delta = None
        self.output_delta = None
    
    def connect(self, layer: 'Layer') -> None:
        """Connect this layer to the previous layer."""
        self.previous = layer
        layer.next = self

    def forward(self, input_data: Optional[np.ndarray] = None) -> None:
        """Forward pass - must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement forward")

    def get_forward_input(self) -> np.ndarray:
        """Get input for forward pass."""
        if self.previous is not None:
            return self.previous.output_data
        else:
            return self.input_data
    
    def backward(self) -> None:
        """Backward pass - must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement backward")
    
    def get_backward_input(self) -> np.ndarray:
        """Get input for backward pass."""
        if self.next is not None:
            return self.next.output_delta
        else:
            return self.input_delta
        
    def clear_deltas(self) -> None:
        """Clear gradient accumulations."""
        pass

    def update_params(self, learning_rate: float) -> None:
        """Update layer parameters."""
        pass

    def describe(self) -> str:
        """Return layer description."""
        raise NotImplementedError("Subclasses must implement describe")


class DenseLayer(Layer):
    """
    Fully connected (dense) layer.
    
    Implements: output = input @ weights.T + bias
    
    Mathematical foundation:
    - Forward: y = Wx + b where W is (output_dim, input_dim)
    - Backward: ∂L/∂W = δ * x.T, ∂L/∂b = δ, ∂L/∂x = W.T * δ
    """
    
    def __init__(self, input_dim: int, output_dim: int, weight_init: str = 'xavier'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Initialize weights using Xavier/Glorot initialization
        if weight_init == 'xavier':
            limit = np.sqrt(6.0 / (input_dim + output_dim))
            self.weights = np.random.uniform(-limit, limit, (output_dim, input_dim))
        else:
            self.weights = np.random.randn(output_dim, input_dim) * 0.01
        
        self.bias = np.zeros(output_dim)
        self.delta_b = None
        self.delta_w = None

    def forward(self, input_data: Optional[np.ndarray] = None) -> None:
        """Forward pass: y = Wx + b"""
        data = self.get_forward_input()
        self.output_data = np.dot(self.weights, data) + self.bias

    def backward(self) -> None:
        """Backward pass with gradient computation."""
        delta = self.get_backward_input()
        data = self.get_forward_input()

        # Accumulate gradients
        if self.delta_b is None:
            self.delta_b = delta.copy()
        else:
            self.delta_b += delta
        
        if self.delta_w is None:
            self.delta_w = np.outer(delta, data)
        else:
            self.delta_w += np.outer(delta, data)

        # Compute delta for previous layer
        self.output_delta = np.dot(self.weights.T, delta)

    def update_params(self, learning_rate: float) -> None:
        """Update weights and biases using gradient descent."""
        if self.delta_w is not None:
            self.weights -= learning_rate * self.delta_w
        if self.delta_b is not None:
            self.bias -= learning_rate * self.delta_b

    def clear_deltas(self) -> None:
        """Clear accumulated gradients."""
        self.delta_b = None
        self.delta_w = None

    def describe(self) -> str:
        return f"DenseLayer(input_dim={self.input_dim}, output_dim={self.output_dim})"


class ActivationLayer(Layer):
    """
    Activation layer supporting multiple activation functions.
    
    Supported activations:
    - sigmoid: σ(x) = 1/(1 + e^(-x))
    - relu: ReLU(x) = max(0, x)  
    - softmax: softmax(x)_i = e^(x_i) / Σ(e^(x_j))
    """
    
    def __init__(self, input_dim: int, activation_type: str = 'sigmoid'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.activation_type = activation_type

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        return np.where(x >= 0, 
                       1 / (1 + np.exp(-x)), 
                       np.exp(x) / (1 + np.exp(x)))

    def _sigmoid_prime(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid derivative."""
        s = self._sigmoid(x)
        return s * (1 - s)

    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation."""
        return np.maximum(0, x)

    def _relu_prime(self, x: np.ndarray) -> np.ndarray:
        """ReLU derivative."""
        return (x > 0).astype(float)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def forward(self, input_data: Optional[np.ndarray] = None) -> None:
        """Forward pass through activation function."""
        data = self.get_forward_input()
        
        if self.activation_type == 'sigmoid':
            self.output_data = self._sigmoid(data)
        elif self.activation_type == 'relu':
            self.output_data = self._relu(data)
        elif self.activation_type == 'softmax':
            self.output_data = self._softmax(data)
        else:
            raise ValueError(f"Unsupported activation type: {self.activation_type}")

    def backward(self) -> None:
        """Backward pass through activation function."""
        delta = self.get_backward_input()
        data = self.get_forward_input()
        
        if self.activation_type == 'sigmoid':
            self.output_delta = delta * self._sigmoid_prime(data)
        elif self.activation_type == 'relu':
            self.output_delta = delta * self._relu_prime(data)
        elif self.activation_type == 'softmax':
            # For softmax + cross-entropy, gradient simplifies
            self.output_delta = delta
        else:
            raise ValueError(f"Unsupported activation type: {self.activation_type}")

    def describe(self) -> str:
        return f"ActivationLayer(dim={self.input_dim}, activation={self.activation_type})"


class CrossEntropyLoss:
    """Cross-entropy loss for multi-class classification."""
    
    @staticmethod
    def loss_function(predictions: np.ndarray, labels: np.ndarray) -> float:
        """Compute cross-entropy loss."""
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        return -np.sum(labels * np.log(predictions))
    
    @staticmethod
    def loss_function_derivative(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Compute cross-entropy loss derivative."""
        return predictions - labels


class SequentialNNClassifier:
    """
    Sequential Neural Network Classifier.
    
    A modular neural network implementation with:
    - Flexible layer architecture
    - Mini-batch gradient descent
    - Cross-entropy loss with softmax
    - Comprehensive training metrics
    """
    
    def __init__(self, random_state: int = 42):
        self.layers: List[Layer] = []
        self.loss = CrossEntropyLoss()
        self.random_state = random_state
        self.is_trained = False

    def add(self, layer: Layer) -> 'SequentialNNClassifier':
        """Add a layer to the network."""
        self.layers.append(layer)
        if len(self.layers) > 1:
            layer.connect(self.layers[-2])
        return self

    def _single_forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for a single sample."""
        self.layers[0].input_data = x
        for layer in self.layers:
            layer.forward()
        return self.layers[-1].output_data

    def _forward_backward(self, mini_batch: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        """Forward and backward pass for a mini-batch."""
        for layer in self.layers:
            layer.clear_deltas()
        
        for x, y in mini_batch:
            # Forward pass
            predictions = self._single_forward(x)
            
            # Compute loss gradient
            loss_gradient = self.loss.loss_function_derivative(predictions, y)
            
            # Backward pass
            self.layers[-1].input_delta = loss_gradient
            for layer in reversed(self.layers):
                layer.backward()

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 10,
        batch_size: int = 100,
        learning_rate: float = 0.1,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> 'SequentialNNClassifier':
        """
        Train the neural network.
        
        Args:
            X_train: Training data
            y_train: Training labels (one-hot encoded)
            epochs: Number of training epochs
            batch_size: Mini-batch size
            learning_rate: Learning rate for gradient descent
            X_val: Validation data (optional)
            y_val: Validation labels (optional)
            verbose: Whether to print training progress
            
        Returns:
            Self for method chaining
        """
        set_random_seed(self.random_state)
        
        # Prepare training data
        training_data = list(zip(X_train, y_train))
        n = len(training_data)
        
        # Prepare validation data if provided
        val_data = None
        if X_val is not None and y_val is not None:
            val_data = list(zip(X_val, y_val))
        
        if verbose:
            print(f"Training network for {epochs} epochs...")
            print(f"Training samples: {n}, Batch size: {batch_size}")
        
        for epoch in range(epochs):
            # Shuffle training data
            random.shuffle(training_data)
            
            # Create mini-batches
            mini_batches = [
                training_data[k:k+batch_size] 
                for k in range(0, n, batch_size)
            ]
            
            # Train on each mini-batch
            for mini_batch in mini_batches:
                self._forward_backward(mini_batch)
                self._update_params(mini_batch, learning_rate)
            
            # Evaluate and print progress
            if verbose:
                train_acc = self._evaluate_accuracy(training_data)
                train_pct = train_acc * 100
                
                if val_data:
                    val_acc = self._evaluate_accuracy(val_data)
                    val_pct = val_acc * 100
                    print(f"Epoch {epoch+1:2d}/{epochs}: "
                          f"Train acc: {train_pct:.2f}% | "
                          f"Val acc: {val_pct:.2f}%")
                else:
                    print(f"Epoch {epoch+1:2d}/{epochs}: "
                          f"Train acc: {train_pct:.2f}%")
        
        self.is_trained = True
        return self

    def _update_params(self, mini_batch: List[Tuple[np.ndarray, np.ndarray]], learning_rate: float) -> None:
        """Update network parameters."""
        batch_learning_rate = learning_rate / len(mini_batch)
        for layer in self.layers:
            layer.update_params(batch_learning_rate)

    def _evaluate_accuracy(self, data: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """Evaluate accuracy on a dataset."""
        correct = 0
        for x, y in data:
            predictions = self._single_forward(x)
            if np.argmax(predictions) == np.argmax(y):
                correct += 1
        return correct / len(data)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = []
        for x in X:
            output = self._single_forward(x)
            predictions.append(np.argmax(output))
        
        return np.array(predictions)

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test data
            y_test: Test labels (one-hot or integer format)
            
        Returns:
            Tuple of (accuracy, confusion_matrix)
        """
        predictions = self.predict(X_test)
        
        # Handle label format conversion
        if y_test.ndim > 1 and y_test.shape[1] == 10:  # One-hot encoded
            true_labels = decode_one_hot(y_test)
        else:  # Integer labels
            true_labels = y_test.astype(int)
        
        accuracy = calculate_accuracy(predictions, true_labels)
        cm = confusion_matrix(true_labels, predictions)
        
        return accuracy, cm

    def describe(self) -> str:
        """Return network architecture description."""
        layer_descriptions = [layer.describe() for layer in self.layers]
        return f"SequentialNeuralNet(layers={layer_descriptions})" 