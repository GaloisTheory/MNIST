"""
MNIST Classification Package

A collection of machine learning models for MNIST digit classification,
including baseline methods and neural network implementations.
"""

__author__ = "Dohun Lee"

from .baseline import BaselineClassifier
from .sequential_nn import SequentialNNClassifier  
from .torch_nn import TorchNNClassifier

__all__ = [
    "BaselineClassifier",
    "SequentialNNClassifier", 
    "TorchNNClassifier",
] 