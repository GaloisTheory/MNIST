import random
import numpy as np 


class MSE: 
    def __init__(self): 
        pass

    @staticmethod
    def loss_function(predictions, labels): 
        diff = predictions - labels
        return 0.5 * np.sum(diff * diff)

    @staticmethod
    def loss_function_derivative(predictions, labels): 
        return predictions - labels


class CrossEntropy:
    def __init__(self):
        pass
    
    @staticmethod
    def loss_function(predictions, labels):
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        return -np.sum(labels * np.log(predictions))
    
    @staticmethod
    def loss_function_derivative(predictions, labels):
        # When combined with softmax, derivative simplifies to:
        return predictions - labels
    