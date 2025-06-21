import numpy as np 

def sigmoid_double(x): 
    return 1 / (1 + np.exp(-x))

def sigmoid(x): 
    return np.vectorize(sigmoid_double)(x)

def sigmoid_prime_double(x): 
    return sigmoid_double(x) * (1 - sigmoid_double(x))

def sigmoid_prime(x): 
    return np.vectorize(sigmoid_prime_double)(x)

# ReLU activation functions
def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return (x > 0).astype(float)

# Softmax activation function
def softmax(x):
    # Numerical stability: subtract max
    if x.ndim == 1:
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    else:
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)

def softmax_prime(x):
    # For softmax, the derivative is computed differently
    # This is a placeholder - actual derivative is handled in the layer
    s = softmax(x)
    return s * (1 - s)
