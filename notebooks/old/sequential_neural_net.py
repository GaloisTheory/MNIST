import numpy as np 
import random 
from .activation import sigmoid, sigmoid_prime, relu, relu_prime, softmax, softmax_prime
from .error_func import MSE, CrossEntropy


class Layer: 
    def __init__(self):
        self.params = {}

        self.previous = None
        self.next = None

        self.input_data = None
        self.output_data = None

        self.input_delta = None
        self.output_delta = None
    
    def connect(self, layer): 
        self.previous = layer
        layer.next = self

    def forward(self, input_data): 
        raise NotImplementedError("Subclasses must implement forward")

    def get_forward_input(self): 
        if self.previous is not None: 
            return self.previous.output_data
        else: 
            return self.input_data
    
    def backward(self): 
        raise NotImplementedError("Subclasses must implement backward")
    
    def get_backward_input(self): 
        if self.next is not None: 
            return self.next.output_delta
        else: 
            return self.input_delta
        
    def clear_deltas(self): 
        pass

    def update_params(self, learning_rate): 
        pass

    def describe(self): 
        raise NotImplementedError("Subclasses must implement describe")


class ActivationLayer(Layer): 
    def __init__(self, input_dim, activation_type='sigmoid'): 
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.activation_type = activation_type

    def forward(self, input_data=None): 
        data = self.get_forward_input()
        if self.activation_type == 'sigmoid':
            self.output_data = sigmoid(data)
        elif self.activation_type == 'relu':
            self.output_data = relu(data)
        elif self.activation_type == 'softmax':
            self.output_data = softmax(data)
        else:
            raise ValueError(f"Unsupported activation type: {self.activation_type}")

    def backward(self): 
        delta = self.get_backward_input()
        data = self.get_forward_input()
        
        if self.activation_type == 'sigmoid':
            self.output_delta = delta * sigmoid_prime(data)
        elif self.activation_type == 'relu':
            self.output_delta = delta * relu_prime(data)
        elif self.activation_type == 'softmax':
            # For softmax with cross-entropy, the gradient simplifies
            # This assumes we're using softmax with cross-entropy loss
            self.output_delta = delta
        else:
            raise ValueError(f"Unsupported activation type: {self.activation_type}")
        
        # Ensure output_delta is properly shaped
        if self.output_delta is not None and hasattr(self.output_delta, 'squeeze'):
            self.output_delta = self.output_delta.squeeze()

    def describe(self): 
        return f"ActivationLayer(input_dim={self.input_dim}, output_dim={self.output_dim}, activation={self.activation_type})"
    

class DenseLayer(Layer): 
    def __init__(self, input_dim, output_dim): 
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.weights = np.random.randn(output_dim, input_dim)
        self.bias = np.random.randn(output_dim)

        self.delta_b = None
        self.delta_w = None

    def forward(self, input_data=None): 
        data = self.get_forward_input()
        self.output_data = np.dot(self.weights, data) + self.bias

    def backward(self): 
        delta = self.get_backward_input()
        data = self.get_forward_input()

        if self.delta_b is None: 
            self.delta_b = delta
        else: 
            self.delta_b += delta
        
        if self.delta_w is None: 
            self.delta_w = np.outer(delta, data)
        else: 
            self.delta_w += np.outer(delta, data)

        self.output_delta = np.dot(self.weights.T, delta)

    def update_params(self, learning_rate): 
        self.weights -= learning_rate * self.delta_w
        self.bias -= learning_rate * self.delta_b

    def clear_deltas(self): 
        self.delta_b = np.zeros_like(self.bias)
        self.delta_w = np.zeros_like(self.weights)

    def describe(self): 
        return f"DenseLayer(input_dim={self.input_dim}, output_dim={self.output_dim})"
    

class SequentialNeuralNet: 
    def __init__(self, loss=None): 
        self.layers = []
        if loss is None: 
            self.loss = CrossEntropy()

    def add(self, layer): 
        self.layers.append(layer)
        if len(self.layers) > 1: 
            layer.connect(self.layers[-2])

    def train(self, X_train, y_train, epochs, mini_batch_size, learning_rate, X_test=None, y_test=None): 
        # Create training data by zipping X_train and y_train
        training_data = list(zip(X_train, y_train))
        n = len(training_data)
        
        # Create test data if provided
        test_data = None
        if X_test is not None and y_test is not None:
            test_data = list(zip(X_test, y_test))
        
        for j in range(epochs): 
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches: 
                self.train_batch(mini_batch, learning_rate)
            
            # Evaluate on training data
            train_accuracy = self.evaluate(training_data)
            train_total = len(training_data)
            train_percent = (train_accuracy / train_total) * 100
            
            if test_data: 
                # Evaluate on test data
                test_accuracy = self.evaluate(test_data)
                test_total = len(test_data)
                test_percent = (test_accuracy / test_total) * 100
                print(f"Epoch {j}: Training accuracy: {train_accuracy}/{train_total} ({train_percent:.2f}%) | Test accuracy: {test_accuracy}/{test_total} ({test_percent:.2f}%)")
            else: 
                print(f"Epoch {j}: Training accuracy: {train_accuracy}/{train_total} ({train_percent:.2f}%)")
        
        # Final evaluation summary
        if test_data:
            print("\n" + "="*50)
            print("FINAL RESULTS:")
            final_train_accuracy = self.evaluate(training_data)
            final_test_accuracy = self.evaluate(test_data)
            final_train_percent = (final_train_accuracy / len(training_data)) * 100
            final_test_percent = (final_test_accuracy / len(test_data)) * 100
            print(f"Final Training Accuracy: {final_train_accuracy}/{len(training_data)} ({final_train_percent:.2f}%)")
            print(f"Final Test Accuracy: {final_test_accuracy}/{len(test_data)} ({final_test_percent:.2f}%)")
            print("="*50)

    def train_batch(self, mini_batch, learning_rate): 
        self.forward_backward(mini_batch)
        self.update(mini_batch, learning_rate)
    

    def update(self, mini_batch, learning_rate): 
        learning_rate = learning_rate / len(mini_batch)
        for layer in self.layers: 
            layer.update_params(learning_rate)
        for layer in self.layers: 
            layer.clear_deltas()

    def forward_backward(self, mini_batch): 
        for x, y in mini_batch: 
            self.layers[0].input_data = x
            for layer in self.layers: 
                layer.forward()
            self.layers[-1].input_delta = self.loss.loss_function_derivative(self.layers[-1].output_data, y.T)
            for layer in reversed(self.layers): 
                layer.backward()
    
    def single_forward(self, x): 
        self.layers[0].input_data = x
        for layer in self.layers: 
            layer.forward()
        return self.layers[-1].output_data
    
    def evaluate(self, test_data): 
        test_results = [(np.argmax(self.single_forward(x)), np.argmax(y)) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    
    def describe(self): 
        return f"SequentialNeuralNet(layers={self.layers})"
    
