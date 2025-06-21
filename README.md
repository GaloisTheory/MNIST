# MNIST Classification Project

They say you can't start learning about neural nets without trying this - so here we go.....

A comprehensive implementation of multiple machine learning approaches for MNIST digit classification, ranging from simple baseline methods to modern neural networks. This project demonstrates different levels of complexity and performance trade-offs in machine learning.

## 🚀 Quick Start

```bash
# Clone and navigate to repository
git clone <your-repo-url>
cd MNIST

# Install dependencies
pip install -r requirements.txt

# Run the comprehensive summary notebook
jupyter notebook notebooks/mnist_summary.ipynb

# Or run tests
python -m pytest tests/ -v
```

## 📊 Project Overview

This project implements three different approaches to MNIST digit classification:

1. **Baseline Classifier** - Average pixel values + cosine similarity
2. **Sequential Neural Network** - From-scratch implementation using NumPy
3. **PyTorch Neural Network** - Modern deep learning implementation

Each approach demonstrates different aspects of machine learning, from simple statistical methods to modern deep learning frameworks.

## 🏗️ Project Structure

```
.
├── data/                     # MNIST data (auto-downloaded)
├── notebooks/
│   ├── mnist_summary.ipynb  # Comprehensive comparison notebook
│   └── (original notebooks) # Historical implementations
├── src/
│   ├── __init__.py
│   ├── baseline.py          # Cosine similarity classifier
│   ├── datasets.py          # Data loading utilities
│   ├── sequential_nn.py     # From-scratch neural network
│   ├── torch_nn.py         # PyTorch implementation
│   └── utils.py            # Common utilities
├── requirements.txt        # Dependencies
├── .flake8               # Code style configuration
└── README.md
```

## 🎯 Models and Performance

### Expected Performance Metrics

| Model | Training Time | Test Accuracy | Description |
|-------|---------------|---------------|-------------|
| Baseline | ~10 seconds | ~75% | Average pixel + cosine similarity |
| Sequential NN | ~2 minutes | ~89% | From-scratch implementation |
| PyTorch NN | ~1 minute | ~97% | Modern deep learning |

### Model Details

#### 1. Baseline Classifier (`BaselineClassifier`)
- **Method**: Computes average pixel values (centroids) for each digit class
- **Classification**: Uses cosine similarity to nearest centroid
- **Pros**: Fast, interpretable, no hyperparameters
- **Cons**: Limited accuracy, assumes linear separability

#### 2. Sequential Neural Network (`SequentialNNClassifier`)
- **Architecture**: 784 → 392 → 196 → 10 (ReLU activation)
- **Implementation**: Pure NumPy with modular layer design
- **Features**: Mini-batch gradient descent, cross-entropy loss
- **Purpose**: Educational - demonstrates backpropagation from scratch

#### 3. PyTorch Neural Network (`TorchNNClassifier`)
- **Architecture**: Same as sequential (784 → 392 → 196 → 10)
- **Features**: GPU support, modern optimizers, batch normalization
- **Pros**: Production-ready, highly optimized, extensible

## 💻 Usage Examples

### Basic Usage

```python
from src.datasets import load_mnist_data
from src.baseline import BaselineClassifier

# Load data
X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_data()

# Train baseline classifier
model = BaselineClassifier()
model.train(X_train, y_train)

# Evaluate
accuracy, cm = model.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")
```

### Training Neural Networks

```python
from src.sequential_nn import SequentialNNClassifier, DenseLayer, ActivationLayer

# Create and train sequential network
model = SequentialNNClassifier()
model.add(DenseLayer(784, 392))
model.add(ActivationLayer(392, 'relu'))
model.add(DenseLayer(392, 196))
model.add(ActivationLayer(196, 'relu'))
model.add(DenseLayer(196, 10))
model.add(ActivationLayer(10, 'softmax'))

model.train(X_train, y_train, epochs=10, batch_size=100)
```

### PyTorch Implementation

```python
from src.torch_nn import TorchNNClassifier

# Train PyTorch model
model = TorchNNClassifier()
model.train(X_train, y_train, epochs=10, batch_size=64)

# Save trained model
model.save_model('trained_model.pth')
```

## 🧪 Testing

Run the test suite to verify implementations:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_baseline.py -v

# Run with coverage
python -m pytest tests/ --cov=src
```

## 📈 Reproducing Results

All models use fixed random seeds (42) for reproducible results. The `notebooks/mnist_summary.ipynb` notebook provides a comprehensive comparison with:

- Side-by-side accuracy comparisons
- Confusion matrices for each model
- Sample misclassification analysis
- Training time comparisons
- Visualization of learned features

## 🛠️ Development

### Code Style

This project follows PEP 8 with some modifications:
- Max line length: 100 characters
- Uses type hints throughout
- Comprehensive docstrings with mathematical explanations

Check code style:
```bash
flake8 src/ tests/
```

### Adding New Models

To add a new classifier:

1. Implement the model in `src/your_model.py`
2. Follow the interface pattern: `train()`, `predict()`, `evaluate()`
3. Add tests in `tests/test_your_model.py`
4. Update the summary notebook

## 📚 Mathematical Background

### Baseline Classifier
- **Cosine Similarity**: `cos(θ) = (a·b) / (||a|| ||b||)`
- **Classification**: `argmax_i cos(x, centroid_i)`

### Neural Networks
- **Forward Pass**: `y = σ(Wx + b)`
- **Backpropagation**: `∂L/∂W = δ * x^T`
- **Cross-Entropy Loss**: `L = -Σ y_i log(ŷ_i)`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Follow code style guidelines
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- MNIST dataset: Yann LeCun et al.
- Educational inspiration from various deep learning courses
- PyTorch team for the excellent framework

---

**Note**: Training times may vary based on hardware. GPU acceleration available for PyTorch implementation.