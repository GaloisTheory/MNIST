from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np 


def plot_digit(data: np.array) -> None:
    image = data.reshape(28, 28)
    plt.imshow(image, cmap="binary")
    plt.axis("off")


def encode_label(j: str) -> np.array: 
    e = np.zeros((10, 1))
    e[int(j)] = 1.0
    return e


def prepare_data() -> tuple[np.array, np.array, np.array, np.array, np.array, np.array]:
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data, mnist.target
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    y_train = np.array([encode_label(y) for y in y_train])
    y_test = np.array([encode_label(y) for y in y_test])
    y_val = np.array([encode_label(y) for y in y_val])
    
    return X_train, y_train, X_test, y_test, X_val, y_val