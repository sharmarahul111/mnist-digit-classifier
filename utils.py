import numpy as np
from network import Network

def train_set():
	return np.loadtxt('data/mnist_train.csv', delimiter=',')

def test_set():
	return np.loadtxt('data/mnist_test.csv', delimiter=',')

def accuracy(network, inputs, targets):
    outputs = np.array([network.forward(inp) for inp in inputs])
    predictions = np.argmax(outputs, axis=1)
    return 100 * np.mean(predictions == targets)

# save.py
import pickle

def save_model(model, filename="model.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(model, f)


def load_model(filename="model.pkl"):
    with open(filename, "rb") as f:
        return pickle.load(f)