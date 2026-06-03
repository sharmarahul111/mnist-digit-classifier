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
