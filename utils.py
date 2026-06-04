import numpy as np
from network import Network

def train_set():
	return np.loadtxt('data/mnist_train.csv', delimiter=',')

def test_set():
	return np.loadtxt('data/mnist_test.csv', delimiter=',')

def accuracy(network, inputs, targets):
	outputs = np.array([network.forward(inp) for inp in inputs])
	predictions = np.argmax(outputs, axis=1)
	accur = 100 * np.mean(predictions == targets)
	return accur

def num_to_probability(num):
	x = np.zeros(10)
	x[int(num)] = 0.99
	return x

# save.py
import pickle

def save_model(model, filename="model.pkl"):
	with open(filename, "wb") as f:
		pickle.dump(model, f)


def load_model(filename="model.pkl"):
	with open(filename, "rb") as f:
		return pickle.load(f)