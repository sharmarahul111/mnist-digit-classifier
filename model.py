import numpy as np
from network import Network
from utils import *
from test import test_accuracy

network = Network(784, 64, 64, 10, lr=1)

# load the training set
test_data = train_set()

# shuffle the dataset
np.random.shuffle(test_data)
inputs = test_data[:, 1:]/255
outputs = test_data[:,0]

print(f"Size of dataset: {len(test_data)}")
BATCH_SIZE = 2000
EPOCH = 30

print(f"Before training, accuracy = {test_accuracy(network)}")

for i in range(EPOCH):
	curr_inputs = inputs[BATCH_SIZE*i:BATCH_SIZE*(i+1)]
	curr_outputs = outputs[BATCH_SIZE*i:BATCH_SIZE*(i+1)]

	# dirty trick to get the shape of the np array and initialize to 0
	gradient = network.backprop(curr_inputs[0], num_to_probability(curr_outputs[0]))
	for j in range(len(gradient)):
		gradient[j].weights = np.zeros(gradient[j].weights.shape)
		gradient[j].biases = np.zeros(gradient[j].biases.shape)

	# getting gradients for each data points
	for j in range(BATCH_SIZE):
		g = network.backprop(curr_inputs[j], num_to_probability(curr_outputs[j]))
		# add the g to gradient
		for j in range(len(gradient)):
			gradient[j].weights += g[j].weights
			gradient[j].biases += g[j].biases
	
	# get the mean from the total sum
	for j in range(len(gradient)):
		gradient[j].weights /= BATCH_SIZE
		gradient[j].biases /= BATCH_SIZE

	# finally descent with the gradient
	# n = Network(*network.design)
	# n.layers = gradient
	# print(n)
	network.descent(gradient)

	# check accuracy after each epoch
	print(f"Epoch {i+1}: {test_accuracy(network)}")

print(f"After training, accuracy = {test_accuracy(network)}")
