import numpy as np
from network import Network
from utils import *
from test import test_accuracy

network = Network(784, 64, 64, 10, lr=1)

# load the training set
train_data = train_set()

# shuffle the dataset
# np.random.shuffle(train_data)
inputs = train_data[:, 1:]/255
outputs = train_data[:,0]

print(f"Size of dataset: {len(train_data)}")
BATCH_SIZE = 500
EPOCH = 200

print(f"Before training, accuracy = {test_accuracy(network)}")

for i in range(EPOCH):
	# np.random.shuffle(train_data)
	curr_inputs = inputs[0:BATCH_SIZE]
	curr_outputs = outputs[0:BATCH_SIZE]

	# dirty trick to get the shape of the np array and initialize to 0
	gradient = network.backprop(curr_inputs[0], num_to_probability(curr_outputs[0]))
	for j in range(len(gradient)):
		gradient[j].weights = np.zeros(gradient[j].weights.shape)
		gradient[j].biases = np.zeros(gradient[j].biases.shape)

	# getting gradients for each data points
	for j in range(BATCH_SIZE):
		output = (curr_outputs[j])
		g = network.backprop(curr_inputs[j], num_to_probability(curr_outputs[j]))
		# add the g to gradient
		for k in range(len(gradient)):
			gradient[k].weights += g[k].weights
			gradient[k].biases += g[k].biases
	
	# get the mean from the total sum
	for j in range(len(gradient)):
		gradient[j].weights /= BATCH_SIZE
		gradient[j].biases /= BATCH_SIZE

	# finally descent with the gradient
	# n = Network(*network.design)
	# n.layers = gradient
	if i == 199:
		print(gradient[1].weights)
	network.descent(gradient)

	# check accuracy after each epoch
	print(f"Epoch {i+1}: {test_accuracy(network) :.2f}%")

print(f"After training, accuracy = {test_accuracy(network)}")
save_model(network)