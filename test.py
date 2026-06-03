import numpy as np
from network import Network
from utils import *

def test_accuracy(net):
	# load the testing set
	test_data = test_set()
	inputs = test_data[:, 1:]/255
	outputs = test_data[:,0]

	accur = accuracy(net, inputs, outputs)
	return accur

if __name__ == "__main__":
	# net = Network(784, 64, 10)
	net = load_model()
	accur = test_accuracy(net)
	print(f"Accuracy = {accur}%")