import numpy as np
from network import Network
from utils import *

# load the testing set
test_data = test_set()
inputs = test_data[:, 1:]/255
outputs = test_data[:,0]

# net = Network(784, 64, 10)
net = load_model()

accur = accuracy(net, inputs, outputs)
print(f"Accuracy = {accur}%")