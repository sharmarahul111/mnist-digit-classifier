import numpy as np
class Layer:
	def __init__(self, weights=None, biases=None):
		self.weights = weights
		self.biases = biases

class Network():
	def __init__(self, *args: int, lr=.3):
		self.design = args
		self.layers = []
		self.lr = lr
		for i in range(1,len(args)):
			weights = np.random.randn(args[i-1], args[i])*.5
			biases = np.zeros(args[i])
			self.layers.append(Layer(weights, biases))
	
	def __str__(self):
		s = "\n["
		i=1
		for layer in self.layers:
			s+= f"\n\tlayer[{i}]: "
			# weights
			s+= "\n\t\tWeights ["
			for l in layer.weights:
				s+= "\n\t\t\t"
				s+= str(l)
				s+= " "
			s+= "\n\t\t]"

			# biases
			s+= "\n\t\tBiases ["
			for l in layer.biases:
				s+= "\n\t\t\t"
				s+= str(l)
				s+= " "
			s+= "\n\t\t]"

			i+=1
		s+= "\n]\n"
		return s

	def sigmoid(self, x):
		return 1./(1+np.exp(-x))

	def tanh(self, x):
		return np.tanh(x)

	def forward(self, inp):
		activation = inp
		for i in range(len(self.layers)-1):
			activation = activation @ self.layers[i].weights + self.layers[i].biases
			activation = self.sigmoid(activation)
		activation = activation @ self.layers[-1].weights + self.layers[-1].biases
		activation = self.sigmoid(activation)
		return activation

	def backprop(self, inputs, expected):
		activations = [inputs]
		activation = inputs
		# pre_activations = []
		for i in range(len(self.layers)-1):
			activation = activation @ self.layers[i].weights + self.layers[i].biases
			activation = self.sigmoid(activation)
			activations.append(activation)
		# last layer may decide to use different activation function
		activation = activation @ self.layers[-1].weights + self.layers[-1].biases
		activation = self.sigmoid(activation)
		activations.append(activation)

		# doing the backprop
		output = activations[-1]
		delta = (output - expected) * output * (1-output)
		deltas = [delta]
		for i in range(len(self.layers)-2, -1, -1):
			delta = (delta @ self.layers[i+1].weights.T) * activations[i+1] * (1 - activations[i+1])
			deltas.append(delta)
		deltas.reverse()

		# calculatinge the gradients
		gradient = []
		for i in range(len(self.layers)):
			g = Layer()
			g.weights = np.outer(activations[i], deltas[i])
			g.biases = deltas[i].copy()

			gradient.append(g)
		# return activations, deltas
		return gradient

	def cost(self,inputs, expected):
		error = 0
		for inp, ex in zip(inputs, expected):
			error += .5*np.mean((self.forward(inp)-ex)**2)
		return error/len(inputs)
	
	def finite_diff(self, inp, expected, eps):
		gradient = []
		for layer in self.layers:
			# weights
			g = Layer()
			g.weights = np.zeros(np.shape(layer.weights))
			for i in range(g.weights.shape[0]):
				for j in range(g.weights.shape[1]):
					cost_original = self.cost(inp, expected)
					saved = layer.weights[i][j]
					layer.weights[i][j] += eps
					cost_new = self.cost(inp, expected)
					layer.weights[i][j] = saved
					g.weights[i][j] = (cost_new-cost_original)/eps

			# biases
			g.biases = np.zeros(np.shape(layer.biases))
			for i in range(g.biases.shape[0]):
				cost_original = self.cost(inp, expected)
				saved = layer.biases[i]
				layer.biases[i] += eps
				cost_new = self.cost(inp, expected)
				layer.biases[i] = saved
				g.biases[i] = (cost_new-cost_original)/eps

			gradient.append(g)
		return gradient

	def descent(self, gradient, rate=1e-3):
		for layer, g in zip(self.layers, gradient):
			layer.weights -= g.weights*rate
			layer.biases -= g.biases*rate

	def mutate(self, diversity=.1):
		net = Network(*self.design)
		for i in range(len(self.layers)):
			net.layers[i].weights = self.layers[i].weights + np.random.randn(*self.layers[i].weights.shape)*diversity
			net.layers[i].biases = self.layers[i].biases + np.random.randn(*self.layers[i].biases.shape)*diversity
		return net