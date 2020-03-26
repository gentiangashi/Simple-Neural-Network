import numpy as np

class NeuralNetwork():

	def __init__(self):
		# Seed the random number generator, so it generates the same numbers
		# Every time the program runs.
		np.random.seed(1)
		# Model a single neuron, with 3 input connections and 1 output connection.
		# We assign random weights to a 3x1 matrix, with values in the range -1 to 1
		# and mean 0.
		self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

	# The sigmoid function, which describes an S shaped curve.
	# We pass the weighted sum of the inputs through this function to
	# normalise them between 0 and 1.
	def sigmoid(self, x):
			return 1 / (1 + np.exp(-x))

	# The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
	def sigmoid_derivative(self, x):
			return x * (1 - x)

	# We train the neural network through a process of trial and error. 
	# Adjusting the synaptic weights each time.
	def train(self, training_inputs, training_outputs, training_iterations):

		for iteration in range(training_iterations):
			# Pass the training set through our neural network (a single neuron).
			output = self.think(training_inputs)

			# Calculate the error (The difference between the desired output
            # and the predicted output).
			error = training_outputs - output

			# Multiply the error by the input and again by the gradient of the Sigmoid curve.
            # This means less confident weights are adjusted more.
            # This means inputs, which are zero, do not cause changes to the weights.
			adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))

			# Weights get adjusted
			self.synaptic_weights += adjustments

	# The neural network thinks.
	def think(self, inputs):
		# Pass inputs through our neural network (our single neuron).
		inputs = inputs.astype(float)
		output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
		return output

if __name__ == "__main__":

	#Intialise a single neuron neural network.
	neural_network = NeuralNetwork()

	print("Random Synaptic Weights: ")
	print(neural_network.synaptic_weights)

	# The training set. We have 4 examples, each consisting of 3 input values
    # and 1 output value.
	training_inputs = np.array([[0,0,1],
                                [1,1,1],
                                [1,0,1],
                                [0,1,1]])

	training_outputs = np.array([[0,1,1,0]]).T

	# Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
	neural_network.train(training_inputs, training_outputs, 10000)

	print("Synaptic Weights After Training: ")
	print(neural_network.synaptic_weights)

	# Test the neural network with a new situation.
	Input1 = str(input("Input 1: "))
	Input2 = str(input("Input 2: "))
	Input3 = str(input("Input 3: "))

	print("New Situation: Input Data = ", Input1, Input2, Input3)
	print("Output Data: ")
	print(neural_network.think(np.array([Input1, Input2, Input3])))
