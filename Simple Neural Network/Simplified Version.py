import numpy as np
#Defines Sigma
def sigmoid(x):
	return 1 / (1 + np.exp(-x))
#Defines Sigma Derivative
def sigmoid_derivative(x):
	return x * (1 - x)
#Inputs
training_inputs = np.array([[0,0,1],
						    [1,1,1],
						    [1,0,1],
						    [0,1,1]])
#Outputs
training_outputs = np.array([[0,1,1,0]]).T

np.random.seed(1)
#Random Weights
synaptic_weights = 2 * np.random.random((3, 1)) - 1

print('Random starting synaptic weights: ')
print(synaptic_weights)
#How many times it iterates through the neuron
for iteration in range(100000):
	#Inputs
	input_layer = training_inputs
	#Outputs = Sum(Inputs & Weights)
	outputs = sigmoid(np.dot(input_layer, synaptic_weights))
	#Calculates difference between training output and real output
	error = training_outputs - outputs
	#Adjusts weights depending if there's a margin of error
	adjustments = error * sigmoid_derivative(outputs)
	#Adjusted weights = inputs + adjustments
	synaptic_weights += np.dot(input_layer.T, adjustments)

print('\n Synaptic Weights after training: ')
print(synaptic_weights)

print('\nOutputs after training: ')
print(outputs)