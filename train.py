import numpy as np
from neuron_network import *
# Define inputs and targets
inputs = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
targets = np.array([[0], [1], [1], [0]])
 # Create a neural network with 3 inputs, 4 hidden nodes, and 1 output
nn = NeuralNetwork(3, 4, 1)
 # Train the neural network with a learning rate of 0.1 and 10000 epochs
nn.train(inputs, targets, 0.1, 10000)
 # Test the neural network on some new inputs
test_inputs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]])
print(nn.feed_forward(test_inputs))