import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

class NeuralNetwork:
    # Initialize weights randomly
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.weights2 = np.random.randn(hidden_size, output_size)

    # Feed forward through network and return the output
    def feed_forward(self, inputs):
        self.hidden = sigmoid(np.dot(inputs, self.weights1))
        self.output = sigmoid(np.dot(self.hidden, self.weights2))
        return self.output

    def train(self, inputs, epochs):
        for i in range(epochs):
            output = self.feed_forward(inputs)
            if i % 10 == 0:
                print("Epoch", i, "output:", output)


