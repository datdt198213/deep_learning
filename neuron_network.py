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

    # Calculate error and delta for output and hidden layer
    def backpropagation(self, inputs, targets, learning_rate):
        # Output layer
        output_error = targets - self.output
        output_delta = output_error * sigmoid_derivative(self.output)

        # Hidden layer
        hidden_error = np.dot(output_delta, self.weights2.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden)

        # Update weights for both layers
        self.weights2 += learning_rate * np.dot(self.hidden.T, output_delta)
        self.weights1 += learning_rate * np.dot(inputs.T, hidden_delta)

    def train(self, inputs, targets, learning_rate, epochs):
        for i in range(epochs):
            output = self.feed_forward(inputs)
            self.backpropagation(inputs, targets, learning_rate)

            if i % 10 == 0:
                loss = np.mean(np.square(targets - output))
                print("Epoch", i, "loss:", loss)


