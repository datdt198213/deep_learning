import numpy as np

# Define sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigma(param1, param2):
    return np.dot(param1, param2)

# Initialize weights randomly
w1 = np.random.randn(2, 4) # 2 input features, 4 nodes in first hidden layer
w2 = np.random.randn(4, 3) # 4 nodes in first hidden layer, 3 nodes in second hidden layer
w3 = np.random.randn(3, 1) # 3 nodes in second hidden layer, 1 output node

print("w1",w1)
print("w2",w2)
print("w3",w3)

# Define input features
input_features = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])

# Feed forward without backpropagation
for input_feature in input_features:
    hidden_layer1 = sigmoid(sigma(input_feature, w1))
    hidden_layer2 = sigmoid(sigma(input_feature, w2))
    output = sigmoid(sigma(input_feature, w3))
    print(output)