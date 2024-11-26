# prompt: Buatkan codingan neural network

import numpy as np

# Define the sigmoid activation function
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
  return x * (1 - x)

# Input dataset
inputs = np.array([[0, 0, 1],
                   [0, 1, 1],
                   [1, 0, 1],
                   [1, 1, 1]])

# Output dataset
outputs = np.array([[0],
                    [1],
                    [1],
                    [0]])

# Seed the random number generator for reproducibility
np.random.seed(1)

# Initialize weights with random values
synaptic_weights = 2 * np.random.random((3, 1)) - 1

# Training loop
for iteration in range(10000):
  # Forward propagation
  input_layer = inputs
  outputs_predicted = sigmoid(np.dot(input_layer, synaptic_weights))

  # Calculate the error
  error = outputs - outputs_predicted

  # Backpropagation
  adjustment = error * sigmoid_derivative(outputs_predicted)

  # Update the weights
  synaptic_weights += np.dot(input_layer.T, adjustment)

# Print the trained weights
print("Weights after training:")
print(synaptic_weights)

# Test the neural network
print("Output after training:")
outputs_predicted