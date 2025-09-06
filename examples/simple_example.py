import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core import Value
from neural_network import MLP

# Create a simple neural network: 2 inputs -> 4 hidden -> 1 output
model = MLP([2, 4, 1])

# Example input data
x = [Value(1.0), Value(2.0)]

# Forward pass
output = model(x)
print(f"Network output: {output.value}")

# Compute loss (simple squared error for demonstration)
target = Value(3.0)
loss = (output - target).pow(2)
print(f"Loss: {loss.value}")

# Backward pass
loss.zero_grad()  # Reset gradients
loss.backward()

# Print gradients for some parameters
print(f"Gradient of first weight in first neuron: {model.layers[0].neurons[0].weights[0].gradient}")
print(f"Gradient of bias in first neuron: {model.layers[0].neurons[0].bias.gradient}")

# Simple gradient descent step
learning_rate = 0.01
for param in model.parameters():
    param.value -= learning_rate * param.gradient

print(f"After one gradient step - Loss: {loss.value}")
print(f"Network output after update: {output.value}")
