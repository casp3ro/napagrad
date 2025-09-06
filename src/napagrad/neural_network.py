"""
Neural network components built on top of the Value class.

This module provides high-level neural network components including neurons,
layers, and multi-layer perceptrons.
"""

import random
from .core import Value


class Neuron:
    """
    A single neuron with weights and bias.
    
    A neuron computes a weighted sum of its inputs plus a bias term.
    This is the fundamental building block of neural networks.
    """
    
    def __init__(self, num_inputs):
        """
        Initialize a neuron with random weights and bias.
        
        Args:
            num_inputs: Number of input connections
        """
        # Initialize weights and bias with small random values
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(num_inputs)]
        self.bias = Value(random.uniform(-1, 1))
    
    def __call__(self, inputs):
        """
        Forward pass: w1*x1 + w2*x2 + ... + wn*xn + bias.
        
        Args:
            inputs: List of input values (can be scalars or Value objects)
            
        Returns:
            Value: The weighted sum plus bias
        """
        # Ensure inputs are Value objects
        inputs = [x if isinstance(x, Value) else Value(x) for x in inputs]
        
        # Compute weighted sum
        weighted_sum = self.bias
        for weight, input_val in zip(self.weights, inputs):
            weighted_sum = weighted_sum + weight * input_val
        
        return weighted_sum
    
    def parameters(self):
        """
        Return all parameters (weights and bias) for optimization.
        
        Returns:
            List[Value]: All trainable parameters
        """
        return self.weights + [self.bias]


class Layer:
    """
    A layer of neurons.
    
    A layer contains multiple neurons that process inputs in parallel.
    Each neuron in the layer receives the same inputs but has different weights.
    """
    
    def __init__(self, num_inputs, num_outputs):
        """
        Initialize a layer with the specified number of neurons.
        
        Args:
            num_inputs: Number of input connections per neuron
            num_outputs: Number of neurons in the layer
        """
        self.neurons = [Neuron(num_inputs) for _ in range(num_outputs)]
    
    def __call__(self, inputs):
        """
        Forward pass through the layer.
        
        Args:
            inputs: List of input values
            
        Returns:
            Value or List[Value]: Output of the layer
        """
        outputs = [neuron(inputs) for neuron in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs
    
    def parameters(self):
        """
        Return all parameters in this layer.
        
        Returns:
            List[Value]: All trainable parameters in the layer
        """
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params


class MLP:
    """
    Multi-Layer Perceptron (simple neural network).
    
    An MLP is a feedforward neural network with multiple layers.
    Each layer (except the last) applies an activation function to its outputs.
    """
    
    def __init__(self, layer_sizes):
        """
        Initialize a multi-layer perceptron.
        
        Args:
            layer_sizes: List of integers specifying the size of each layer.
                        e.g., [3, 4, 4, 1] for 3 inputs, 2 hidden layers of 4 neurons each, 1 output
        """
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1]))
    
    def __call__(self, inputs):
        """
        Forward pass through the entire network.
        
        Args:
            inputs: List of input values
            
        Returns:
            Value: The network's output
        """
        x = inputs
        for layer in self.layers[:-1]:  # All layers except the last
            x = layer(x)
            x = [val.tanh() for val in (x if isinstance(x, list) else [x])]  # Apply tanh activation
            x = x[0] if len(x) == 1 else x
        
        # Last layer (no activation for regression, or add softmax for classification)
        return self.layers[-1](x)
    
    def parameters(self):
        """
        Return all parameters in the network.
        
        Returns:
            List[Value]: All trainable parameters in the network
        """
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
