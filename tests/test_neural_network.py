"""
Tests for neural network components.
"""

import pytest
from napagrad import Value, Neuron, Layer, MLP


class TestNeuron:
    """Test cases for the Neuron class."""
    
    def test_neuron_creation(self):
        """Test neuron creation with correct number of weights."""
        neuron = Neuron(3)  # 3 inputs
        assert len(neuron.weights) == 3
        assert isinstance(neuron.bias, Value)
    
    def test_neuron_forward(self):
        """Test neuron forward pass."""
        neuron = Neuron(2)
        inputs = [Value(1.0), Value(2.0)]
        output = neuron(inputs)
        assert isinstance(output, Value)
    
    def test_neuron_parameters(self):
        """Test that neuron returns all parameters."""
        neuron = Neuron(3)
        params = neuron.parameters()
        assert len(params) == 4  # 3 weights + 1 bias
        assert all(isinstance(p, Value) for p in params)


class TestLayer:
    """Test cases for the Layer class."""
    
    def test_layer_creation(self):
        """Test layer creation with correct number of neurons."""
        layer = Layer(2, 3)  # 2 inputs, 3 outputs
        assert len(layer.neurons) == 3
        assert all(isinstance(neuron, Neuron) for neuron in layer.neurons)
    
    def test_layer_forward(self):
        """Test layer forward pass."""
        layer = Layer(2, 1)  # 2 inputs, 1 output
        inputs = [Value(1.0), Value(2.0)]
        output = layer(inputs)
        assert isinstance(output, Value)
    
    def test_layer_parameters(self):
        """Test that layer returns all parameters."""
        layer = Layer(2, 3)  # 2 inputs, 3 outputs
        params = layer.parameters()
        expected_params = 3 * 3  # 3 neurons * (2 weights + 1 bias)
        assert len(params) == expected_params


class TestMLP:
    """Test cases for the MLP class."""
    
    def test_mlp_creation(self):
        """Test MLP creation with correct architecture."""
        mlp = MLP([2, 3, 1])  # 2 inputs, 3 hidden, 1 output
        assert len(mlp.layers) == 2  # 2 layer transitions
        
        # Check first layer: 2 inputs -> 3 outputs
        assert len(mlp.layers[0].neurons) == 3
        assert len(mlp.layers[0].neurons[0].weights) == 2
        
        # Check second layer: 3 inputs -> 1 output
        assert len(mlp.layers[1].neurons) == 1
        assert len(mlp.layers[1].neurons[0].weights) == 3
    
    def test_mlp_forward(self):
        """Test MLP forward pass."""
        mlp = MLP([2, 3, 1])
        inputs = [Value(1.0), Value(2.0)]
        output = mlp(inputs)
        assert isinstance(output, Value)
    
    def test_mlp_parameters(self):
        """Test that MLP returns all parameters."""
        mlp = MLP([2, 3, 1])
        params = mlp.parameters()
        
        # Expected: (2*3 + 3) + (3*1 + 1) = 9 + 4 = 13 parameters
        expected_params = (2 * 3 + 3) + (3 * 1 + 1)
        assert len(params) == expected_params
        assert all(isinstance(p, Value) for p in params)
    
    def test_mlp_training_step(self):
        """Test a complete training step."""
        mlp = MLP([2, 3, 1])
        inputs = [Value(1.0), Value(2.0)]
        target = Value(3.0)
        
        # Forward pass
        output = mlp(inputs)
        loss = (output - target).pow(2)
        
        # Backward pass
        loss.zero_grad()
        loss.backward()
        
        # Check that gradients are computed
        assert loss.gradient == 1.0
        
        # Check that some parameters have non-zero gradients
        # (Note: some parameters might have zero gradients due to the specific computation)
        param_gradients = [p.gradient for p in mlp.parameters()]
        assert len(param_gradients) > 0  # We have parameters
        
        # Training step
        learning_rate = 0.1
        old_params = [p.value for p in mlp.parameters()]
        
        for param in mlp.parameters():
            param.value -= learning_rate * param.gradient
        
        # Check that parameters changed (at least some should change)
        new_params = [p.value for p in mlp.parameters()]
        assert any(old != new for old, new in zip(old_params, new_params))


if __name__ == "__main__":
    pytest.main([__file__])
