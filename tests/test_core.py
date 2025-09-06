"""
Tests for the core Value class and automatic differentiation.
"""

import pytest
from napagrad import Value


class TestValue:
    """Test cases for the Value class."""
    
    def test_value_creation(self):
        """Test basic Value creation."""
        v = Value(5.0, label="test")
        assert v.value == 5.0
        assert v.gradient == 0.0
        assert v.label == "test"
    
    def test_addition(self):
        """Test addition operation."""
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        assert c.value == 5.0
        assert c.operation == "+"
    
    def test_multiplication(self):
        """Test multiplication operation."""
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        assert c.value == 6.0
        assert c.operation == "*"
    
    def test_subtraction(self):
        """Test subtraction operation."""
        a = Value(5.0)
        b = Value(3.0)
        c = a - b
        assert c.value == 2.0
        assert c.operation == "-"
    
    def test_power(self):
        """Test power operation."""
        a = Value(2.0)
        b = a.pow(3)
        assert b.value == 8.0
        assert b.operation == "pow"
    
    def test_relu(self):
        """Test ReLU activation."""
        # Positive value
        a = Value(2.0)
        b = a.relu()
        assert b.value == 2.0
        
        # Negative value
        c = Value(-1.0)
        d = c.relu()
        assert d.value == 0.0
    
    def test_tanh(self):
        """Test tanh activation."""
        import math
        a = Value(0.0)
        b = a.tanh()
        assert b.value == 0.0
        
        a = Value(1.0)
        b = a.tanh()
        assert abs(b.value - math.tanh(1.0)) < 1e-10
    
    def test_backpropagation(self):
        """Test basic backpropagation."""
        # Simple computation: (2 + 3) * 4 = 20
        a = Value(2.0)
        b = Value(3.0)
        c = Value(4.0)
        
        d = a + b  # 5
        e = d * c  # 20
        
        # Backward pass
        e.zero_grad()
        e.backward()
        
        # Check gradients
        assert e.gradient == 1.0
        assert d.gradient == 4.0  # derivative of d*4 with respect to d
        assert c.gradient == 5.0  # derivative of 5*c with respect to c
        assert a.gradient == 4.0  # derivative flows through addition
        assert b.gradient == 4.0  # derivative flows through addition
    
    def test_scalar_operations(self):
        """Test operations with scalar values."""
        a = Value(2.0)
        b = a + 3.0  # Should convert 3.0 to Value(3.0)
        assert b.value == 5.0
        
        c = a * 4.0
        assert c.value == 8.0
        
        d = a - 1.0
        assert d.value == 1.0


if __name__ == "__main__":
    pytest.main([__file__])
