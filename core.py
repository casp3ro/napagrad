"""
Core Value class for automatic differentiation.

This module contains the fundamental Value class that enables automatic differentiation
and backpropagation through computational graphs.
"""

import math


class Value:
    """
    A node in a computational graph that stores a scalar value and its gradient.
    
    This is the fundamental building block for automatic differentiation.
    Each Value object represents a scalar value and tracks how it was computed,
    enabling automatic computation of gradients through backpropagation.
    """

    def __init__(self, value, children=(), operation='', label=''):
        """
        Initialize a Value node.
        
        Args:
            value: The scalar value
            children: Tuple of Value objects that this node depends on
            operation: String describing the operation that created this node
            label: Optional label for visualization
        """
        self.value = value  # The actual numerical value
        self.gradient = 0.0  # Gradient for backpropagation
        self.children = set(children)  # Previous nodes in the computation
        self.operation = operation  # The operation that created this node
        self.label = label  # Optional label for visualization

    def __repr__(self):
        return f"Value(value={self.value}, label={self.label})"
    
    def __add__(self, other):
        """Addition operation with automatic differentiation."""
        other_value = other if isinstance(other, Value) else Value(other)
        result = Value(
            value=self.value + other_value.value,
            children=(self, other_value),
            operation="+"
        )
        return result

    def __mul__(self, other):
        """Multiplication operation with automatic differentiation."""
        other_value = other if isinstance(other, Value) else Value(other)
        result = Value(
            value=self.value * other_value.value,
            children=(self, other_value),
            operation="*"
        )
        return result

    def __sub__(self, other):
        """Subtraction operation with automatic differentiation."""
        other_value = other if isinstance(other, Value) else Value(other)
        result = Value(
            value=self.value - other_value.value,
            children=(self, other_value),
            operation="-"
        )
        return result

    def  __rmul__(self,other):
        print('__rmul__')
        return self * other

    def relu(self):
        """Rectified Linear Unit activation function."""
        result = Value(
            value=0 if self.value < 0 else self.value,
            children=(self,),
            operation="relu"
        )
        return result
    
    def backward(self):
        """
        Initialize gradient for backpropagation and compute gradients.
        
        This method performs automatic differentiation by:
        1. Topologically sorting the computational graph
        2. Initializing the output gradient to 1.0
        3. Backpropagating gradients through all operations
        """
        # Topological sort to order nodes for backpropagation
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.children:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        # Initialize gradient of output node
        self.gradient = 1.0
        
        # Backpropagate through the graph
        for node in reversed(topo):
            if node.operation:
                if node.operation == "+":
                    # Gradient flows equally to both children
                    for child in node.children:
                        child.gradient += node.gradient
                elif node.operation == "-":
                    # Subtraction: d/dx(a-b) = da - db
                    children = list(node.children)
                    if len(children) == 2:
                        a, b = children
                        a.gradient += node.gradient
                        b.gradient -= node.gradient
                elif node.operation == "*":
                    # Product rule: d/dx(ab) = b*da + a*db
                    children = list(node.children)
                    if len(children) == 2:
                        a, b = children
                        a.gradient += b.value * node.gradient
                        b.gradient += a.value * node.gradient
                elif node.operation == "relu":
                    # ReLU derivative: 1 if x > 0, else 0
                    child = list(node.children)[0]
                    child.gradient += (1.0 if child.value > 0 else 0.0) * node.gradient
                elif node.operation == "pow":
                    # Power rule: d/dx(x^n) = n*x^(n-1)
                    child, exponent = list(node.children)
                    child.gradient += exponent.value * (child.value ** (exponent.value - 1)) * node.gradient
                elif node.operation == "exp":
                    # Exponential derivative: d/dx(e^x) = e^x
                    child = list(node.children)[0]
                    child.gradient += node.value * node.gradient
                elif node.operation == "tanh":
                    # Tanh derivative: d/dx(tanh(x)) = 1 - tanh²(x)
                    child = list(node.children)[0]
                    child.gradient += (1 - node.value**2) * node.gradient

    def zero_grad(self):
        """Reset all gradients to zero."""
        def reset_gradients(v):
            v.gradient = 0.0
            for child in v.children:
                reset_gradients(child)
        reset_gradients(self)

    def pow(self, exponent):
        """Power operation: self^exponent."""
        exponent_value = exponent if isinstance(exponent, Value) else Value(exponent)
        result = Value(
            value=self.value ** exponent_value.value,
            children=(self, exponent_value),
            operation="pow"
        )
        return result

    def exp(self):
        """Exponential function: e^self."""
        result = Value(
            value=math.exp(self.value),
            children=(self,),
            operation="exp"
        )
        return result

    def tanh(self):
        """Hyperbolic tangent activation function."""
        result = Value(
            value=math.tanh(self.value),
            children=(self,),
            operation="tanh"
        )
        return result
