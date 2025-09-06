"""
Napagrad: A simple neural network framework with automatic differentiation.

This package provides the core components for building and training neural networks
from scratch, including automatic differentiation and backpropagation.
"""

from .core import Value
from .neural_network import Neuron, Layer, MLP
from .visualization import draw_computational_graph

__version__ = "0.1.0"
__author__ = "Napagrad Team"

__all__ = [
    "Value",
    "Neuron", 
    "Layer",
    "MLP",
    "draw_computational_graph"
]
