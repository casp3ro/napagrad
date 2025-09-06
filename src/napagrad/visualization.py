"""
Visualization utilities for computational graphs.

This module provides functions to visualize the computational graphs
created during neural network forward and backward passes.
"""

from graphviz import Digraph
from .core import Value


def trace_computational_graph(root):
    """
    Trace the computational graph starting from the root node.
    
    Args:
        root: The root Value node to start tracing from
        
    Returns:
        tuple: (nodes, edges) sets containing all nodes and edges in the graph
    """
    nodes, edges = set(), set()
    
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v.children:
                edges.add((child, v))
                build(child)
    
    build(root)
    return nodes, edges


def draw_computational_graph(root, format='svg', rankdir='LR', filename=None):
    """
    Create a visual representation of the computational graph.
    
    Args:
        root: The root Value node to visualize
        format: Output format ('svg', 'png', 'pdf', etc.)
        rankdir: Graph direction ('LR' for left-to-right, 'TB' for top-to-bottom)
        filename: Optional filename to save the graph (without extension)
        
    Returns:
        Digraph: The graphviz Digraph object
        
    Raises:
        AssertionError: If rankdir is not 'LR' or 'TB'
    """
    assert rankdir in ['LR', 'TB'], "rankdir must be 'LR' or 'TB'"
    
    nodes, edges = trace_computational_graph(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})
    
    # Add nodes to the graph
    for node in nodes:
        # Create the main value node
        dot.node(
            name=str(id(node)), 
            label=f"{{ value {node.value:.4f} | gradient {node.gradient:.4f} }}", 
            shape='record'
        )
        
        # Add operation node if this node was created by an operation
        if node.operation:
            dot.node(
                name=str(id(node)) + node.operation, 
                label=node.operation
            )
            dot.edge(str(id(node)) + node.operation, str(id(node)))
    
    # Add edges between nodes
    for parent, child in edges:
        if child.operation:
            dot.edge(str(id(parent)), str(id(child)) + child.operation)
        else:
            dot.edge(str(id(parent)), str(id(child)))
    
    # Save the graph if filename is provided
    if filename:
        dot.render(filename, format=format, cleanup=True)
    
    return dot


# Backward compatibility alias
draw_dot = draw_computational_graph
