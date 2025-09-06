# 🧠 Simple Neural Network from Scratch

A minimal neural network implementation to understand how AI works under the hood.

## What is this?

This is a **neural network** built from scratch in Python - like building a mini brain that can learn! It's designed to be simple and educational, showing you exactly how neural networks work.

## 🏗️ How it works

### The Building Blocks

**Value Class** - The basic unit
- Like a single number that remembers how it was calculated
- Can do math: add, multiply, etc.
- Tracks its "gradient" (how much it should change to improve)

**Neuron** - A mini calculator
- Takes multiple inputs
- Multiplies each by a "weight" (importance)
- Adds a "bias" (offset)
- Outputs one number

**MLP** - The full network
- Multiple layers of neurons stacked together
- Like a factory assembly line for processing information

### The Learning Process

1. **Forward Pass**: Information flows through the network
2. **Calculate Loss**: Compare prediction with correct answer
3. **Backward Pass**: Figure out what went wrong (backpropagation)
4. **Update**: Adjust all the weights and biases

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install graphviz
```

### 2. Run the examples
```bash
# Simple example
python examples/simple_example.py

# House price prediction with visualization
python examples/house_price_prediction.py
```

### 3. Create your own network
```python
from core import Value
from neural_network import MLP
from visualization import draw_computational_graph

# Create a network: 2 inputs → 4 hidden → 1 output
model = MLP([2, 4, 1])

# Give it some data
inputs = [Value(1.0), Value(2.0)]
prediction = model(inputs)

# Check how wrong it was
target = Value(3.0)
loss = (prediction - target).pow(2)

# Learn from the mistake
loss.zero_grad()  # Reset gradients
loss.backward()   # Calculate what to change

# Update the network
for param in model.parameters():
    param.value -= 0.01 * param.gradient

# Visualize the network
dot = draw_computational_graph(loss)
dot.render('my_network', format='svg')
```

## 📁 Files

- **`core.py`** - The Value class (automatic differentiation)
- **`neural_network.py`** - Neurons, layers, and MLP
- **`visualization.py`** - Create visual graphs of your network
- **`examples/`** - Working examples to learn from

## 💡 Key Concepts

- **Weights**: How important each input is
- **Bias**: A constant offset
- **Gradient**: How much to change each weight
- **Backpropagation**: The algorithm that figures out gradients
- **Loss**: How wrong the prediction was

## 🎓 Why build from scratch?

Building this from scratch helps you understand:
- How AI actually works under the hood
- Why neural networks can learn
- What happens during training
- How to debug and improve models

It's like learning to cook by making everything from basic ingredients!

---

*This is a simplified version for learning. Real neural networks have many more features, but the core ideas are the same!*