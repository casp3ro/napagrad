# 🧠 Simple Neural Network from Scratch

## What is this?

This is a **neural network** built from scratch in Python - like building a mini brain that can learn! Think of it as a very simple version of what powers AI systems like ChatGPT, but much smaller and easier to understand.

## 🤔 What does a neural network do?

Imagine you want to teach a computer to recognize if a picture shows a cat or a dog. A neural network is like a very smart calculator that:

1. **Takes in information** (like numbers describing a picture)
2. **Processes it through layers** (like different parts of a brain)
3. **Makes a prediction** (like "this is probably a cat")
4. **Learns from mistakes** (gets better over time)

## 🏗️ How does it work?

### The Building Blocks

**1. Value Class** - The basic unit
- Like a single number that remembers how it was calculated
- Can do math: add, multiply, etc.
- Tracks its "gradient" (how much it should change to improve)

**2. Neuron** - A mini calculator
- Takes multiple inputs (like 2 numbers)
- Multiplies each by a "weight" (importance)
- Adds a "bias" (offset)
- Outputs one number

**3. Layer** - A group of neurons
- Like a row of mini calculators working together

**4. MLP (Multi-Layer Perceptron)** - The full network
- Multiple layers stacked together
- Like a factory assembly line for processing information

### The Learning Process

1. **Forward Pass**: Information flows through the network
   ```
   Input → Layer 1 → Layer 2 → Output
   ```

2. **Calculate Loss**: Compare prediction with correct answer
   ```
   Loss = (Prediction - Correct Answer)²
   ```

3. **Backward Pass**: Figure out what went wrong
   - Calculate how much each number should change
   - This is called "backpropagation"

4. **Update**: Adjust all the weights and biases
   - Make small changes to get better next time

## 🚀 How to use it

### Simple Example
```python
from napagrad import Value, MLP

# Create a network: 2 inputs → 4 hidden neurons → 1 output
model = MLP([2, 4, 1])

# Give it some data
input_data = [Value(1.0), Value(2.0)]
prediction = model(input_data)

# Check how wrong it was
target = Value(3.0)
loss = (prediction - target).pow(2)

# Learn from the mistake
loss.zero_grad()  # Reset
loss.backward()   # Calculate what to change

# Update the network
for param in model.parameters():
    param.value -= 0.01 * param.gradient
```

### Visualizing the Network
```python
# See how the network looks
from napagrad import draw_computational_graph
dot = draw_computational_graph(prediction)
dot.render('network_graph', format='svg')
```

## 🎯 What can you do with this?

- **Learn the basics** of how AI works
- **Understand backpropagation** (the learning algorithm)
- **Build simple predictors** (like guessing house prices)
- **Experiment** with different network sizes
- **Visualize** how information flows through the network

## 📁 Project Structure

```
napagrad/
├── src/napagrad/           # Main package
│   ├── core.py            # Value class for automatic differentiation
│   ├── neural_network.py  # Neural network components
│   └── visualization.py   # Graph visualization
├── examples/              # Example scripts
│   ├── simple_example.py
│   └── house_price_prediction.py
└── README.md
```

## 📦 Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install the package
pip install -e .
```

## 🧪 Try it yourself!

1. Run the examples:
   ```bash
   # Basic example
   python examples/simple_example.py
   
   # Advanced example with visualization
   python examples/house_price_prediction.py
   ```

2. Create your own network:
   ```python
   from napagrad import Value, MLP, draw_computational_graph
   
   # Try different sizes: [inputs, hidden1, hidden2, outputs]
   my_network = MLP([3, 5, 5, 1])
   ```

3. Visualize it:
   ```python
   output = my_network([Value(1), Value(2), Value(3)])
   dot = draw_computational_graph(output)
   dot.render('my_network', format='svg')
   ```

## 💡 Key Concepts

- **Weights**: How important each input is
- **Bias**: A constant offset (like adjusting a scale)
- **Gradient**: How much to change each weight
- **Backpropagation**: The algorithm that figures out gradients
- **Loss**: How wrong the prediction was

## 🎓 Why build from scratch?

Building this from scratch helps you understand:
- How AI actually works under the hood
- Why neural networks can learn
- What happens during training
- How to debug and improve models

It's like learning to cook by making everything from basic ingredients instead of using pre-made meals!

---

*This is a simplified version for learning. Real neural networks have many more features, but the core ideas are the same!*
