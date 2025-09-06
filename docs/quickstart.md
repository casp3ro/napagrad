# 🚀 Quick Start Guide

Get up and running with Napagrad in 5 minutes!

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd napagrad

# Install the package
pip install -e .
```

## Your First Neural Network

```python
from napagrad import Value, MLP

# Create a simple network: 2 inputs → 3 hidden → 1 output
model = MLP([2, 3, 1])

# Create some input data
x1 = Value(1.0)
x2 = Value(2.0)
inputs = [x1, x2]

# Forward pass
output = model(inputs)
print(f"Prediction: {output.value}")

# Calculate loss
target = Value(3.0)
loss = (output - target).pow(2)
print(f"Loss: {loss.value}")

# Backward pass (learning)
loss.zero_grad()
loss.backward()

# Update parameters
learning_rate = 0.1
for param in model.parameters():
    param.value -= learning_rate * param.gradient

print(f"New prediction: {output.value}")
```

## Visualizing Your Network

```python
from napagrad import draw_computational_graph

# Create a visualization
dot = draw_computational_graph(loss)
dot.render('my_network', format='svg')
print("Network graph saved as 'my_network.svg'")
```

## Next Steps

- Try the [examples](../examples/) for more complex scenarios
- Read about [basic concepts](concepts.md) to understand how it works
- Check out the [API reference](api/) for detailed documentation

Happy learning! 🎉
