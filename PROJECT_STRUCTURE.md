# 📁 Project Structure

```
napagrad/
├── src/
│   └── napagrad/                 # Main package
│       ├── __init__.py          # Package initialization and exports
│       ├── core.py              # Core Value class for automatic differentiation
│       ├── neural_network.py    # Neural network components (Neuron, Layer, MLP)
│       └── visualization.py     # Graph visualization utilities
├── examples/                     # Example scripts
│   ├── simple_example.py        # Basic neural network example
│   └── house_price_prediction.py # Advanced example with visualization
├── docs/                        # Documentation (future)
├── tests/                       # Unit tests (future)
├── setup.py                     # Package installation script
├── requirements.txt             # Python dependencies
├── README.md                    # Main documentation
└── PROJECT_STRUCTURE.md         # This file
```

## 📦 Package Components

### `src/napagrad/core.py`
- **Value class**: The fundamental building block for automatic differentiation
- Mathematical operations: `+`, `-`, `*`, `pow()`, `exp()`, `tanh()`, `relu()`
- Backpropagation algorithm with topological sorting
- Gradient management (`zero_grad()`)

### `src/napagrad/neural_network.py`
- **Neuron class**: Single neuron with weights and bias
- **Layer class**: Collection of neurons
- **MLP class**: Multi-layer perceptron (complete neural network)

### `src/napagrad/visualization.py`
- **draw_computational_graph()**: Create visual representations of computational graphs
- Graph tracing and node/edge extraction
- Support for multiple output formats (SVG, PNG, PDF)

### `examples/`
- **simple_example.py**: Basic usage demonstration
- **house_price_prediction.py**: Complete example with training and visualization

## 🚀 Installation

### Development Installation
```bash
# Clone the repository
git clone <repository-url>
cd napagrad

# Install in development mode
pip install -e .

# Or install dependencies manually
pip install -r requirements.txt
```

### Usage
```python
from napagrad import Value, MLP, draw_computational_graph

# Create a neural network
model = MLP([2, 4, 1])

# Use it
output = model([Value(1.0), Value(2.0)])
```

## 🧪 Running Examples

```bash
# Basic example
python examples/simple_example.py

# Advanced example with visualization
python examples/house_price_prediction.py
```

## 🔧 Development

### Adding New Operations
1. Add the operation method to `Value` class in `core.py`
2. Add the derivative in the `backward()` method
3. Update tests and documentation

### Adding New Network Components
1. Create new classes in `neural_network.py`
2. Follow the existing pattern with `__call__()` and `parameters()` methods
3. Add to package exports in `__init__.py`

### Code Style
- Follow PEP 8
- Use type hints where appropriate
- Add docstrings for all public methods
- Keep examples simple and educational
