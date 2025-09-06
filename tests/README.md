# 🧪 Test Suite

This directory contains comprehensive tests for the Napagrad neural network framework.

## 📁 Test Structure

```
tests/
├── __init__.py              # Test package initialization
├── test_core.py            # Tests for Value class and automatic differentiation
├── test_neural_network.py  # Tests for neural network components
└── README.md               # This file
```

## 🚀 Running Tests

### Run All Tests
```bash
# From the project root
python -m pytest tests/

# Or with verbose output
python -m pytest tests/ -v
```

### Run Specific Test Files
```bash
# Test core functionality
python -m pytest tests/test_core.py

# Test neural network components
python -m pytest tests/test_neural_network.py
```

### Run Individual Test Classes
```bash
# Test Value class
python -m pytest tests/test_core.py::TestValue

# Test MLP class
python -m pytest tests/test_neural_network.py::TestMLP
```

### Run Individual Test Methods
```bash
# Test specific functionality
python -m pytest tests/test_core.py::TestValue::test_backpropagation
```

## 📋 Test Coverage

### Core Module (`test_core.py`)
- ✅ Value creation and initialization
- ✅ Basic arithmetic operations (+, -, *, pow)
- ✅ Activation functions (relu, tanh)
- ✅ Automatic differentiation and backpropagation
- ✅ Scalar operation handling

### Neural Network Module (`test_neural_network.py`)
- ✅ Neuron creation and forward pass
- ✅ Layer creation and parameter management
- ✅ MLP architecture and forward pass
- ✅ Complete training step (forward + backward + update)
- ✅ Parameter counting and management

## 🔧 Adding New Tests

When adding new features:

1. **Create test cases** in the appropriate test file
2. **Follow naming convention**: `test_<functionality>`
3. **Use descriptive names** that explain what's being tested
4. **Test both success and edge cases**
5. **Update this README** if adding new test files

### Example Test Structure
```python
def test_new_feature(self):
    """Test description of what this test verifies."""
    # Arrange: Set up test data
    input_data = Value(1.0)
    
    # Act: Perform the operation
    result = input_data.new_operation()
    
    # Assert: Verify the result
    assert result.value == expected_value
    assert result.operation == "new_operation"
```

## 📊 Test Requirements

- **pytest**: Main testing framework
- **Python 3.7+**: Minimum Python version
- **napagrad package**: Must be installed (`pip install -e .`)

## 🎯 Test Goals

- **Verify correctness**: All mathematical operations work as expected
- **Ensure stability**: Backpropagation computes correct gradients
- **Validate architecture**: Neural network components work together
- **Catch regressions**: Changes don't break existing functionality

---

*Run tests regularly to ensure code quality and catch issues early!*
