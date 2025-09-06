from napagrad import Value, MLP, draw_computational_graph

print("🧠 Neural Network Example with Visualization")
print("=" * 50)

# Create a simple neural network: 2 inputs -> 3 hidden -> 1 output
print("Creating neural network: 2 inputs → 3 hidden neurons → 1 output")
model = MLP([2, 3, 1])

# Example input data (like features of a house: size and age)
print("\n📊 Input data:")
x1 = Value(2.0, label="house_size")  # House size in 1000 sq ft
x2 = Value(0.5, label="house_age")   # House age in decades
inputs = [x1, x2]

print(f"House size: {x1.value} (1000 sq ft)")
print(f"House age: {x2.value} (decades)")

# Forward pass through the network
print("\n🔄 Forward pass through the network...")
output = model(inputs)
print(f"Network prediction: ${output.value:.2f} (predicted house price in $100k)")

# Let's say the actual house price is $180k
target_price = Value(1.8, label="actual_price")
print(f"Actual house price: ${target_price.value} (in $100k)")

# Calculate loss (how wrong we were)
loss = (output - target_price).pow(2)
print(f"Loss (squared error): {loss.value:.4f}")

# Visualize the computational graph BEFORE backpropagation
print("\n📈 Creating computational graph visualization...")
dot = draw_computational_graph(loss)
dot.render('neural_network_before_backprop', format='svg', cleanup=True)
print("Graph saved as 'neural_network_before_backprop.svg'")

# Show some initial gradients (should be 0)
print(f"\nInitial gradients:")
print(f"Output gradient: {output.gradient}")
print(f"First weight gradient: {model.layers[0].neurons[0].weights[0].gradient}")

# Backward pass (backpropagation)
print("\n⬅️  Backward pass (backpropagation)...")
loss.zero_grad()  # Reset all gradients to 0
loss.backward()   # Compute gradients

# Show gradients after backpropagation
print(f"\nGradients after backpropagation:")
print(f"Output gradient: {output.gradient}")
print(f"Loss gradient: {loss.gradient}")
print(f"First weight gradient: {model.layers[0].neurons[0].weights[0].gradient:.6f}")
print(f"First bias gradient: {model.layers[0].neurons[0].bias.gradient:.6f}")

# Visualize the computational graph AFTER backpropagation
dot_after = draw_computational_graph(loss)
dot_after.render('neural_network_after_backprop', format='svg', cleanup=True)
print("Graph saved as 'neural_network_after_backprop.svg'")

# Training step: update parameters using gradients
print("\n🎯 Training step: updating parameters...")
learning_rate = 0.1

# Show some parameters before update
print(f"Before update:")
print(f"First weight: {model.layers[0].neurons[0].weights[0].value:.6f}")
print(f"First bias: {model.layers[0].neurons[0].bias.value:.6f}")

# Update all parameters
for param in model.parameters():
    param.value -= learning_rate * param.gradient

# Show parameters after update
print(f"After update:")
print(f"First weight: {model.layers[0].neurons[0].weights[0].value:.6f}")
print(f"First bias: {model.layers[0].neurons[0].bias.value:.6f}")

# Test the updated network
print("\n🔄 Testing updated network...")
new_output = model(inputs)
new_loss = (new_output - target_price).pow(2)
print(f"New prediction: ${new_output.value:.2f}")
print(f"New loss: {new_loss.value:.4f}")
print(f"Loss improvement: {loss.value - new_loss.value:.4f}")

# Multiple training steps
print("\n🏋️  Multiple training steps...")
for step in range(5):
    # Forward pass
    pred = model(inputs)
    loss_val = (pred - target_price).pow(2)
    
    # Backward pass
    loss_val.zero_grad()
    loss_val.backward()
    
    # Update
    for param in model.parameters():
        param.value -= learning_rate * param.gradient
    
    print(f"Step {step + 1}: Loss = {loss_val.value:.4f}, Prediction = ${pred.value:.2f}")

# Final visualization
print("\n📊 Final computational graph...")
final_dot = draw_computational_graph(loss_val)
final_dot.render('neural_network_final', format='svg', cleanup=True)
print("Final graph saved as 'neural_network_final.svg'")

print("\n✅ Example complete!")
print("Check the generated .svg files to see the computational graphs!")
print("The graphs show how data flows through the network and how gradients are computed.")
