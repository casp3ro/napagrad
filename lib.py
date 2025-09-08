import torch

# 1. Data: x=1, y=2 (we want model to learn y = 2x)
x = torch.tensor([1.0])
y = torch.tensor([2.0])

# 2. Model parameter (a weight), start with a random guess
w = torch.tensor([0.0], requires_grad=True)

# Training loop
for epoch in range(20):
    # 3. Forward pass: compute prediction
    y_pred = w * x   # model: y = w * x
    
    # 4. Compute loss (error) → mean squared error
    loss = (y_pred - y) ** 2
    
    # 5. Backward pass: compute gradient
    loss.backward()
    
    # 6. Update weight using gradient (gradient descent)
    with torch.no_grad():   # disable gradient tracking
        w -= 0.1 * w.grad   # learning rate = 0.1
    
    # 7. Reset gradients (important!)
    w.grad.zero_()
    
    print(f"Epoch {epoch+1}: w = {w.item():.4f}, loss = {loss.item():.4f}")

print("Final learned weight:", w.item())
