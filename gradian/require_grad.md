# 🔥 PyTorch `requires_grad` Explained

This guide explains the concept of **`requires_grad` in PyTorch**, which is essential for training neural networks and performing automatic differentiation.

---

## 🧠 What is `requires_grad`?

In PyTorch, `requires_grad` is a parameter that tells the framework:

👉 Whether to **track operations on a tensor** and compute gradients during backpropagation.

---

## 🔹 Basic Example

```python
import torch

a = torch.randn(2, requires_grad=True)
print(a)
```

- Creates a tensor with random values
- Enables gradient tracking

---

## ✅ `requires_grad=True`

When set to **True**:

- PyTorch tracks all operations performed on the tensor
- Builds a computation graph
- Allows gradient calculation using `.backward()`

### Example

```python
import torch

a = torch.tensor([2.0], requires_grad=True)

b = a * 3
b.backward()

print(a.grad)
```

### Output
```
tensor([3.])
```

### Explanation

- b = 3a  
- Gradient (db/da) = 3  
- So, `a.grad = 3`

---

## ❌ `requires_grad=False`

When set to **False** (default):

- No computation graph is created
- No gradients are stored
- Faster execution and lower memory usage

### Example

```python
import torch

a = torch.tensor([2.0], requires_grad=False)

b = a * 3
# b.backward() ❌ ERROR
```

---

## ⚖️ Comparison

| Feature | True | False |
|--------|------|-------|
| Track operations | ✅ Yes | ❌ No |
| Store gradients | ✅ Yes | ❌ No |
| Backpropagation | ✅ Supported | ❌ Not supported |
| Memory usage | Higher | Lower |

---

## 🎯 When to Use?

### ✅ Use `requires_grad=True`
- Training neural networks
- Updating model weights
- Backpropagation

### ❌ Use `requires_grad=False`
- Inference (prediction only)
- Data preprocessing
- Evaluation phase

---

## 💡 Disable Gradient Tracking

```python
import torch

with torch.no_grad():
    y = a * 2
```

- Temporarily disables gradient tracking
- Improves performance during inference

---

## 🚀 Key Takeaways

- `requires_grad=True` → Enables learning
- `requires_grad=False` → No learning
- Essential for deep learning workflows

---

## 👨‍💻 Author

Darshan Ajudiya