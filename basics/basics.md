# 🔥 PyTorch Tensor Fundamentals & Operations

This project is a complete guide to understanding **PyTorch tensors**, covering creation, operations, indexing, reshaping, NumPy conversion, GPU usage, and autograd.

---

## 🧠 What is a Tensor?

A **tensor** is the core data structure in PyTorch.  
It is similar to:
- Scalar (0D)
- Vector (1D)
- Matrix (2D)
- Multi-dimensional arrays (ND)

All deep learning operations are performed using tensors.

---

## 📦 Installation

```bash
pip install torch torchvision numpy
```

---

## 🏗️ Tensor Creation

```python
a = torch.empty(2,2,3)   # Uninitialized tensor
b = torch.zeros(2,2,3)   # All zeros
c = torch.ones(2,3, dtype=torch.int)  # All ones with integer type
```

---

## 📏 Tensor Properties

```python
print(c.dtype)   # Data type
print(c.size())  # Shape of tensor
```

---

## 🧾 Creating Tensor from Data

```python
e = torch.tensor([[1.2,2.3,3.1],[4,5,6]])
```

---

## ➕ Tensor Operations

```python
x = torch.rand(2,2)
y = torch.rand(2,2)

z1 = torch.add(x,y)   # Addition
z2 = x - y            # Subtraction
z3 = torch.mul(x,y)   # Multiplication
z4 = x.div_(y)        # In-place division
```

---

## ⚠️ In-place Operation

- Functions ending with `_` modify original tensor
- Example: `x.div_(y)`

---

## 🔍 Indexing & Slicing

```python
p = torch.rand(5,4)

print(p[1,:])        # Row
print(p[:,1])        # Column
print(p[3,3].item()) # Single value
```

---

## 🔄 Reshaping

```python
q = torch.rand(4,4)
r = q.view(2,8)
```

---

## 🔗 NumPy ↔ PyTorch

```python
k = torch.ones(5)
print(k.numpy())

f = np.ones(5)
g = torch.from_numpy(f)
```

### ⚠️ Note:
- Both share same memory
- Changes affect both

---

## 🚀 GPU Support

```python
if(torch.cuda.is_available()):
    device = torch.device("cuda")

    x = torch.ones(5, device=device)
    y = torch.ones(5).to(device)

    z = x + y
    z = z.to("cpu")

    print(z)
```

---

## 🧮 Autograd

```python
m = torch.ones(5, requires_grad=True)
```

- Tracks gradients for backpropagation

---

## ▶️ Run Project

```bash
python index.py
```

---

## 🎯 Key Takeaways

- Tensor = core of PyTorch
- Supports CPU & GPU
- Works with NumPy
- Enables deep learning via autograd

---

## 👨‍💻 Author

Darshan Ajudiya
