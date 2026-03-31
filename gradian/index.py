import torch

x = torch.randn(2,requires_grad=True)
print(x)

y = x+2
print(y)

z = y*y*y
z = z.mean()
print(z)

v = torch.tensor([1.0,1.0],dtype=torch.float32)
z.backward(v) # dz/dx
print(x.grad)

a = torch.randn(2,requires_grad=True)
print(a)


# 3 ways to stop gradient tracking
# a.requires_grad_(False)
# a.detach()
# with torch.no_grad():

a.requires_grad_(False)
print(a)

b=a.detach()
print(b)

with torch.no_grad():
    c = a+2
    print(c)
    print(c.requires_grad)

weights = torch.ones(3,requires_grad=True)

for epoch in range(3):
    pred = (weights * 2).sum()
    print(pred)
    pred.backward()
    print(weights.grad)
    weights.grad.zero_()
