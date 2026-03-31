import torch
import numpy as np

a=torch.empty(2,2,3)
print(a)

b=torch.zeros(2,2,3)
print(b)

c=torch.ones(2,3,dtype=torch.int)
print(c.dtype)

d=torch.ones(2,3,dtype=torch.int)
print(d.size())

e=torch.tensor([[1.2,2.3,3.1],[4,5,6]])
print(e)

x = torch.rand(2,2)
y = torch.rand(2,2)
print(x)
print(y)
z1=torch.add(x,y)
print(z1)

z2=x-y
print(z2)

z3=torch.mul(x,y)
print(z3)

z4=x.div_(y)
print(z4)

p=torch.rand(5,4)
print(p)
print(p[1,:])
print(p[:,1])
print(p[3,3].item())

q=torch.rand(4,4)
print(q)
r=q.view(2,8)
print(r)

k=torch.ones(5)
print(k)
print(k.numpy())
print(type(k.numpy()))

k.add_(2)
print(k)
print(k.numpy())

f=np.ones(5)
print(f)
g = torch.from_numpy(f)
print(g)
print(type(g))
f=f+2
print(f)
print(g)

if(torch.cuda.is_available()):
    device = torch.device("cuda")
    x=torch.ones(5,device=device)
    y=torch.ones(5)
    y=y.to(device)
    z=x+y
    z=z.to("cpu")
    print(z)
    print(z.device)

m = torch.ones(5,requires_grad=True)
print(m)