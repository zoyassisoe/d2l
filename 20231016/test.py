import torch

a = torch.arange(5, dtype=torch.float32, requires_grad=True)
b = a*a
b.sum().backward()

print(id(a))
a.data -= a.grad
print(id(a))

a.grad.zero_()
print(id(a))