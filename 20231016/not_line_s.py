import torch
import matplotlib.pyplot as plt

def f(x, c, H):
    return (torch.matmul(torch.matmul(x, H), x) + torch.matmul(c, x))/2

# 随机维度
shape = 100
# 当g<p是停止迭代
p = 1
maxk = 5000
# 随机参数
H = torch.rand(shape,shape, dtype=torch.float32)*10
H = torch.matmul(H.T,H)
c = torch.rand(shape, dtype=torch.float32)

# zero x
x = torch.ones(shape, dtype=torch.float32, requires_grad=True)
beta = torch.tensor(0.5)
sigma = torch.tensor(0.4)

# 存储f(x)值
ls = []
# 存储梯度值
grads = []
pre_g = torch.zeros(shape, dtype=torch.float32)
for i in range(maxk):
    loss = f(x, c, H)
    ls.append(loss.detach().numpy())
    loss.backward()
    g = x.grad
    
    grads.append(g.clone())
    # 前一步的grad，使用clone开辟新的内存空间存储，detach不会开辟新的内存空间
    pre_g = g.clone()
    
    if torch.norm(g) < p:
        break
    d = -g
    
    mk = 0
    for i in range(20):
        if f(x+torch.pow(beta, i)*d, c, H) <= loss + sigma*torch.pow(beta, i)*torch.matmul(d,g):
            mk=i
            break
    
    # 验证相邻步梯度是否正交，验证代码逻辑
    print(torch.matmul(pre_g, g), mk)
    x.data += torch.pow(beta, mk)*d.data
    x.grad.zero_()
    
# print([torch.matmul(grads[i],grads[i+1]) for i in range(len(grads)-1)])
# 输出最优解
print('最优解:{},minf(x):{},迭代次数:{}'.format(x, loss, len(ls)))

plt.plot(range(len(ls)-2), [i-loss.detach().numpy() for i in ls[2:]])
plt.xlabel('epoch')
plt.ylabel('|f(x)-f(x*)|')
plt.title('The fastest descent method, non-precise line search for Armijo.')
plt.show()