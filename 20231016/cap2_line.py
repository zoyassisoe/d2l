import torch
import random

# 生成数据
def normal_data(w, b, n):
    x = torch.normal(0, 1, (n, len(w)))
    y = torch.matmul(x, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return x, y.reshape((-1, 1))

# 批数据
def batch_data(x, y, batch_size):
    n = len(x)
    indexs = list(range(n))
    random.shuffle(indexs)
    for i in range(0, n, batch_size):
        batch_indexs = torch.tensor(
            indexs[i: min(i+batch_size, n)]
        )
        yield x[batch_indexs], y[batch_indexs]

# 定义模型
def linreg(w, b, x):
    return torch.matmul(x, w) +b

# 损失函数
def squared_loss(y, y_hat):
    return (y.reshape(y_hat.shape)-y_hat)**2/2

# sgd
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr*param.grad/batch_size
            param.grad.zero_()

# 生成数据
n = 1000
w = torch.tensor([4.5, 6])
b = torch.tensor([1.7])
x,y = normal_data(w,b,n)

# 超参数
lr = 0.03
num_epochs = 10
net = linreg
loss = squared_loss

# 初始化
batch_size = 50
w = torch.normal(0, 1, [2, 1], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)
trainer = torch.optim.SGD([w,b], lr=lr)

# 训练
for i in range(num_epochs):
    for x_batch, y_batch in batch_data(x, y, batch_size):
        l = loss(y_batch, net(w, b, x_batch))
        l.sum().backward()
        sgd([w, b], lr, batch_size)
        
    l = loss(y, net(w ,b, x))
    print('epoch:{},loss:{}'.format(i, l.mean()))