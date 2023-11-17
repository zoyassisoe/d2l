import torch
import torchvision
from torchvision import transforms
from torch.utils import data
import matplotlib.pyplot as plt
from torch import nn


mnist_train = torchvision.datasets.FashionMNIST(
    root='../data', train=True, transform=transforms.ToTensor(), download=True
)
mnist_test = torchvision.datasets.FashionMNIST(
    root='../data', train=False, transform=transforms.ToTensor(), download=True
)

def init_weights(m):
    if type == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


def ac(data_iter, net):
    num_acs = []
    for x, y in data_iter:
        y_hat = net(x)
        maxs, indexs = torch.max(y_hat, dim=1)
        num_acs.append(y.eq(indexs).sum()/indexs.shape[0])
    return sum(num_acs)/len(num_acs)

# 超参数
batch_size = 256
num_epochs = 10
lr = 0.1
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4)
test_iter = data.DataLoader(mnist_test, batch_size,shuffle=True, num_workers=4)
net = nn.Sequential(nn.Flatten(), nn.Linear(28*28, 256), nn.ReLU(), nn.Linear(256, 10))


net.apply(init_weights)
loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=lr)

# train
if __name__=='__main__':
    train_acs = []
    test_acs = []
    loss_epochs = []
    for i in range(num_epochs):
        loss_epoch = []
        for x, y in train_iter:
            y_hat = net(x)
            l = loss(y_hat, y)
            trainer.zero_grad()
            l.backward()
            trainer.step()

            loss_epoch.append(l.detach().numpy())
            
        train_ac = ac(train_iter, net)
        test_ac = ac(test_iter, net)
        train_acs.append(train_ac)
        test_acs.append(test_ac)
        loss_epoch = sum(loss_epoch)/len(loss_epoch)
        loss_epochs.append(loss_epoch)
        print('epoch:{}, train iter accuracy:{}, test iter accuracy:{}, loss:{}'.format(
            i, train_ac, test_ac, loss_epoch))
    fig, axes = plt.subplots(1,2, figsize=(8,2))
    axes = axes.flatten()
    axes[0].plot(range(10), loss_epochs)
    axes[1].plot(range(10), train_acs, label='train data')
    axes[1].plot(range(10), test_acs, label='test data')
    axes[1].legend()