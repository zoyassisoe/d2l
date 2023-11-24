import torchvision
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from torch import nn
from model.resnet18 import ResNet18
from local_data.leave import load_data
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 2
lr = 0.1
epoch = 100

# 加载数据
# data_root_dir = './data'
# trans = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
# set_train = torchvision.datasets.FashionMNIST(root=data_root_dir, train=True, transform=trans)
# set_test = torchvision.datasets.FashionMNIST(root=data_root_dir, train=False, transform=trans)
trans = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
set_train, set_test = load_data(trans)

iter_train = DataLoader(set_train, batch_size=batch_size,num_workers=4)
iter_test = DataLoader(set_test, batch_size=batch_size,num_workers=4)

# net = ResNet18(3,176)
net_path = os.path.join(os.getcwd(), 'CNN/model/resnet18_30.pth')
print(net_path)
net = torch.load(net_path)
loss = nn.CrossEntropyLoss()
updater = torch.optim.SGD(net.parameters(), lr=lr)

def ac(data_iter, net, device):
    num_acs = []
    for x, y in data_iter:
        x = x.to(device)
        y = y.to(device)
        y_hat = net(x)
        maxs, indexs = torch.max(y_hat, dim=1)
        num_acs.append(y.eq(indexs).sum()/indexs.shape[0])
    return sum(num_acs)/len(num_acs)


if __name__=='__main__':
    train_acc = ac(iter_train, net, device)
    print(train_acc)

# if __name__=='__main__':
#     print('train start.')
#     net.to(device)
#     for i in range(epoch):
#         for batch_idx,(x,y) in enumerate(iter_train):
#             x = x.to(device)
#             y = y.to(device)
            
#             y_hat = net(x)
#             l = loss(y_hat, y)
#             updater.zero_grad()
#             l.backward()
#             updater.step()
#             # display
#             if batch_idx%2 == 0:
#                 test_accuracy = ac(iter_test, net, device)
#                 train_accuracy = ac(iter_train, net, device)
#                 print(f'epoch:{i} [{batch_idx*batch_size}/{len(set_train)}({100.0*batch_idx*batch_size/len(set_train):.2f}%)]\t',
#                     f'loss:{l.item():.2f}\t',
#                     f'train accuracy:{100.0*test_accuracy:.2f}%, test accuracy:{100.0*train_accuracy:.2f}%'
#                     )
#         if (i+1)%10 == 0:
#             torch.save(net, 'resnet18_{}.pth'.format(i))
#     # torch.save(net, 'resnet18.pth')