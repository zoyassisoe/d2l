import torch
import d2l.torch as d2l
import pandas as pd
import numpy as np
from torch import nn
from torch.utils import data
import os

class Net(nn.Module):
    def __init__(self, num_input, num_output) -> None:
        super().__init__()
        self.layer1 = nn.Linear(num_input, num_output)
    
    def forward(self, x):
        o = self.layer1(x)

        return o
    
def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clip(net(features), 1, float('inf'))
    return torch.sqrt(2 * loss(torch.log(clipped_preds), torch.log(labels)).mean())

current_dir = os.path.dirname(os.path.abspath(__file__))
path = './kaggle_house_data/house-prices-advanced-regression-techniques/train.csv'
path = os.path.join(current_dir, path)
print(path)

all_features = pd.read_csv(path)
label = all_features['SalePrice']
all_features = all_features.iloc[:,1:-1]

# 若无法获得测试数据，则可根据训练数据计算均值和标准差
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)
all_features = pd.get_dummies(all_features, dummy_na=True)

all_features = torch.tensor(all_features.to_numpy(dtype=np.float32), dtype=torch.float32)
label = torch.tensor(label.values, dtype=torch.float32).reshape(-1, 1)

all_features, label = all_features.cuda(), label.cuda()

lr = 0.5
batch_size = 50

train_set = data.TensorDataset(all_features, label)
train_iter = data.DataLoader(train_set, batch_size)
net = Net(331, 1).cuda()
loss = nn.MSELoss(reduction='none')
updater = torch.optim.Adam(net.parameters(), lr=lr)

def init_weights(m):
        if type == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)
net.apply(init_weights)

epoch = 500
log_rmse_loss_list = []
for i in range(epoch):
    for x,y in train_iter:
        updater.zero_grad()
        l = loss(net(x), y)
        l.sum().backward()
        updater.step()
    a = log_rmse(net, all_features, label)
    log_rmse_loss_list.append(a.cpu().detach().numpy())
    print(a)

import matplotlib.pyplot as plt
plt.plot(log_rmse_loss_list)
plt.show()

torch.save(net, 'net.pth')