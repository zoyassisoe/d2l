import torch
from torch import nn
import torchvision
from torchvision import transforms
import pandas as pd
import os
from PIL import Image
from torch.utils import data
from sklearn.utils import shuffle


def load_train_csv():
    df = pd.read_csv('../data/classify-leaves/train.csv')
    list_type = df['label'].unique()
    dic = {}
    for i,t in enumerate(list_type):
        dic[t] = i
    df['num_label'] = df['label'].map(dic)
    return shuffle(df)

class Dataset(object):
    def __init__(self, data_frame, root_dir, transform=None) -> None:
        self.data_frame = data_frame
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        imgpath = os.path.join(self.root_dir, self.data_frame.iloc[idx,0])
        label = self.data_frame.iloc[idx,2]
        img = Image.open(imgpath).convert('RGB')
        if self.transform:
            img = self.transform(img)

        return img, label

def ac(data_iter, net, device):
    num_acs = []
    for x, y in data_iter:
        x = x.to(device)
        y = y.to(device)
        y_hat = net(x)
        maxs, indexs = torch.max(y_hat, dim=1)
        num_acs.append(y.eq(indexs).sum()/indexs.shape[0])
    return sum(num_acs)/len(num_acs)


if __name__ == '__main__':
    data_csv = load_train_csv()
    train_csv = data_csv.iloc[:15000,:]
    test_csv = data_csv.iloc[15000:,:]
    trans = transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()])
    train_dataset = Dataset(train_csv, '../data/classify-leaves', trans)
    test_dataset = Dataset(test_csv, '../data/classify-leaves', trans)
    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

    batch_size = 1024
    num_epochs = 100
    lr = 0.9
    train_iter = data.DataLoader(train_dataset, batch_size=batch_size)
    test_iter = data.DataLoader(test_dataset, batch_size=batch_size)

    net = Net
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    loss = nn.CrossEntropyLoss()
    trainer = torch.optim.SGD(net.parameters(), lr=lr)

    print('start')
    for i in range(num_epochs):
        for x,y in train_iter:
            x = x.to(device)
            y = y.to(device)
            y_hat = net(x)
            l = loss(y_hat, y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
            print(l)
        print('acc--------------------------------------------------')
        print(ac(test_iter, net, device))