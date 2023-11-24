import torch
from torchinfo import summary
import os
# path = os.path.join(os.getcwd(), 'CNN/model/resnet50-19c8e357.pth')
# print(path)
# net = torch.load(path)

# print(net)
# print(summary(net, input_size=(1,1,224,224)))

from torchvision.models import resnet50

net = resnet50()
# print(net)
print('________________________________')
print(summary(net, input_size=(1,3,224,224)))