import torch
from resnet18 import *

net = torch.load('./resnet18_30.pth')


from torchvision import transforms

transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

from torch import nn

from torch.nn import functional as F

F.softmax