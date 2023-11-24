import torch
from torch import nn

def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, (3,3), padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels

    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

# conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
# 每个vgg块把输出高宽降低一半
def vgg(in_chanells, conv_arch):
    conv_blks = []
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_chanells, out_channels))
        in_chanells = out_channels
    
    return nn.Sequential(
        *conv_blks,
        nn.Flatten(),
        nn.Linear(out_channels*7*7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 176)
    )

class VGG(nn.Module):
    def __init__(self, in_chanells, conv_arch) -> None:
        super().__init__()
        self.vgg_layer = vgg(in_chanells, conv_arch)
    
    def forward(self, x):
        return self.vgg_layer(x)