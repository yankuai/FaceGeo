'''
resnet for segmentation or other fully conv purpose
from: https://github.com/warmspringwinds/pytorch-segmentation-detection
@ d6e7e82

'''

import torch
import torch.nn as nn
import model.encoder.resnet_component as models

class Resnet34_16s(nn.Module):

    def __init__(self, input_dim = 3, num_classes=1000, pretrained=False, upsample=False, intermediate=False):
        super(Resnet34_16s, self).__init__()

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet34_16s = models.resnet34(fully_conv=True,
                                       pretrained=pretrained,
                                       output_stride=16,
                                       remove_avg_pool_layer=True,
                                       intermediate=intermediate)

        resnet34_16s.conv1 = nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Randomly initialize the 1x1 Conv scoring layer
        resnet34_16s.fc = nn.Conv2d(resnet34_16s.inplanes, num_classes, 1)

        self.resnet34_16s = resnet34_16s

        self.upsample = upsample
        self._normal_initialization(self.resnet34_16s.fc)

    def _normal_initialization(self, layer):
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    def forward(self, x):
        input_spatial_dim = x.size()[2:]

        x = self.resnet34_16s(x)

        if self.upsample:
            x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)

        return x


class Resnet18_16s(nn.Module):

    def __init__(self, input_dim=3, out_dim=1000):
        super(Resnet18_16s, self).__init__()

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 16
        resnet18_16s = models.resnet18(fully_conv=True,
                                       pretrained=True,
                                       output_stride=16,
                                       remove_avg_pool_layer=True)
        resnet18_16s.conv1 = nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Randomly initialize the 1x1 Conv scoring layer
        resnet18_16s.fc = nn.Conv2d(resnet18_16s.inplanes, out_dim, 1)

        self.resnet18_16s = resnet18_16s

        self._normal_initialization(self.resnet18_16s.fc)

    def _normal_initialization(self, layer):
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    def forward(self, x):
        x = self.resnet18_16s(x)
        return x
