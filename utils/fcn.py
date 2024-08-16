"""
    Fully Convolution Networks (FCN) using ResNet as backbone architecture by Yi-Jiun Su, 2023/10/26
       modified from the easiest FCN https://github.com/pochih/FCN-pytorch

    resnet should be modified to output each modification of tensor size
    image can be any arbitray size

    Conv2d and ConvTranspose2d are initialized with same parameters, 
    they are inverses of each other in regard to the input and output shapes
    ConvTranspose2d aka as a fractionally-strided convolution or a deconvolution
    (although it is not an actual deconvolution operation as it does not compute a true 
    inverse of convolution).

    Test samples

import torch
from utils import resnet
batch_size, n_class, h, w = 10, 1, 82, 100 # use reduce VAP image size
# test output size
backbone = resnet.resnet50(head_type='multi_layer')
input = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
c0, c1, c2, c3, c4, c5 = backbone(input)

fcn_model = FCN32s(pretrained_net=backbone, n_class=n_class, in_channels=c5.shape[1])
input = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
output = fcn_model(input)
assert output.size() == torch.Size([batch_size, n_class, h, w])

fcn_model = FCN16s(pretrained_net=backbone, n_class=n_class, in_channels=c5.shape[1])
input = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
output = fcn_model(input)
assert output.size() == torch.Size([batch_size, n_class, h, w])

fcn_model = FCN8s(pretrained_net=backbone, n_class=n_class, in_channels=c5.shape[1])
input = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
output = fcn_model(input)
assert output.size() == torch.Size([batch_size, n_class, h, w])

fcn_model = FCNs(pretrained_net=backbone, n_class=n_class, in_channels=c5.shape[1])
input = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
output = fcn_model(input)
assert output.size() == torch.Size([batch_size, n_class, h, w])
"""

import torch.nn as nn

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
                in_channels, 
                out_channels, 
                kernel_size=3, 
                stride=2, 
                padding=1, 
                dilation=1, 
                output_padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x, out_size, norm=True):
        # padding is for the reconstricted images have the same size as the original images
        x = self.deconv(x)
        diffY = out_size[2] - x.shape[2]
        diffX = out_size[3] - x.shape[3]
        x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2, 
                                  diffY // 2, diffY - diffY // 2])
        if norm:
            return self.bn(self.relu(x))
        else:
            return self.relu(x)

class FCN32s(nn.Module):
# pixel-level (L-2) upsample without combining with previous output
    def __init__(self, pretrained_net, n_class, in_channels):
        super().__init__()
        self.n_class = n_class
        self.in_channels = in_channels
        self.pretrained_net = pretrained_net
        self.up1 = (Up(in_channels, in_channels//2))
        self.up2 = (Up(in_channels//2, in_channels//2**2))
        self.up3 = (Up(in_channels//2**2, in_channels//2**3))
        self.conv4 = nn.ConvTranspose2d(in_channels//2**3, 64, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn4 = nn.BatchNorm2d(64)
        self.up5 = (Up(64, 64))
        self.up6 = (Up(64, 32))
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)
    def forward(self, x):
        _, c5, c4, c3, c2, c1, c0 = self.pretrained_net(x,return_multi=True)
        score = self.up1(c5, c4.shape)    # same shape as c4
        score = self.up2(score, c3.shape) # same shape as c3
        score = self.up3(score, c2.shape) # same shape as c3
        score = self.bn4(self.relu(self.conv4(score))) # same shape as c1
        score = self.up5(score, c0.shape) # same shape as c0
        score = self.up6(score, x.shape)  # [B, 32, x.shape[2], x.shape[3]]
        return self.classifier(score)     # [B, n_class, x.shape[2], x.shape[3]]

class FCN16s(nn.Module):
# pixel-level (L-2) upsample + combine L-3 output
    def __init__(self, pretrained_net, n_class, in_channels):
        super().__init__()
        self.n_class = n_class
        self.in_channels = in_channels
        self.pretrained_net = pretrained_net
        self.up1 = (Up(in_channels, in_channels//2))
        self.bn1 = nn.BatchNorm2d(in_channels//2)
        self.up2 = (Up(in_channels//2, in_channels//2**2))
        self.up3 = (Up(in_channels//2**2, in_channels//2**3))
        self.conv4 = nn.ConvTranspose2d(in_channels//2**3, 64, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn4 = nn.BatchNorm2d(64)
        self.up5 = (Up(64, 64))
        self.up6 = (Up(64, 32))
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)
    def forward(self, x):
        _, c5, c4, c3, c2, c1, c0 = self.pretrained_net(x,return_multi=True)
        score = self.up1(c5, c4.shape, norm=False)
        score = self.bn1(score + c4)
        score = self.up2(score, c3.shape) # same shape as c3
        score = self.up3(score, c2.shape) # same shape as c3
        score = self.bn4(self.relu(self.conv4(score))) # same shape as c1
        score = self.up5(score, c0.shape) # same shape as c0
        score = self.up6(score, x.shape)  # [B, 32, x.shape[2], x.shape[3]]
        return self.classifier(score)

class FCN8s(nn.Module):
# pixel-level (L-2) upsample + combine L-3 and L-4 outputs
    def __init__(self, pretrained_net, n_class, in_channels):
        super().__init__()
        self.n_class = n_class
        self.in_channels = in_channels
        self.pretrained_net = pretrained_net
        self.up1 = (Up(in_channels, in_channels//2))
        self.bn1 = nn.BatchNorm2d(in_channels//2)
        self.up2 = (Up(in_channels//2, in_channels//2**2))
        self.bn2 = nn.BatchNorm2d(in_channels//2**2)
        self.up3 = (Up(in_channels//2**2, in_channels//2**3))
        self.conv4 = nn.ConvTranspose2d(in_channels//2**3, 64, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn4 = nn.BatchNorm2d(64)
        self.up5 = (Up(64, 64))
        self.up6 = (Up(64, 32))
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)
    def forward(self, x):
        _, c5, c4, c3, c2, c1, c0 = self.pretrained_net(x,return_multi=True)
        score = self.up1(c5, c4.shape, norm=False)
        score = self.bn1(score + c4)
        score = self.up2(score, c3.shape, norm=False)
        score = self.bn2(score + c3)
        score = self.up3(score, c2.shape)
        score = self.bn4(self.relu(self.conv4(score))) # same shape as c1
        score = self.up5(score, c0.shape)
        score = self.up6(score, x.shape)
        return self.classifier(score) 

class FCNs(nn.Module):
# pixel-level (L-2) upsample + combine L-3, L-4, & L-5 outputs
    def __init__(self, pretrained_net, n_class, in_channels):
        super().__init__()
        self.n_class = n_class
        self.in_channels = in_channels
        self.pretrained_net = pretrained_net
        self.up1 = (Up(in_channels, in_channels//2))
        self.up2 = (Up(in_channels//2, in_channels//2**2))
        self.up3 = (Up(in_channels//2**2, in_channels//2**3))
        self.conv4 = nn.ConvTranspose2d(in_channels//2**3, 64, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn4 = nn.BatchNorm2d(64)
        self.up5 = (Up(64, 64))
        self.up6 = (Up(64, 32))
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)
    def forward(self, x):
        _, c5, c4, c3, c2, c1, c0 = self.pretrained_net(x,return_multi=True)
        score = self.up1(c5, c4.shape)
        score = score + c4
        score = self.up2(score, c3.shape)
        score = score + c3
        score = self.up3(score, c2.shape)
        score = score + c2
        score = self.bn4(self.relu(self.conv4(score))) # same shape as c1
        score = self.up5(score, c0.shape)
        score = self.up6(score, x.shape)
        return self.classifier(score)
