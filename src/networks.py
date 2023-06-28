import math
import torch
from torch import nn


# U-Net model
# Reference: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class UNet(nn.Module):

    def __init__(self, args, gen=True):
        super(UNet, self).__init__()

        feature_enc = args.nf
        feature_dec = args.nf
        nz = args.nf * 8
        n_layers = int(math.log2(args.size)) - 6

        blocks = UNetBlock((feature_enc*8, feature_dec*8), (nz, nz), innermost=True)
        for i in range(3):
            blocks = UNetBlock((feature_enc*8, feature_dec*8), (feature_enc*8, feature_dec*8), submodule=blocks, inner=True)
        for i in range(n_layers):
            blocks = UNetBlock((feature_enc*8, feature_dec*8), (feature_enc*8, feature_dec*8), submodule=blocks)
        blocks = UNetBlock((feature_enc*4, feature_dec*4), (feature_enc*8, feature_dec*8), submodule=blocks)
        blocks = UNetBlock((feature_enc*2, feature_dec*2), (feature_enc*4, feature_dec*4), submodule=blocks)
        blocks = UNetBlock((feature_enc, feature_dec), (feature_enc*2, feature_dec*2), submodule=blocks)

        if gen:
            self.model = UNetBlock((args.nc, args.nc), (feature_enc, feature_dec), submodule=blocks, outermost=True)
            self.act = nn.Tanh()
        else:
            self.model = UNetBlock((args.nc, 1), (feature_enc, feature_dec), submodule=blocks, outermost=True)
            self.act = nn.Identity()

    def forward(self, x):
        out = self.act(self.model(x))
        return out


# U-Net skip connection block
# Reference: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class UNetBlock(nn.Module):

    def __init__(self, outside_nc, inside_nc, submodule=None, outermost=False, innermost=False, inner=False):
        super(UNetBlock, self).__init__()

        self.outermost = outermost

        if outermost:
            down = [nn.Conv2d(outside_nc[0], inside_nc[0], kernel_size=3, stride=1, padding=1, bias=False)] + [nn.ReLU(inplace=True)]
            up = [nn.Conv2d(inside_nc[0] + inside_nc[1], outside_nc[1], kernel_size=3, stride=1, padding=1, bias=False)]
            model = down + [submodule] + up
        elif innermost:
            down = Downsampling((outside_nc[0], inside_nc[0]), (2, 1, 0), BN=False, ReLU=True)
            mid = [nn.Conv2d(inside_nc[0], inside_nc[1], kernel_size=1, stride=1, padding=0, bias=False)]
            up = Upsampling((inside_nc[1], outside_nc[1]), (2, 1, 0), BN=True, ReLU=True)
            model = down + mid + up
        elif inner:
            down = Downsampling((outside_nc[0], inside_nc[0]), (3, 1, 0), BN=True, ReLU=True)
            up = Upsampling((inside_nc[0] + inside_nc[1], outside_nc[1]), (3, 1, 0), BN=True, ReLU=True)
            model = down + [submodule] + up
        else:
            down = Downsampling((outside_nc[0], inside_nc[0]), (4, 2, 1), BN=True, ReLU=True)
            up = Upsampling((inside_nc[0] + inside_nc[1], outside_nc[1]), (4, 2, 1), BN=True, ReLU=True)
            model = down + [submodule] + up
        
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], dim=1)



# PatchGAN Discriminator network model
# Reference: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
# A : making the final output a feature map with a certain size and making a true or false decision at each pixel
# B : making the input image a patch and making a true or false decision at the output of each patch
# This is A type. (A is equivalent to B)
class PatchGAN_Discriminator(nn.Module):

    def __init__(self, args):
        super(PatchGAN_Discriminator, self).__init__()

        feature = args.nf
        n_layers = int(math.log2(args.size)) - 5
        blocks = [nn.Conv2d(args.nc, feature, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True)]
        mul = 1
        mul_pre = 1
        for n in range(1, n_layers):
            mul_pre = mul
            mul = min(2**n, 8)
            blocks += [
                nn.Conv2d(feature*mul_pre, feature*mul, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(feature*mul),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        mul_pre = mul
        mul = min(2**n_layers, 8)
        blocks += [
            nn.Conv2d(feature*mul_pre, feature*mul, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(feature*mul),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        blocks += [nn.Conv2d(feature*mul, 1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        out = self.model(x)
        return out


# Downsampling network block
def Downsampling(nc, size, BN, ReLU, bias=False):

    blocks = [nn.Conv2d(nc[0], nc[1], kernel_size=size[0], stride=size[1], padding=size[2], bias=bias)]

    if BN:
        blocks += [nn.BatchNorm2d(nc[1], affine=True)]
    if ReLU:
        blocks += [nn.ReLU(inplace=True)]

    return blocks


# Upsampling network block
def Upsampling(nc, size, BN, ReLU, bias=False):

    blocks = [nn.ConvTranspose2d(nc[0], nc[1], kernel_size=size[0], stride=size[1], padding=size[2], bias=bias)]

    if BN:
        blocks += [nn.BatchNorm2d(nc[1], affine=True)]
    if ReLU:
        blocks += [nn.ReLU(inplace=True)]

    return blocks
