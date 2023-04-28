from torch import nn

def weights_init(m):
    # Conv2d and ConvTranspose2d initialization
    if (type(m) == nn.Conv2d) or (type(m) == nn.ConvTranspose2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    # BatchNorm initialization
    elif type(m) == nn.BatchNorm2d:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
