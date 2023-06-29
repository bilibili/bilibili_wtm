import torch
import torch.nn as nn
from .HaarDownsample import HaarDownsampling


class ConvBNRelu(nn.Module):
    """
    A sequence of Convolution, Batch Normalization, and ReLU activation
    """

    def __init__(self, channels_in, channels_out, stride=1):
        super(ConvBNRelu, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class ConvNet(nn.Module):
    '''
    Network that composed by layers of ConvBNRelu
    '''

    def __init__(self, in_channels, out_channels, blocks):
        super(ConvNet, self).__init__()

        layers = [ConvBNRelu(in_channels, out_channels)] if blocks != 0 else []
        for _ in range(blocks - 1):
            layer = ConvBNRelu(out_channels, out_channels)
            layers.append(layer)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class PixShufBNRelu(nn.Module):
    """
    A sequence of Pixel Shuffle, Batch Normalization, and ReLU activation
    """

    def __init__(self, channels_out, ratio=2):
        super(PixShufBNRelu, self).__init__()

        self.layers = nn.Sequential(
            nn.PixelShuffle(ratio),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class ConvPixelShuf(nn.Module):
    '''
    Network that composed by layers of ConvBNRelu
    '''

    def __init__(self, in_channels, ratio, blocks):
        super(ConvPixelShuf, self).__init__()

        layers = []
        for _ in range(blocks):
            layers.append(ConvBNRelu(in_channels, in_channels * (ratio ** 2)))
            layers.append(PixShufBNRelu(channels_out=in_channels, ratio=ratio))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ConvPixelShufHaar(nn.Module):
    '''
    Network that composed by layers of ConvBNRelu, enhanced by Haar.
    '''

    def __init__(self, in_channels, ratio):
        super(ConvPixelShufHaar, self).__init__()

        self.layers1 = nn.Sequential(
            ConvBNRelu(in_channels + 3, in_channels * (ratio ** 2)),
            PixShufBNRelu(channels_out=in_channels, ratio=ratio)
        )
        self.layers2 = nn.Sequential(
            ConvBNRelu(in_channels + 3, in_channels * (ratio ** 2)),
            PixShufBNRelu(channels_out=in_channels, ratio=ratio)
        )
        self.layers3 = nn.Sequential(
            ConvBNRelu(in_channels + 3, in_channels * (ratio ** 2)),
            PixShufBNRelu(channels_out=in_channels, ratio=ratio)
        )
        self.layers4 = nn.Sequential(
            ConvBNRelu(in_channels + 3, in_channels * (ratio ** 2)),
            PixShufBNRelu(channels_out=in_channels, ratio=ratio)
        )
        self.haar2 = HaarDownsampling(3)

    def forward(self, x, img):
        ll1 = self.haar2(img)[:, :img.size(1)]
        ll2 = self.haar2(ll1)[:, :img.size(1)]
        ll3 = self.haar2(ll2)[:, :img.size(1)]
        ll4 = self.haar2(ll3)[:, :img.size(1)]

        x = self.layers1(torch.cat([x, ll4], 1))
        x = self.layers2(torch.cat([x, ll3], 1))
        x = self.layers3(torch.cat([x, ll2], 1))
        x = self.layers4(torch.cat([x, ll1], 1))
        return x
