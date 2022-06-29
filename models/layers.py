
from torch import nn


class CBR(nn.Module):
    def __init__(self, ch0, ch1, bn=True, sample='down', activation=nn.ReLU(True), dropout=False):
        super().__init__()
        self.bn = bn
        self.activation = activation
        self.dropout = dropout
        if sample=='down':
            self.c = nn.Conv2d(ch0, ch1, 4, 2, 1)
        else:
            self.c = nn.ConvTranspose2d(ch0, ch1, 4, 2, 1)
        if bn:
            self.batchnorm = nn.BatchNorm2d(ch1, affine=True)
        if dropout:
            self.Dropout = nn.Dropout()

    def forward(self, x):
        h = self.c(x)
        if self.bn:
            h = self.batchnorm(h)
        if self.dropout:
            h = self.Dropout(h)
        if not self.activation is None:
            h = self.activation(h)
        return h


class UpSamplePixelShuffle(nn.Module):
    def __init__(self, in_ch, out_ch, up_scale=2, activation=nn.ReLU(True)):
        super().__init__()
        self.activation = activation

        self.c = nn.Conv2d(in_channels=in_ch, out_channels=out_ch*up_scale*up_scale, kernel_size=3, stride=1, padding=1, bias=False)
        self.ps = nn.PixelShuffle(up_scale)

    def forward(self, x):
        h = self.c(x)
        h = self.ps(h)
        if not self.activation is None:
            h = self.activation(h)
        return h
