import torch
from torch import nn

class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # down sample blocks consist of a 2x2 stride 2 pooling operation followed by 2 3x3 ReLU activated convolutions.
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same', padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same', padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.network(self.pool(x))

class UpSampleBlock(nn.Module):
    def __init__(self, inchannels, outchannels):
        super().__init__()

        # up sample blocks consist of a 2x2 transposed convolution followed by 2 3x3 ReLU activated convolutions
        self.upconv = nn.ConvTranspose2d(inchannels, inchannels//2, kernel_size=2, stride=2)

        self.network = nn.Sequential(
            nn.Conv2d(inchannels, outchannels, kernel_size=3, padding='same', padding_mode='reflect'),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(),
            nn.Conv2d(outchannels, outchannels, kernel_size=3, padding='same', padding_mode='reflect'),
            nn.BatchNorm2d(outchannels),
            nn.ReLU()
        )

    def forward(self, skipx, x):
        x = self.upconv(x)
        return self.network(torch.cat((skipx, x), dim=1 if x.dim() == 4 else 0))

class TumorNet(nn.Module):
    def __init__(self, basechannels=64):
        super().__init__()

        # Tumor net uses a U-Net architecture (Ronneberger et al., 2015)
        # input convolution
        self.inconv = nn.Sequential(
            nn.Conv2d(1, basechannels, kernel_size=3, padding='same', padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(basechannels, basechannels, kernel_size=3, padding='same', padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(basechannels, basechannels, kernel_size=3, padding='same', padding_mode='reflect'),
            nn.ReLU()
        )

        # first downsample
        self.down1 = DownSampleBlock(basechannels, basechannels * 2)

        # second downsample
        self.down2 = DownSampleBlock(basechannels * 2, basechannels * 4)

        # third downsample
        self.down3 = DownSampleBlock(basechannels * 4, basechannels * 8)

        # fourth downsample ("Bottleneck" segment)
        self.down4 = DownSampleBlock(basechannels * 8, basechannels * 16)

        # first upsample
        self.up1 = UpSampleBlock(basechannels * 16, basechannels * 8)

        # second upsample
        self.up2 = UpSampleBlock(basechannels * 8, basechannels * 4)

        # third upsample
        self.up3 = UpSampleBlock(basechannels * 4, basechannels * 2)

        # output convolution
        self.up4 = nn.ConvTranspose2d(basechannels * 2, basechannels, kernel_size=2, stride=2)
        self.outputconv = nn.Sequential(
            nn.Conv2d(basechannels * 2, basechannels, kernel_size=3, padding='same', padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(basechannels, basechannels, kernel_size=3, padding='same', padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(basechannels, 1, kernel_size=1, padding='same', padding_mode='reflect'),
            nn.Sigmoid() # probability of tumor 0-1
        )

    def forward(self, x):
        if x.dim() == 3:
            x.unsqueeze_(0)

        # downsample
        x1 = self.inconv(x) # -> up4
        x2 = self.down1(x1) # -> up3
        x3 = self.down2(x2) # -> up2
        x4 = self.down3(x3) # -> up1

        # bottleneck
        x = self.down4(x4)

        # upsample
        x = self.up1(x4, x)
        x = self.up2(x3, x)
        x = self.up3(x2, x)

        # render to mask
        x = self.up4(x)
        x = self.outputconv(torch.cat((x1, x), dim=1 if x.dim() == 4 else 0))

        return x