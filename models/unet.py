import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34

class ResNet34Backbone(nn.Module):
    def __init__(self, in_channels):
        super(ResNet34Backbone, self).__init__()
        resnet = resnet34(weights='DEFAULT')

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.bn1,
            resnet.relu
        )
        
        self.layer1 = nn.Sequential(*list(resnet.layer1.children()))
        self.layer1[0].conv1.stride = (2, 2)
        
        if self.layer1[0].downsample is None:
            self.layer1[0].downsample = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=1, stride=2, bias=False), # Stride 2 untuk downsampling
                nn.BatchNorm2d(64)
            )
        else:
            if isinstance(self.layer1[0].downsample[0], nn.Conv2d):
                 self.layer1[0].downsample[0].stride = (2, 2)

        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        del resnet

    def forward(self, x):
        x1 = self.stem(x)      
        x2 = self.layer1(x1)  # -> 64 x 128 x 128
        x3 = self.layer2(x2)  # -> 128 x 64 x 64
        x4 = self.layer3(x3)  # -> 256 x 32 x 32
        return x1, x2, x3, x4

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(out_channels + skip_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, skip):
        x = self.deconv(x)

        if x.size()[2:] != skip.size()[2:]:
            x = F.interpolate(x, size=skip.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = ResNet34Backbone(in_channels)

        self.bottleneck = ConvBlock(256, 512) 
        self.bottleneck = ConvBlock(256, 128)

        self.deconv1 = DeconvBlock(128, 256, 256)  # Input: bottleneck (128), Skip: x4 (256). Output: 256
        self.deconv2 = DeconvBlock(256, 128, 128)  # Input: d1 (256), Skip: x3 (128). Output: 128
        self.deconv3 = DeconvBlock(128, 64, 64)    # Input: d2 (128), Skip: x2 (64). Output: 64
        self.deconv4 = DeconvBlock(64, 64, 64)     # Input: d3 (64), Skip: x1 (64). Output: 64


        self.final_conv1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.final_relu = nn.ReLU(inplace=True) 
        self.final_conv2 = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        input_size = x.size()[2:]

        x1, x2, x3, x4 = self.encoder(x)

        bottleneck = self.bottleneck(x4) # x4: 256 channels -> bottleneck: 128 channels

        d1 = self.deconv1(bottleneck, x4) # bottleneck(128) + x4(256) -> d1(256)
        d2 = self.deconv2(d1, x3)         # d1(256) + x3(128) -> d2(128)
        d3 = self.deconv3(d2, x2)         # d2(128) + x2(64) -> d3(64)
        d4 = self.deconv4(d3, x1)         # d3(64) + x1(64) -> d4(64)

        output = self.final_conv1(d4)     # d4(64) -> 32 channels
        output = self.final_relu(output)  # ReLU
        
        output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=True)
        output = self.final_conv2(output) # 32 channels -> out_channels
        
        return output