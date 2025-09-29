import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class AtrousConvolution(nn.Module):
    def __init__(self, input_channels, kernel_size, pad, dilation_rate, output_channels=128):
        super(AtrousConvolution, self).__init__()
        self.conv = nn.Conv2d(in_channels=input_channels,
                              out_channels=output_channels,
                              kernel_size=kernel_size, padding=pad,
                              dilation=dilation_rate, bias=False)
        self.batchnorm = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels): 
        super(ASPP, self).__init__()
        self.conv_1x1 = AtrousConvolution(in_channels, 1, 0, 1, out_channels)
        self.conv_3x3_1 = AtrousConvolution(in_channels, 3, 6, 6, out_channels) # Dilation rate 6
        self.conv_3x3_2 = AtrousConvolution(in_channels, 3, 12, 12, out_channels) # Dilation rate 12
        self.conv_3x3_3 = AtrousConvolution(in_channels, 3, 18, 18, out_channels) # Dilation rate 18

        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False)
        self.final_batchnorm = nn.BatchNorm2d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        input_feature_map_size = x.size()[2:] 

        x1 = self.conv_1x1(x)
        x2 = self.conv_3x3_1(x)
        x3 = self.conv_3x3_2(x)
        x4 = self.conv_3x3_3(x)
        
        x5 = self.image_pool(x)
        x5 = F.interpolate(x5, size=input_feature_map_size, mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1) 
        x = self.final_conv(x) 
        x = self.final_batchnorm(x)
        x = self.final_relu(x)
        return x
    
class ResNet34DeepLab(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        if in_channels != 3:
            self.stem_conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.stem_conv1 = resnet.conv1
            
        self.stem = nn.Sequential(
            self.stem_conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        
        self.layer1 = resnet.layer1 # Output: 64 channels, Stride /4
        self.layer2 = resnet.layer2 # Output: 128 channels, Stride /8
        self.layer3 = resnet.layer3 # Output: 256 channels, Stride /16

        del resnet 

    def forward(self, x):
        x_stem = self.stem(x)   
        x1 = self.layer1(x_stem) 
        x2 = self.layer2(x1)   
        x3 = self.layer3(x2)   

        return x1, x3 

class DeepLabv3Plus(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.backbone = ResNet34DeepLab(in_channels)
        self.aspp = ASPP(in_channels=256, out_channels=256) 
        
        self.project = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(256 + 64, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x):
        x_low_level, x_high_level = self.backbone(x)
        x_aspp = self.aspp(x_high_level) # Output ASPP, Stride /16, 128 channels

        x_aspp_upsampled = F.interpolate(x_aspp, scale_factor=4, mode='bilinear', align_corners=False)
        x_low_level_processed = self.project(x_low_level) 
        x_concat = torch.cat((x_aspp_upsampled, x_low_level_processed), dim=1) 
        x_final = self.classifier(x_concat)
        x_final = F.interpolate(x_final, scale_factor=4, mode='bilinear', align_corners=False)
        
        return x_final