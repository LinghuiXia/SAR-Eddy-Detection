from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
from nets.darknet import darknet53

def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape([-1, 3, 2])[::-1, :, :]

def conv2d(filter_in, filter_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

def atrous_conv(filter_in, filter_out, kernel_size, rate):

    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=rate, dilation = rate, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

#------------------------------------------------------------------------#
#   make_last_layers
#------------------------------------------------------------------------#
def make_last_layers(filters_list, in_filters, out_filter):
    m = nn.ModuleList([
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        nn.Conv2d(filters_list[1], out_filter, kernel_size=1,
                                        stride=1, padding=0, bias=True)
    ])
    return m

#------------------------------------------------------------------------#
#   FSPM
#------------------------------------------------------------------------#
def make_three_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1)
    )

    return m

class MultiScaleFeature(nn.Module):
    def __init__(self, rates=[1, 2, 4]):
        super(MultiScaleFeature, self).__init__()
        

        self.maxpool = nn.MaxPool2d(5, 1, 5//2)
        self.conv = conv2d(512, 512, 3)
        self.atrous_convs = nn.ModuleList([atrous_conv(512, 512, 3, rate) for rate in rates])

    def forward(self, x):
        
        x_pool = self.maxpool(x)
        x_conv = self.conv(x)
        x_atrous_features = [atrous_conv(x) for atrous_conv in self.atrous_convs[::-1]]

        features = torch.cat([x_pool] + [x_conv] + x_atrous_features + [x], dim=1)

        return features

#------------------------------------------------------------------------#
#   EIFM Upsample
#------------------------------------------------------------------------#
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()
        
        self.upsample = nn.Sequential(
            conv2d(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode="nearest")
        )

    def forward(self, x):
        x = self.upsample(x)
        return x

#------------------------------------------------------------------------#
#   EIFM Downsample
#------------------------------------------------------------------------#
class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample, self).__init__()
        
        self.downsample = nn.Sequential(
            conv2d(in_channels, out_channels, 1),
            nn.MaxPool2d(8)
        )

    def forward(self, x):
        x = self.downsample(x)
        return x

#------------------------------------------------------------------------#
#   EIFM CA
#------------------------------------------------------------------------#
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

#------------------------------------------------------------------------#
#   EIFM SA
#------------------------------------------------------------------------#
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

#------------------------------------------------------------------------#
#   EIFM
#------------------------------------------------------------------------#
class EdgeInformationFusion(nn.Module):
    def __init__(self, low_channels, middle_channels, high_channels):
        super(EdgeInformationFusion, self).__init__()
        
        self.middle_channels = middle_channels

        self.upsample = Upsample(high_channels, middle_channels)
        self.downsample = Downsample(low_channels, middle_channels)

        self.channelattention = ChannelAttention(middle_channels)
        self.spatialattention = SpatialAttention()

    
    def forward(self, x_low, x_middle, x_high):
        
        x_low = self.downsample(x_low)
        x_low = x_low + x_low * self.channelattention(x_low)
        
        if self.middle_channels != 1024:
            x_high = self.upsample(x_high)

        x_fusion = x_low * self.spatialattention(x_high)
        
        x_high_sa = x_high + x_high * self.spatialattention(x_high)

        x = torch.cat([x_high, x_high_sa, x_fusion], 1)
        
        return x


class CEABody(nn.Module):
    def __init__(self, anchor, num_classes):
        super(CEABody, self).__init__()
        #---------------------------------------------------#   
        # darknet53
        #---------------------------------------------------#
        self.backbone = darknet53(None)

        out_filters = self.backbone.layers_out_filters

        self.conv1 = make_three_conv([512, 1024], 1024)
        self.MSFE = MultiScaleFeature()
        self.conv2 = make_three_conv([1024, 512], 3072)


        #---------------------------------------------------#
        # EIFM
        #---------------------------------------------------#
        self.EIFM0 = EdgeInformationFusion(128, 1024, 1024)

        #------------------------------------------------------------------------#
        # yolo_head
        #------------------------------------------------------------------------#
        final_out_filter0 = len(anchor[0]) * (5 + num_classes)
        self.last_layer0 = make_last_layers([512, 1024], out_filters[-1] * 3, final_out_filter0)

        final_out_filter1 = len(anchor[1]) * (5 + num_classes)
        self.last_layer1_conv = conv2d(512, 256, 1)
        self.last_layer1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer1 = make_last_layers([256, 512], out_filters[-2] + 256, final_out_filter1)

        final_out_filter2 = len(anchor[2]) * (5 + num_classes)
        self.last_layer2_conv = conv2d(256, 128, 1)
        self.last_layer2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer2 = make_last_layers([128, 256], out_filters[-3] + 128, final_out_filter2)


    def forward(self, x):
        def _branch(last_layer, layer_in):
            for i, e in enumerate(last_layer):
                layer_in = e(layer_in)
                if i == 4: 
                    out_branch = layer_in
            return layer_in, out_branch 
        
        _, _, x3, x2, x1, x0 = self.backbone(x)

        x_MSF = self.conv1(x0)
        x_MSF = self.MSFE(x_MSF)
        x_MSF = self.conv2(x_MSF)

        x_EIFM0 = self.EIFM0(x3, x0, x_MSF) 

        out0, out0_branch = _branch(self.last_layer0, x_EIFM0)

        x1_in = self.last_layer1_conv(out0_branch)
        x1_in = self.last_layer1_upsample(x1_in)

        x1_in = torch.cat([x1_in, x1], 1)

        out1, out1_branch = _branch(self.last_layer1, x1_in)

        x2_in = self.last_layer2_conv(out1_branch)
        x2_in = self.last_layer2_upsample(x2_in)

        x2_in = torch.cat([x2_in, x2], 1)

        out2, _ = _branch(self.last_layer2, x2_in)
        return out0, out1, out2

if __name__ == "__main__":

    anchors_path = 'model_data/yolo_anchors.txt'
    anchors      = get_anchors(anchors_path)
    num_classes = 2

    model = CEABody(anchors, num_classes)
    model.eval()
    print(model.backbone)
    print(model.EIFM0)
    print(model.MSFE)
    image = torch.randn(1, 3, 416, 416)
    with torch.no_grad():
        outputs = model.forward(image)
    for i in range(0, 3):
        print(outputs[i].size())

    summary(model.cuda(), input_size=(3, 416, 416)) # 计算模型参数量
