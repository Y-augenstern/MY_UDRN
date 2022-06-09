import torch.nn.functional as F
import torch
import torch.nn as nn

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, groups, bais):
        super(Conv, self).__init__()
        kernel_size = 3
        padding = 1
        features = 1
        groups =1
        self.conv = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, groups=groups, bias=False)
    def forward(self,x):
        return self.conv(x)

class Residual(nn.Module):

    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)


net = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=1, padding=1))


def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=1))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)



class DudeNet(nn.Module):
    def __init__(self, channels, num_of_layers=15):
        super(DudeNet, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        groups = 1
        layers = []
        kernel_size1 = 1
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels=channels,out_channels=features,kernel_size=kernel_size,padding=padding,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=2,groups=groups,bias=False,dilation=2),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_3 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_4 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_5 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=2,groups=groups,bias=False,dilation=2),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_6 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_7 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=padding,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_8 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_9 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=2,groups=groups,bias=False,dilation=2),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_10 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_11 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_12 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=2,groups=groups,bias=False,dilation=2),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_13 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=padding,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_14 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=padding,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_15 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_16 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, groups=groups, bias=False)
        self.conv1_17 = nn.Conv2d(in_channels=features, out_channels=3, kernel_size=1,padding=0, groups=groups, bias=False)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=1, stride=1,padding=0, groups=1, bias=True)
        self.ReLU = nn.ReLU(inplace=True)
        self.BN = nn.BatchNorm2d(2*features)
        self.Tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.conv2_1 = nn.Sequential(resnet_block(in_channels=channels,out_channels=features,num_residuals=4))
        self.conv2_17 = nn.Conv2d(in_channels=features,out_channels=3,kernel_size=1,padding=0,groups=groups,bias=False)
        self.conv3_1 = nn.Conv2d(in_channels=2*features,out_channels=3,kernel_size=1,padding=0,groups=groups,bias=False)
        self.conv3_2 = nn.Conv2d(in_channels=6,out_channels=3,kernel_size=1,padding=0,groups=groups,bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
                clip_b = 0.025
                w = m.weight.data.shape[0]
                for j in range(w):
                    if m.weight.data[j] >= 0 and m.weight.data[j] < clip_b:
                        m.weight.data[j] = clip_b
                    elif m.weight.data[j] > -clip_b and m.weight.data[j] < 0:
                        m.weight.data[j] = -clip_b
                m.running_var.fill_(0.01)
    def _make_layers(self, block,features, kernel_size, num_of_layers, padding=1, groups=1, bias=False):
        layers = []
        for _ in range(num_of_layers):
            layers.append(block(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, groups=groups, bias=bias))
        return nn.Sequential(*layers)
    def forward(self, x):
        input = x
        x1 = self.conv1_1(x)
        x1 = self.conv1_2(x1)
        x1 = self.conv1_3(x1)
        x1 = self.conv1_4(x1)
        x1 = self.conv1_5(x1)
        x1 = self.conv1_6(x1)
        x1 = self.conv1_7(x1)
        x1t = self.conv1_8(x1)
        x1 = self.conv1_9(x1t)
        x1 = self.conv1_10(x1)
        x1 = self.conv1_11(x1)
        x1 = self.conv1_12(x1)
        x1 = self.conv1_13(x1)
        x1 = self.conv1_14(x1)
        x1 = self.conv1_15(x1)
        x1 = self.conv1_16(x1)


        x2 = self.conv2_1(x)
        x3t = torch.cat([x1, x2], 1)
        x3 = self.BN(x3t)
        x3 = self.ReLU(x3)
        x3 = self.conv3_1(x3)
        x4 = torch.cat([x, x3], 1)
        x4 = self.conv3_2(x4)
        out = x - x4
        return out
