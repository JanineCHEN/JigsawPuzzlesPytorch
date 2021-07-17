import torch.nn as nn
import torch
from thop import profile
from thop import clever_format

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class Vgg16(nn.Module):

    def __init__(self, in_channels):
        super(Vgg16, self).__init__()

        # block 1 64 * 64
        self.conv1_1 = ConvBlock(in_channels, 64)
        self.conv1_2 = ConvBlock(64, 64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 2 32 * 32
        self.conv2_1 = ConvBlock(64, 128)
        self.conv2_2 = ConvBlock(128, 128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 3 16 * 16
        self.conv3_1 = ConvBlock(128, 256)
        self.conv3_2 = ConvBlock(256, 256)
        self.conv3_3 = ConvBlock(256, 256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 4 8 * 8
        self.conv4_1 = ConvBlock(256, 512)
        self.conv4_2 = ConvBlock(512, 512)
        self.conv4_3 = ConvBlock(512, 512)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # block5 4 * 4
        self.conv5_1 = ConvBlock(512, 512)
        self.conv5_2 = ConvBlock(512, 512)
        self.conv5_3 = ConvBlock(512, 512)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.maxpool1(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.maxpool2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.maxpool3(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.maxpool4(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.maxpool5(x)

        return x


class JigsawNet(nn.Module):

    def __init__(self, in_channels, n_classes):
        super(JigsawNet, self).__init__()

        self.conv = Vgg16(in_channels)

        self.fc6 = nn.Linear(2048, 512)
        self.fc7 = nn.Linear(4608, 4096)
        self.classifier = nn.Linear(4096, n_classes)

    def forward(self, x):

        B, _, _, _, _ = x.size()
        res = []
        for i in range(9):
            p = self.conv(x[:, i, ...])
            p = p.view(B, -1)
            p = self.fc6(p)
            res.append(p)

        p = torch.cat(res, 1)
        p = self.fc7(p)
        p = self.classifier(p)

        return p

    def encode(self, x):
        B, _, _, _, _ = x.size()
        res = []
        for i in range(9):
            p = self.conv(x[:, i, ...])
            p = p.view(B, -1)
            p = self.fc6(p)
            res.append(p)

        p = torch.cat(res, 1)
        p = self.fc7(p)
        return p


if __name__ == '__main__':

    x = torch.rand(32, 9, 1, 64, 64)
    model = JigsawNet(in_channels=1, n_classes=1000)

    flops, params = profile(model, inputs=(x,))
    flops, params = clever_format([flops, params], "%.3f")

    print(flops, params)

    print(model(x).shape)






