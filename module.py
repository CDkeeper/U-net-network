import torch
from torch import nn


class DoubleConv(nn.Module):  # 封装常用的两次卷积操作
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        input = self.conv(input)
        return input


class UpSample(nn.Module):  # 封装上采样加卷积操作
    def __init__(self, in_ch, out_ch):
        super(UpSample, self).__init__()
        self.sample = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, (1, 1)),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, input):
        input = self.sample(input)
        return input


class MyModule(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(MyModule, self).__init__()
        # 下采样
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)

        # 上采样
        self.up6 = UpSample(1024, 512)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = UpSample(512, 256)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = UpSample(256, 128)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = UpSample(128, 64)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, (1, 1))

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        u6 = self.up6(c5)
        merge6 = torch.cat([u6, c4], dim=1)
        c6 = self.conv6(merge6)
        u7 = self.up7(c6)
        merge7 = torch.cat([u7, c3], dim=1)
        c7 = self.conv7(merge7)
        u8 = self.up8(c7)
        merge8 = torch.cat([u8, c2], dim=1)
        c8 = self.conv8(merge8)
        u9 = self.up9(c8)
        merge9 = torch.cat([u9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        out = nn.Softsign()(c10)
        return out


if __name__ == '__main__':
    myModule = MyModule(1, 1)
    Input = torch.ones(64, 1, 64, 64)
    Output = myModule(Input)
    print(Output.shape)
