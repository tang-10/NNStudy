from torch import nn
import torch


class ConvBNLeakyReLU(nn.Module):
    """Conv + BatchNorm + LeakyReLu(可选)"""

    def __init__(self, in_c, out_c, k, stride=1, relu=True):
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=k,
                stride=stride,
                padding=k // 2,
                bias=False,
            ),
            nn.BatchNorm2d(out_c),
        ]
        if relu:
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ResBlock(nn.Module):
    """ConvBNLeakyReLU + ConvBNLeakyReLU + Residual"""

    def __init__(self, channels):
        super().__init__()
        # dark53中的残差块，两个卷积后都加了ReLU，跟传统Bottleneck残差块不一样
        # (传统Bottleneck残差块最后一次卷积不加ReLu，做完Shortcut之后再ReLu一次)
        hidden_channels = channels // 2
        self.layers = nn.Sequential(
            ConvBNLeakyReLU(channels, hidden_channels, 1, relu=True),
            ConvBNLeakyReLU(hidden_channels, channels, 3, relu=True),
        )

    def forward(self, x):
        return self.layers(x) + x


class DarKNetStage(nn.Module):
    """DownSample + n*ResBlock"""

    def __init__(self, in_c, out_c, num_resblocks):
        super().__init__()
        self.num_resblocks = num_resblocks
        # 下采样
        self.downSample = ConvBNLeakyReLU(in_c, out_c, 3, 2)
        # 残差块
        self.resblock = ResBlock(out_c)

    def forward(self, x):
        x = self.downSample(x)
        for _ in range(self.num_resblocks):
            x = self.resblock(x)
        return x


class FiveConvSet(nn.Module):
    """标准5层卷积序列:1x1降维 → 3x3提取 → 1x1降维 → 3x3提取 → 1x1降维"""

    def __init__(self, in_c, out_c):
        super().__init__()
        # 第一次1x1卷积直接降维到out_c，后续用out_c*2的通道数进行升/降维
        self.layers = nn.Sequential(
            ConvBNLeakyReLU(in_c, out_c, 1),
            ConvBNLeakyReLU(out_c, out_c * 2, 3),
            ConvBNLeakyReLU(out_c * 2, out_c, 1),
            ConvBNLeakyReLU(out_c, out_c * 2, 3),
            ConvBNLeakyReLU(out_c * 2, out_c, 1),
        )

    def forward(self, x):
        return self.layers(x)


class NeckBlock(nn.Module):
    def __init__(self):
        super().__init__()
        # 通道数变化：
        # 大目标(1024x13×13)
        #   输入 1024 -> 512 -> 1024 -> 512 -> 1024 -> 512
        # 中目标(512x26x26) contact通道后
        #   输入 768 -> 256 -> 512 -> 256 -> 512 -> 256
        # 小目标(256x52x52) contact通道后
        #   输入 384 -> 128 -> 256 -> 128 -> 256 -> 128
        # C0 大目标
        self.five_conv1 = FiveConvSet(1024, 512)
        # 1x1卷积
        self.conv1 = ConvBNLeakyReLU(512, 256, 1)
        # C1 中目标
        self.five_conv2 = FiveConvSet(768, 256)
        # 1x1卷积
        self.conv2 = ConvBNLeakyReLU(256, 128, 1)
        # C2 小目标
        self.five_conv3 = FiveConvSet(384, 128)
        # 上采样
        self.upSample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x_13x13, x_26x26, x_52x52):
        # 大目标 512x13x13
        out_13x13 = self.five_conv1(x_13x13)
        p3 = self.conv1(out_13x13)
        p3 = self.upSample(p3)
        # x2:中目标 256x26x26
        out_26x26 = self.five_conv2(torch.cat([x_26x26, p3], dim=1))
        p2 = self.conv2(out_26x26)
        p2 = self.upSample(p2)
        # 小目标 128x52x52
        out_52x52 = self.five_conv3(torch.cat([x_52x52, p2], dim=1))
        return out_13x13, out_26x26, out_52x52
