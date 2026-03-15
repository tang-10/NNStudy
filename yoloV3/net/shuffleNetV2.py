from net.module import ConvBNLeakyReLU
from torch import nn
import torch


def channel_shuffle(x: torch.Tensor, groups):
    """通道混洗"""
    # N C H W
    batch_size, num_channels, height, width = x.size()
    # 每组通道数
    channels_group = num_channels // groups
    # reshape 4维→5维
    x = x.view(batch_size, groups, channels_group, height, width)
    # tranpose 交换第一维(groups)和第二维(channels_group)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batch_size, -1, height, width)
    return x


class BasicBlock(nn.Module):
    """ShuffleNetV2 基本单元:通道数，空间尺寸不变"""

    def __init__(self, channels):
        super().__init__()
        # 通道数必须能被2整除
        branch_channels = channels // 2

        # 左分支：identity
        self.branch1 = nn.Sequential()

        # 右分支
        self.branch2 = nn.Sequential(
            # 1x1Conv + BN + ReLU
            ConvBNLeakyReLU(branch_channels, branch_channels, 1),
            # 3x3DWConv
            nn.Conv2d(
                branch_channels,
                branch_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=branch_channels,
                bias=False,
            ),
            nn.BatchNorm2d(branch_channels),
            # 1x1Conv + BN + ReLU
            ConvBNLeakyReLU(branch_channels, branch_channels, 1),
        )

    def forward(self, x: torch.Tensor):
        # 分离通道：在第一维分为两块
        x1, x2 = x.chunk(2, dim=1)
        out1 = self.branch1(x1)
        out2 = self.branch2(x2)
        # concat
        out = torch.cat([out1, out2], dim=1)
        # 通道混洗
        out = channel_shuffle(out, 2)
        return out


class DownSampleBlock(nn.Module):
    """ShuffleNetV2 下采样模块：通道数变大，空间尺寸减半"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 通道数必须能被2整除
        branch_channels = out_channels // 2
        self.branch1 = nn.Sequential(
            # 3x3DWConv(s=2)
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            # 1x1Conv + BN + ReLU
            ConvBNLeakyReLU(in_channels, branch_channels, 1),
        )
        self.branch2 = nn.Sequential(
            # 1x1Conv + BN + ReLU
            ConvBNLeakyReLU(in_channels, branch_channels, 1),
            # 3x3DWConv(s=2)
            nn.Conv2d(
                branch_channels,
                branch_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=branch_channels,
                bias=False,
            ),
            nn.BatchNorm2d(branch_channels),
            # 1x1Conv + BN + ReLU
            ConvBNLeakyReLU(branch_channels, branch_channels, 1),
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        # concat
        out = torch.cat([out1, out2], dim=1)
        out = channel_shuffle(out, 2)
        return out


class shuffleNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入尺寸 3*224*224 → 24*112*112
        self.conv1 = ConvBNLeakyReLU(3, 24, 3, 2)
        # 最大池化：输出24*56*56
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        # Stage2:1个下采样单元 + 3个基础单元 输出116*28*28
        stage2 = [DownSampleBlock(24, 116)]
        for _ in range(3):
            stage2.append(BasicBlock(116))
        self.stage2 = nn.Sequential(*stage2)
        # Stage3:1个下采样单元 + 7个基础单元 输出232*14*14
        stage3 = [DownSampleBlock(116, 232)]
        for _ in range(7):
            stage3.append(BasicBlock(232))
        self.stage3 = nn.Sequential(*stage3)
        # Stage4:1个下采样单元 + 3个基础单元 输出464*7*7
        stage4 = [DownSampleBlock(232, 464)]
        for _ in range(3):
            stage4.append(BasicBlock(464))
        self.stage4 = nn.Sequential(*stage4)
        # 输出1024*7*7
        self.conv5 = ConvBNLeakyReLU(464, 1024, 1)
        # 平均池化 输出1024*1*1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # 全连接层
        self.liner = nn.Sequential(nn.Flatten(), nn.Linear(1024, 1000))

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        out_28x28 = self.stage2(x)
        out_14x14 = self.stage3(out_28x28)
        out_7x7 = self.stage4(out_14x14)
        out = self.conv5(out_7x7)
        out = self.avgpool(out)
        out = self.liner(out)
        return out


class shuffleNetForYOLO(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入尺寸 3*416*416 → 64*208*208
        self.conv1 = ConvBNLeakyReLU(3, 64, 3, 2)
        # 最大池化：输出64*104*104
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        # Stage2:1个下采样单元 + 3个基础单元 256*52*52
        stage2 = [DownSampleBlock(64, 256)]
        for _ in range(3):
            stage2.append(BasicBlock(256))
        self.stage2 = nn.Sequential(*stage2)
        # Stage3:1个下采样单元 + 7个基础单元 输出512*26*26
        stage3 = [DownSampleBlock(256, 512)]
        for _ in range(7):
            stage3.append(BasicBlock(512))
        self.stage3 = nn.Sequential(*stage3)
        # Stage4:1个下采样单元 + 3个基础单元 输出1024*13*13
        stage4 = [DownSampleBlock(512, 1024)]
        for _ in range(3):
            stage4.append(BasicBlock(1024))
        self.stage4 = nn.Sequential(*stage4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        out_52x52 = self.stage2(x)
        out_26x26 = self.stage3(out_52x52)
        out_13x13 = self.stage4(out_26x26)
        return out_13x13, out_26x26, out_52x52


if __name__ == "__main__":
    # net = BasicBlock(100)
    # x_ = torch.randn(1, 100, 416, 416)
    # out = net(x_)
    # print(out.shape)  # torch.Size([1, 100, 416, 416])
    # net = shuffleNetV2().to("cuda")
    # x_ = torch.randn(1, 3, 224, 224).to("cuda")
    # out = net(x_)
    # print(out.shape)  # torch.Size([1, 1000])
    net = shuffleNetForYOLO().to("cuda")
    x_ = torch.randn(1, 3, 416, 416).to("cuda")
    outs = net(x_)
    for out in outs:
        print(out.shape)
    # 输出
    # torch.Size([1, 1024, 13, 13])
    # torch.Size([1, 512, 26, 26])
    # torch.Size([1, 256, 52, 52])
