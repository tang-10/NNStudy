from net.dark53 import Dark53
from net.module import NeckBlock, ConvBNLeakyReLU
from torch import nn
import torch
from net.shuffleNetV2 import shuffleNetForYOLO
from yoloV3 import cfg


class YOYOV3(nn.Module):
    def __init__(self, num_classes, dark53=True):
        super().__init__()
        self.num_classes = num_classes
        if dark53:
            self.backbone = Dark53()
        else:
            self.backbone = shuffleNetForYOLO()
        self.neck = NeckBlock()

        # P0 侦测头
        self.p0_head = nn.Sequential(
            ConvBNLeakyReLU(512, 1024, 3),
            # ConvBNLeakyReLU(1024, (1 + 4 + self.num_classes) * 3, 1),
            nn.Conv2d(1024, (1 + 4 + self.num_classes) * 3, 1),
        )
        # P1 侦测头
        self.p1_head = nn.Sequential(
            ConvBNLeakyReLU(256, 512, 3),
            # ConvBNLeakyReLU(512, (1 + 4 + self.num_classes) * 3, 1),
            nn.Conv2d(512, (1 + 4 + self.num_classes) * 3, 1),
        )
        # P2 侦测头
        self.p2_head = nn.Sequential(
            ConvBNLeakyReLU(128, 256, 3),
            # ConvBNLeakyReLU(256, (1 + 4 + self.num_classes) * 3, 1),
            nn.Conv2d(256, (1 + 4 + self.num_classes) * 3, 1),
        )

    def forward(self, x):
        out_13x13, out_26x26, out_52x52 = self.backbone(x)
        out_13x13, out_26x26, out_52x52 = self.neck(out_13x13, out_26x26, out_52x52)
        # 输出
        # torch.Size([1, 512, 13, 13])
        # torch.Size([1, 256, 26, 26])
        # torch.Size([1, 128, 52, 52])
        # P0 侦测头
        out_13x13 = self.p0_head(out_13x13)  # torch.Size([1, 27, 13, 13])
        # P1 侦测头
        out_26x26 = self.p1_head(out_26x26)  # torch.Size([1, 27, 13, 13])
        # P2 侦测头
        out_52x52 = self.p2_head(out_52x52)  # torch.Size([1, 27, 52, 52])
        return out_13x13, out_26x26, out_52x52


if __name__ == "__main__":
    yolov3 = YOYOV3().to("cuda")
    x_ = torch.randn(1, 3, 416, 416).to("cuda")
    outs = yolov3(x_)
    for out in outs:
        print(out.shape)
