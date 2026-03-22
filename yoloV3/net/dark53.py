from yoloV3.net.module import ConvBNLeakyReLU, DarKNetStage
from torch import nn
import torch
import yoloV3.cfg

# # in_c,out_c,num_resblocks
# darkNetStages_setting = [
#     [32, 64, 1],
#     [64, 128, 2],
#     [128, 256, 8],
#     [256, 512, 8],
#     [512, 1024, 4],
# ]


class Dark53(nn.Module):
    def __init__(self):
        super().__init__()
        # 第一层 输入3*416*416 →输出32*416*416
        self.initial_layer = ConvBNLeakyReLU(3, 32, 3, 1)

        # 下采样+残差块
        netStages = []
        # 获取darknet53Stage的配置（in_c,out_c,num_resblocks）
        for config in yoloV3.cfg.DARKNETSTAGES_SETTING:
            netStages.append(DarKNetStage(config[0], config[1], config[2]))

        self.netStages = nn.Sequential(*netStages)

    def forward(self, x):
        x = self.initial_layer(x)
        x = self.netStages[0](x)
        x = self.netStages[1](x)
        out_52x52 = self.netStages[2](x)
        out_26x26 = self.netStages[3](out_52x52)
        out_13x13 = self.netStages[4](out_26x26)
        return out_13x13, out_26x26, out_52x52


if __name__ == "__main__":
    dark53 = Dark53().to("cuda")
    x_ = torch.randn(1, 3, 416, 416).to("cuda")
    outs = dark53(x_)
    for out in outs:
        print(out.shape)

    # 输出
    # torch.Size([1, 256, 52, 52])
    # torch.Size([1, 512, 26, 26])
    # torch.Size([1, 1024, 13, 13])
