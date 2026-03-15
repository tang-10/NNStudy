from torch.utils.data import Dataset
from yoloV3 import cfg
from glob import glob
import numpy as np
import math
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os


class LoadImgAndLabel(Dataset):
    def __init__(
        self,
        annotation_path,
        base_image_path,
        anchors_group,
        class_num,
        mean,
        std,
        transform=None,
    ):
        super().__init__()
        self.base_image_path = base_image_path
        self.anchors_group = anchors_group
        self.class_num = class_num
        if transform:
            self.t = transform
        else:
            self.t = transforms.Compose(
                [
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        with open(annotation_path, "r") as f:
            self.data = f.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        img_name = data.split()[0]
        boxes = np.array(data.split()[1:], dtype=np.float32)
        # 五个数据分为一组：cls,cx,cy,gt_w,gt_h
        boxes = np.split(boxes, len(boxes) // 5)
        label_all = {}
        # feature: 特征图尺寸
        for feature in self.anchors_group:
            label_all[feature] = torch.zeros(
                (feature, feature, 3, 1 + 4 + self.class_num)
            )
            factor = cfg.SIZE / feature
            for box in boxes:
                # cx,cy: 标签中心点坐标 gt_w,gt_h: 真实框宽,高
                cls, cx, cy, gt_w, gt_h = box
                # 坐标偏移量，坐标索引
                tx, cx_idx = math.modf(cx / factor)
                ty, cy_idx = math.modf(cy / factor)
                cls_onehot = F.one_hot(torch.tensor(int(cls)), self.class_num)

                # 计算这个gt框与当前scale的3个anchor的IoU
                ious = []
                for anchor in self.anchors_group[feature]:
                    anchor_w, anchor_h = anchor
                    inter = min(gt_w, anchor_w) * min(gt_h, anchor_h)
                    union = gt_w * gt_h + anchor_w * anchor_h - inter
                    iou = inter / union if union > 0 else 0
                    ious.append(iou)

                # 只给IoU最大的那个anchor负责（responsible anchor）
                best_idx = ious.index(max(ious))

                # 只给最佳anchor赋值
                anchor_w, anchor_h = self.anchors_group[feature][best_idx]
                tw = torch.log(torch.tensor(gt_w / anchor_w))
                th = torch.log(torch.tensor(gt_h / anchor_h))
                # # 每个特征图三个锚框：例[[360, 360], [360, 180], [180, 360]]
                # for idx, anchor in enumerate(cfg.ANCHORS_GROUP[feature]):
                #     anchor_w, anchor_h = anchor
                #     # 微调锚框
                #     tw = torch.log(torch.tensor(gt_w / anchor_w))
                #     th = torch.log(torch.tensor(gt_h / anchor_h))
                label_all[feature][int(cy_idx), int(cx_idx), best_idx, :] = (
                    torch.tensor([1, tx, ty, tw, th, *cls_onehot])
                )

        img_path = os.path.join(self.base_image_path, img_name)
        img_data = self.t(Image.open(img_path))

        return img_data, label_all[13], label_all[26], label_all[52]


if __name__ == "__main__":
    annotation_path = "yoloV3/data/yolo_annotation.txt"
    load = LoadImgAndLabel(
        annotation_path,
        cfg.BASE_IMG_PATH,
        cfg.ANCHORS_GROUP,
        cfg.CLASS_NUM,
        cfg.MEAN,
        cfg.STD,
    )
    load.__getitem__(2)
