import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from net.yolo import YOYOV3
import cfg
import os


class Detect:
    def __init__(self, model, model_path, mean, std):
        self.model = self.get_model(model, model_path)
        self.t = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    def run(
        self,
        image_path,
        save_path,
        class_num,
        class_info,
        conf_thresh,
        iou_thresh,
        anchors_group,
        draw_image=True,
    ):
        image: torch.Tensor = self.preprocess(image_path)
        image = image.unsqueeze(dim=0)
        out13, out26, out52 = self.model(image)
        # [X,6] 6:conf, x1, y1, x2, y2, cls
        boxes13 = self.get_boxes(out13, 13, conf_thresh, anchors_group)
        boxes26 = self.get_boxes(out26, 26, conf_thresh, anchors_group)
        boxes52 = self.get_boxes(out52, 52, conf_thresh, anchors_group)
        boxes = torch.cat((boxes13, boxes26, boxes52), dim=0)
        # 每个分类单独做NMS筛选
        boxes_filter = []
        for cls in range(class_num):
            idx = boxes[:, 5] == cls
            same_cls_boxes = boxes[idx]
            boxes_filter.extend(self.nms_filter(same_cls_boxes, iou_thresh))
        if draw_image:
            self.draw_rect(image_path, boxes_filter, save_path, class_info)
        else:
            self.write_txt(boxes_filter, save_path, class_info)
        pass

    def get_filter_boxes(
        self, image, class_num, conf_thresh, iou_thresh, anchors_group
    ):
        image = self.t(image).to(cfg.DEVICE)
        image = image.unsqueeze(dim=0)
        out13, out26, out52 = self.model(image)
        # [X,6] 6:conf, x1, y1, x2, y2, cls
        boxes13 = self.get_boxes(out13, 13, conf_thresh, anchors_group)
        boxes26 = self.get_boxes(out26, 26, conf_thresh, anchors_group)
        boxes52 = self.get_boxes(out52, 52, conf_thresh, anchors_group)
        boxes = torch.cat((boxes13, boxes26, boxes52), dim=0)
        # 每个分类单独做NMS筛选
        boxes_filter = []
        for cls in range(class_num):
            idx = boxes[:, 5] == cls
            same_cls_boxes = boxes[idx]
            boxes_filter.extend(self.nms_filter(same_cls_boxes, iou_thresh))
        return boxes_filter

    def preprocess(self, image_path):
        # 图片预处理
        image_pil = Image.open(image_path).convert("RGB")
        image = self.t(image_pil).to(cfg.DEVICE)
        return image

    def get_model(self, model, model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model

    @staticmethod
    def get_boxes(out: torch.Tensor, feature, thresh, anchors_group):
        # out:N 27 feature feature →N 3 9 feature feature
        out = out.reshape(out.shape[0], 3, -1, out.shape[2], out.shape[3])
        # 置信度Sigmod()后再比较
        obj_conf = torch.sigmoid(out[:, :, 0])
        # 置信度筛选
        class_probs = torch.softmax(out[:, :, 5:], dim=2)
        class_conf = class_probs.max(dim=2)[0]
        final_conf = obj_conf * class_conf
        idx = torch.where(final_conf > thresh)  # N,3,feature,feature
        out = out[idx[0], idx[1], :, idx[2], idx[3]]  # X,9(X:框的个数)
        final_conf = final_conf[idx[0], idx[1], idx[2], idx[3]]  # [X]

        # tx,ty,tw,th 处理成x1,y1,x2,y2
        tx, ty, tw, th = out[:, 1], out[:, 2], out[:, 3], out[:, 4]
        factor = cfg.SIZE / feature
        pred_cx = (tx + idx[3]) * factor
        pred_cy = (ty + idx[2]) * factor
        anchors = torch.tensor(anchors_group[feature]).to(cfg.DEVICE)
        # 整组数组索引，返回形状和索引张量相同，即每个idx[1]相同
        anchors = anchors[idx[1]]
        pred_w = torch.exp(tw) * anchors[:, 0]
        pred_h = torch.exp(th) * anchors[:, 1]
        x1 = (pred_cx - pred_w / 2).clamp(0, cfg.SIZE)
        y1 = (pred_cy - pred_h / 2).clamp(0, cfg.SIZE)
        x2 = (pred_cx + pred_w / 2).clamp(0, cfg.SIZE)
        y2 = (pred_cy + pred_h / 2).clamp(0, cfg.SIZE)
        # one-hot标签处理
        cls = class_probs[idx[0], idx[1], :, idx[2], idx[3]].argmax(dim=1)

        boxes = torch.column_stack((final_conf, x1, y1, x2, y2, cls))
        # debug log：
        if len(final_conf) > 0:
            print(
                f"Feature {feature}: 候选框数 = {len(final_conf)}, 最高final_conf = {final_conf.max().item():.4f}"
            )
        else:
            print(f"Feature {feature}: 候选框数 = 0")
        return boxes

    def nms_filter(self, boxes, thresh):
        boxes = boxes[torch.argsort(boxes[:, 0], descending=True)]
        conf_max_boxes = []
        while len(boxes) > 0:
            conf_max_box = boxes[0]
            conf_max_boxes.append(conf_max_box)
            boxes = boxes[1:,]
            iou_boxes = self.cal_iou(boxes[:, 1:5], conf_max_box[1:5])
            idx = iou_boxes < thresh
            boxes = boxes[idx]
        return conf_max_boxes

    @staticmethod
    def cal_iou(boxes, target_box):
        in_x1 = torch.maximum(boxes[:, 0], target_box[0])
        in_y1 = torch.maximum(boxes[:, 1], target_box[1])
        in_x2 = torch.minimum(boxes[:, 2], target_box[2])
        in_y2 = torch.minimum(boxes[:, 3], target_box[3])
        in_w = torch.maximum(in_x2 - in_x1, torch.tensor(0))
        in_h = torch.maximum(in_y2 - in_y1, torch.tensor(0))
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        target_box_area = (target_box[2] - target_box[0]) * (
            target_box[3] - target_box[1]
        )
        in_area = in_w * in_h
        iou = in_area / (boxes_area + target_box_area - in_area)
        return iou

    @staticmethod
    def draw_rect(image_path, boxes, save_path, class_info):
        image = Image.open(image_path).convert("RGB")
        # TODO 直接resize会导致图片变形
        image = image.resize((cfg.SIZE, cfg.SIZE))
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(
            r"/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15
        )
        for box in boxes:
            text, color, _ = class_info[int(box[-1].item())]
            text = f"{text} {box[0]:.2f}"
            draw.rectangle(box[1:5].tolist(), outline=color, width=2)
            draw.text(box[1:3].tolist(), text, font=font, fill=color)
        image.save(save_path)

    @staticmethod
    def write_txt(boxes, save_path, class_info):
        with open(save_path.replace(".jpg", ".txt"), "w") as f:
            for box in boxes:
                class_name = class_info[int(box[-1].item())][0]
                conf = f"{box[0].item():.2f}"
                x1, y1, x2, y2 = (
                    f"{box[1].item():.2f}",
                    f"{box[2].item():.2f}",
                    f"{box[3].item():.2f}",
                    f"{box[4].item():.2f}",
                )
                f.write("%s %s %s %s %s %s\n" % (class_name, conf, x1, y1, x2, y2))


def detect_base():
    model = YOYOV3(dark53=True).to(cfg.DEVICE)
    detect = Detect(model, "yoloV3/checkpoints/best_model_1500.pth", cfg.MEAN, cfg.STD)
    for image_name in os.listdir("yoloV3/tmp/"):
        image_path = os.path.join("yoloV3/tmp/", image_name)
        save_path = os.path.join("yoloV3/output/", f"out_{image_name}")
        detect.run(
            image_path,
            save_path,
            cfg.CLASS_NUM,
            cfg.CLASS_INFO,
            cfg.CONF_THRESH,
            cfg.IOU_THRESH,
            cfg.ANCHORS_GROUP,
        )


def detect_fleurine():
    model = YOYOV3(num_classes=cfg.CLASS_NUM_FLEURINE, dark53=True).to(cfg.DEVICE)
    detect = Detect(
        model,
        "yoloV3/checkpoints/best_model_fleurine_2000.pt",
        cfg.MEAN_FLEURINE,
        cfg.STD_FLEURINE,
    )
    for image_name in os.listdir("ImageDatas/Fleurine_frames"):
        image_path = os.path.join("ImageDatas/Fleurine_frames", image_name)
        save_path = os.path.join("yoloV3/output/out_fleurine_2000_txt", image_name)
        detect.run(
            image_path,
            save_path,
            cfg.CLASS_NUM_FLEURINE,
            cfg.CLASS_INFO_FLEURINE,
            cfg.CONF_THRESH_FLEURINE,
            cfg.IOU_THRESH_FLEURINE,
            cfg.ANCHORS_GROUP_FOR_FLEURINE,
            draw_image=False,
        )


if __name__ == "__main__":
    detect_fleurine()
