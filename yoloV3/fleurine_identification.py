from net.yolo import YOYOV3
from detect import Detect
import cfg
import cv2
import numpy as np
import torch

"""
1) 加载模型权重
2) 打开视频，读取每帧图像
3) 图像resize到416,输入模型得到侦测框
4) 侦测框按原图像比例转换，在图像上画出框
"""


class Fleurine_Identification:
    def __init__(self):
        model = YOYOV3(cfg.CLASS_NUM_FLEURINE, dark53=False).to(cfg.DEVICE)
        weights_path = "yoloV3/checkpoints/best_shuffule_model_fleurine.pt"
        self.detect = Detect(
            model, weights_path, mean=cfg.MEAN_FLEURINE, std=cfg.STD_FLEURINE
        )

    def run(self, video_path):
        cap = None
        out = None
        try:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter("output_fleurine.mp4", fourcc, 30.0, (w, h))
            while True:
                if frame is None:
                    break
                # 转 RGB（和模型期望一致）
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 图像resize到416x416
                frame_resized, scale, left, top = self.resize_image(frame)
                # 处理后的图像传入模型，得到预测框
                with torch.no_grad():
                    boxes = self.detect.get_filter_boxes(
                        frame_resized,
                        cfg.CLASS_NUM_FLEURINE,
                        cfg.CONF_THRESH_FLEURINE,
                        cfg.IOU_THRESH_FLEURINE,
                        cfg.ANCHORS_GROUP_FOR_FLEURINE,
                    )
                # 原图像上画框
                self.draw_rect_frame(frame, boxes, scale, left, top)
                out.write(frame)
                ret, frame = cap.read()
        finally:
            if cap is not None:
                cap.release()
            if out is not None:
                out.release()

    @staticmethod
    def resize_image(image, target_size=(416, 416)):
        """等比例缩放到416×416"""
        # 创建纯黑色背景图
        background = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        # 获取图原始大小
        original_height, original_width = image.shape[:2]
        target_width, target_height = target_size
        # 计算缩放比例，保持原始纵横比
        scale_x = target_width / original_width
        scale_y = target_height / original_height
        scale = min(scale_x, scale_y)
        # 计算缩放后的尺寸
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        # 缩放前景图
        resized_image = cv2.resize(image, (new_width, new_height))
        # 计算粘贴位置（居中）
        left = (target_width - new_width) // 2
        top = (target_height - new_height) // 2
        # 将缩放后的图像粘贴到背景中央
        background[top : top + new_height, left : left + new_width] = resized_image

        return background, scale, left, top

    @staticmethod
    def convert_coordinates(bbox_416, scale, padding, original_size):
        """
        将416x416上的坐标转换回原始图像上的坐标
        Args:
            bbox_416: 在416x416图像上的边界框 [x1, y1, x2, y2]
            scale: 缩放比例
            padding: 填充信息 (left, top)
            original_size

        Returns:
            在原始图像上的边界框 [x1, y1, x2, y2]
        """
        x1_416, y1_416, x2_416, y2_416 = bbox_416
        left, top = padding
        orig_w, orig_h = original_size

        # 1. 去除填充，得到在缩放后图像上的坐标
        x1_resized = x1_416 - left
        x2_resized = x2_416 - left
        y1_resized = y1_416 - top
        y2_resized = y2_416 - top

        # 2. 缩放到原始尺寸
        orig_x1 = int(x1_resized / scale)
        orig_y1 = int(y1_resized / scale)
        orig_x2 = int(x2_resized / scale)
        orig_y2 = int(y2_resized / scale)

        # 3. 确保坐标在图像范围内
        orig_x1 = max(0, min(orig_x1, orig_w))
        orig_y1 = max(0, min(orig_y1, orig_h))
        orig_x2 = max(0, min(orig_x2, orig_w))
        orig_y2 = max(0, min(orig_y2, orig_h))

        return [orig_x1, orig_y1, orig_x2, orig_y2]

    def draw_rect_frame(self, frame, boxes, scale, left, top):
        for box in boxes:
            conf = f"{box[0].item():.2f}"
            class_name, _, color = cfg.CLASS_INFO_FLEURINE[int(box[-1].item())]
            orig_x1, orig_y1, orig_x2, orig_y2 = self.convert_coordinates(
                box[1:5].tolist(), scale, (left, top), frame.shape[:2]
            )
            cv2.rectangle(
                frame, (orig_x1, orig_y1), (orig_x2, orig_y2), color=color, thickness=2
            )
            cv2.putText(
                frame,
                f"{class_name} {conf}",
                (orig_x1, orig_y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2,
            )


if __name__ == "__main__":
    fi = Fleurine_Identification()
    video_path = "ImageDatas/part1-1.mp4"
    fi.run(video_path)
