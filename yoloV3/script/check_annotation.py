import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def check(annotation_path, image_path, save_path):
    with open(annotation_path, "r", encoding="utf-8") as f:
        infos = f.readlines()
    dic_annotation = {}
    for info in infos:
        image_name = info.split()[0]
        boxes = np.array(info.split()[1:])
        boxes = np.split(boxes, len(boxes) // 5)
        dic_annotation[image_name] = boxes
    for image in os.listdir(image_path):
        for image_name, boxes in dic_annotation.items():
            if image == image_name:
                img_pil = Image.open(os.path.join(image_path, image))
                draw = ImageDraw.Draw(img_pil)
                font = ImageFont.truetype(
                    r"/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15
                )
                for box in boxes:
                    cls = box[0]
                    cx, cy, w, h = np.array(box[1:], dtype=np.float32)
                    x1, y1 = cx - w // 2, cy - h // 2
                    x2, y2 = x1 + w, y1 + h
                    draw.rectangle((x1, y1, x2, y2), outline="red", width=2)
                    draw.text((x1, y1), text=cls, font=font, fill="red")
                img_pil.save(os.path.join(save_path, image))

    pass


if __name__ == "__main__":
    annotation_path = r"ImageDatas/fleurine_annotation_v1.txt"
    image_path = r"ImageDatas/Fleurine_frames"
    save_path = r"yoloV3/output/fleurine_frames_check_annotation"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    check(annotation_path, image_path, save_path)
