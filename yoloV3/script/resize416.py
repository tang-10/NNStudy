import numpy as np
import cv2
import os


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


image_dir = "ImageDatas/Fleurine_frames_origin"
for image in image_dir:
    image_path = os.path.join(image_dir, image)
    image_resized = resize_image(image_path)
    cv2.imwrite("ImageDatas/Fleurine_frames_resized", image_resized)
