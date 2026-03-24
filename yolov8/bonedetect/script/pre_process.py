import cv2
import os
import random
from PIL import Image
from tqdm import tqdm
from torchvision import models
from torch import nn
import numpy as np


# 数据预处理  去雾
def opt_img(img_path):
    img = cv2.imread(img_path)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    cv2.imwrite(img_path, img)


# 数据增样
def img_rotate(img_path, flag=5):
    img = Image.open(img_path)
    for i in range(flag):
        angle = random.randint(-15, 15)
        dst = img.rotate(angle)
        file_path_name, _ = img_path.split(".")
        dst.save(file_path_name + f"{i}.png")


# 图像预处理
def pre_data():
    for pic_folder in tqdm(os.listdir(pic_path_folder)):
        data_path = os.path.join(pic_path_folder, pic_folder)
        # num_class = len(os.listdir(data_path))
        for folder in os.listdir(data_path):
            path = os.path.join(data_path, folder)
            if os.path.isfile(path):
                continue

            img_list = os.listdir(path)
            for index, img in enumerate(img_list):
                # 去雾
                opt_img(os.path.join(path, img))
                # 增样
                img_rotate(os.path.join(path, img))


def save_file(list, path, name):
    myFile = os.path.join(path, name)
    if os.path.exists(myFile):
        os.remove(myFile)
    with open(myFile, "w") as f:
        f.writelines(list)


# 切分数据集
def split_data():
    for pic_folder in tqdm(os.listdir(pic_path_folder)):
        data_path = os.path.join(pic_path_folder, pic_folder)

        num_lass = len(os.listdir(data_path))

        train_list = []
        val_list = []

        train_ratio = 0.9
        for folder in os.listdir(data_path):
            path = os.path.join(data_path, folder)
            if os.path.isfile(path):
                continue

            train_nums = len(os.listdir(path)) * train_ratio
            img_lists = os.listdir(path)
            random.shuffle(img_lists)
            for index, img in enumerate(img_lists):
                if index < train_nums:
                    train_list.append(os.path.join(path, img) + " " + str(int(folder) - 1) + "\n")
                else:
                    val_list.append(os.path.join(path, img) + " " + str(int(folder) - 1) + "\n")
            random.seed(100)
            random.shuffle(train_list)
            random.shuffle(val_list)
            save_file(train_list, data_path, "train.txt")
            save_file(val_list, data_path, "val.txt")


"""
作用:
1. 去雾
2. 增样
3. 划分数据集
"""

pic_path_folder = r"/home/tangl/projects/datasets/arthrosis"

if __name__ == "__main__":
    # 图像预处理
    pre_data()
    # 拆分数据集
    # split_data()
