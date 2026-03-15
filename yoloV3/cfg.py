import torch

# ============================通用参数配置============================
DARKNETSTAGES_SETTING = [
    [32, 64, 1],
    [64, 128, 2],
    [128, 256, 8],
    [256, 512, 8],
    [512, 1024, 4],
]
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
WIDTH = 416
HEIGHT = 416
SIZE = 416
# ============================通用参数配置============================

# ============================人猫狗马数据集============================
ANNOTATION_PATH = "yoloV3/data/yolo_annotation.txt"
"自定义的建议框"
# 20,31, 35,49, 47,77, 80,92, 65,129, 108,139, 122,189, 163,212, 219,270
# ANCHORS_GROUP = {
#     13: [[360, 360], [360, 180], [180, 360]],
#     26: [[180, 180], [180, 90], [90, 180]],
#     52: [[90, 90], [90, 45], [45, 90]],
# }
# ANCHORS_GROUP = {
#     13: [[297, 374], [280, 176], [223, 255]],
#     26: [[169, 216], [123, 281], [105, 170]],
#     52: [[77, 215], [187, 84], [52, 113]],
# }
ANCHORS_GROUP = {
    13: [[248, 80], [171, 99], [91, 137]],
    26: [[149, 27], [205, 22], [98, 51]],
    52: [[60, 27], [30, 57], [23, 119]],
}
CLASS_NUM = 4
CLASS = {0: "person", 1: "cat", 2: "dog", 3: "horse"}
CLASS_COLOR = {0: "blue", 1: "red", 2: "orange", 3: "purple"}
CLASS_INFO = {
    0: ("person", "blue"),
    1: ("cat", "red"),
    2: ("dog", "orange"),
    3: ("horse", "purple"),
}
MODEL_PATH = "params/best_300.pt"
BASE_IMG_PATH = "./yoloV3/data/YOLOv3_JPEGImages"
MEAN = [0.4664, 0.4552, 0.3615]
STD = [0.2147, 0.2044, 0.2046]
BATCH_SIZE = 2
CONF_THRESH = 0.75
IOU_THRESH = 0.45
# ============================人猫狗马数据集============================

# ============================芙莉莲数据集============================
ANNOTATION_PATH_FLEURINE = "ImageDatas/fleurine_annotation_v1.txt"
# anchors = 23,52, 48,90, 62,160, 89,121, 131,140, 102,202, 160,203, 225,218, 322,225
ANCHORS_GROUP_FOR_FLEURINE = {
    13: [(322, 225), (225, 218), (160, 203)],
    26: [(102, 202), (131, 140), (89, 121)],
    52: [(62, 160), (48, 90), (23, 52)],
}
CLASS_NUM_FLEURINE = 7
CLASS_INFO_FLEURINE = {
    0: ("Frieren", "yellow", (0, 255, 255)),
    1: ("Fern", "purple", (255, 0, 128)),
    2: ("Stark", "red", (0, 0, 255)),
    3: ("Himmel", "blue", (255, 0, 0)),
    4: ("Heiter", "green", (0, 255, 0)),
    5: ("Eisen", "brown", (42, 42, 165)),
    6: ("Wirbel", "black", (0, 0, 0)),
}
BASE_IMG_PATH_FLEURINE = "ImageDatas/Fleurine_frames"
BATCH_SIZE_FLEURINE = 20
MEAN_FLEURINE = [0.2361, 0.2424, 0.2506]
STD_FLEURINE = [0.2983, 0.3013, 0.3015]
CONF_THRESH_FLEURINE = 0.99
IOU_THRESH_FLEURINE = 0.3
# ============================芙莉莲数据集============================
