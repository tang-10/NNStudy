import torch
from torch import nn, optim
from trainer.trainer import ModuleTrainer
from dataset.dataset import LoadImgAndLabel
from net.yolo import YOYOV3
import cfg


def train_base(epochs):
    dataset = LoadImgAndLabel(
        cfg.ANNOTATION_PATH,
        cfg.BASE_IMG_PATH,
        cfg.ANCHORS_GROUP,
        cfg.CLASS_NUM,
        cfg.MEAN,
        cfg.STD,
    )
    # 主干网络使用dark53
    model = YOYOV3(num_classes=cfg.CLASS_NUM, dark53=True)
    # 主干网络使用ShuffleNetV2
    # model = YOYOV3(dark53=False)
    # 损失函数DICT
    criterion = {}
    # 多分类损失函数 CrossEntropyLoss
    # 根据样本数调节分类权重,正样本数： 0：13, 1：10, 2：14, 3：7
    # (总正样本44 / 类别数4) / 各类别样本数 归一化
    class_weights = torch.tensor([0.85, 1.10, 0.79, 1.57]).to(cfg.DEVICE)
    criterion["cls"] = nn.CrossEntropyLoss(weight=class_weights, reduction="mean")
    # 位置回归损失函数
    criterion["loc"] = nn.MSELoss()
    # 置信度损失函数
    criterion["conf"] = nn.BCEWithLogitsLoss()
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    # lr优化器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    trainer = ModuleTrainer(
        dataset=dataset,
        bath_size=cfg.BATCH_SIZE,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    weights_save_path = f"yoloV3/checkpoints/best_model_{epochs}.pt"
    detect_image = "yoloV3/tmp/09.jpg"
    # 在训练途中每隔10轮输出检测结果，则需传入detect_image以及右面的参数
    trainer.run(
        epochs,
        weights_save_path,
        detect_image,
        mean=cfg.MEAN,
        std=cfg.STD,
        class_num=cfg.CLASS_NUM,
        class_info=cfg.CLASS_INFO,
        conf_thresh=cfg.CONF_THRESH,
        iou_thresh=cfg.IOU_THRESH,
        anchors_group=cfg.ANCHORS_GROUP,
    )


def train_fleurine(epochs):
    dataset = LoadImgAndLabel(
        cfg.ANNOTATION_PATH_FLEURINE,
        cfg.BASE_IMG_PATH_FLEURINE,
        cfg.ANCHORS_GROUP_FOR_FLEURINE,
        cfg.CLASS_NUM_FLEURINE,
        cfg.MEAN_FLEURINE,
        cfg.STD_FLEURINE,
    )
    model = YOYOV3(cfg.CLASS_NUM_FLEURINE, dark53=False)
    criterion = {}
    # 正样本个数：415 其中类0：120 类1：119 类2：115 类3：18 类4：12 类5：10 类6：21
    # 归一化分类权重（采用balanced方式：总样本数/(类别数×各count))
    # 类0：0.494 类1：0.498 类2：0.516 类3：3.294 类4：4.941 类5：5.929 类6：2.823
    # class_weights = torch.tensor([0.494, 0.498, 0.516, 3.294, 4.941, 5.929, 2.823]).to(
    #     cfg.DEVICE
    # )
    # criterion["cls"] = nn.CrossEntropyLoss(weight=class_weights, reduction="mean")
    criterion["cls"] = nn.CrossEntropyLoss()
    criterion["loc"] = nn.MSELoss()
    criterion["conf"] = nn.BCEWithLogitsLoss()
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    # lr优化器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    trainer = ModuleTrainer(
        dataset=dataset,
        bath_size=cfg.BATCH_SIZE_FLEURINE,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    weights_save_path = f"yoloV3/checkpoints/best_shuffule_model_fleurine_{epochs}.pt"
    detect_image = "yoloV3/tmp/fleurine_frames_1.jpg"
    # 在训练途中每隔10轮输出检测结果，则需传入detect_image以及右面的参数
    trainer.run(
        epochs,
        weights_save_path,
        detect_image,
        mean=cfg.MEAN_FLEURINE,
        std=cfg.STD_FLEURINE,
        class_num=cfg.CLASS_NUM_FLEURINE,
        class_info=cfg.CLASS_INFO_FLEURINE,
        conf_thresh=cfg.CONF_THRESH_FLEURINE,
        iou_thresh=cfg.IOU_THRESH_FLEURINE,
        anchors_group=cfg.ANCHORS_GROUP_FOR_FLEURINE,
    )


if __name__ == "__main__":
    # train_base(20)
    train_fleurine(epochs=2000)
