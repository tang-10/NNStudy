from torch.utils.data import DataLoader
import torch
from yoloV3 import cfg
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class ModuleTrainer:
    def __init__(
        self, dataset, bath_size, model, criterion: dict, optimizer, scheduler
    ):
        # 加载数据集
        self.dataloader = DataLoader(
            dataset, batch_size=bath_size, shuffle=True, num_workers=8, pin_memory=True
        )
        # 网络
        self.model = model.to(cfg.DEVICE)
        # 多分类损失函数
        self.cls_criterion = criterion["cls"]
        # 位置回归损失函数
        self.loc_criterion = criterion["loc"]
        # 置信度损失函数
        self.conf_criterion = criterion["conf"]
        # 优化器
        self.optimizer = optimizer
        # lr优化器
        self.scheduler = scheduler

    def run(self, epochs, weights_save_path, detect_image=None, **keyargs):
        # writer = SummaryWriter("yoloV3/logs")
        least_loss = 100
        for epoch in range(epochs):
            avg_loss = self.train_one_epoch()
            self.scheduler.step()
            if avg_loss < least_loss:
                torch.save(self.model.state_dict(), weights_save_path)
                print(f"Save Scuessful! loss = {avg_loss}")
                least_loss = avg_loss
            # writer.add_scalar("loss", avg_loss, epoch)
            print(f"epoch={epoch} avg_loss={avg_loss}")
            if epoch >= 1500 and epoch % 50 == 0:
                torch.save(
                    self.model.state_dict(),
                    f"yoloV3/checkpoints/model_shuffle_fleurine_{epoch}.pt",
                )
            if detect_image:
                if epoch >= 1000 and epoch % 10 == 0:
                    from detect import Detect

                    torch.save(
                        self.model.state_dict(),
                        "yoloV3/checkpoints/temp_model.pt",
                    )
                    d = Detect(
                        model=self.model,
                        model_path="yoloV3/checkpoints/temp_model.pt",
                        mean=keyargs.get("mean"),
                        std=keyargs.get("std"),
                    )
                    image_name = detect_image.split("/")[-1].split(".")[0]
                    d.run(
                        image_path=detect_image,
                        save_path=f"yoloV3/output/tmp_output/out{image_name}_{epoch}.jpg",
                        class_num=keyargs.get("class_num"),
                        class_info=keyargs.get("class_info"),
                        conf_thresh=keyargs.get("conf_thresh"),
                        iou_thresh=keyargs.get("iou_thresh"),
                        anchors_group=keyargs.get("anchors_group"),
                    )
        print(f"least_loss={least_loss}")

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        for img_data, label_13, label_26, label_52 in tqdm(self.dataloader, ncols=50):
            img_data = img_data.to(cfg.DEVICE)
            label_13, label_26, label_52 = (
                label_13.to(cfg.DEVICE),
                label_26.to(cfg.DEVICE),
                label_52.to(cfg.DEVICE),
            )
            out_13, out_26, out_52 = self.model(img_data)
            loss_13 = self.calc_loss(out_13, label_13)
            loss_26 = self.calc_loss(out_26, label_26)
            loss_52 = self.calc_loss(out_52, label_52)
            loss = loss_13 + loss_26 + loss_52
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss
        avg_loss = total_loss / len(self.dataloader)

        return avg_loss

    def calc_loss(self, pred: torch.Tensor, label: torch.Tensor, factor=0.9):
        # label:B,H,W,3,9   pred:B,27,H,W
        pred = pred.reshape(pred.shape[0], 3, -1, pred.shape[2], pred.shape[3])
        pred = torch.permute(pred, dims=(0, 3, 4, 1, 2))
        pos_mask = label[..., 0] == 1  # 正样本
        noobj_mask = label[..., 0] == 0  # 负样本
        pos_mask = pos_mask.to(cfg.DEVICE)
        # cls_loss = self.cls_criterion(pred[..., 5:][pos_mask], label[..., 5:][pos_mask])
        cls_loss = self.cls_criterion(
            pred[..., 5:][pos_mask], label[..., 5:][pos_mask].argmax(dim=-1)
        )
        loc_loss = self.loc_criterion(
            pred[..., 1:5][pos_mask], label[..., 1:5][pos_mask]
        )
        conf_loss_pos = self.conf_criterion(
            pred[..., 0][pos_mask], label[..., 0][pos_mask]
        )
        conf_loss_noobj = self.conf_criterion(
            pred[..., 0][noobj_mask], label[..., 0][noobj_mask]
        )
        # 正样本权重0.9，负样本0.1
        total_loss = (
            cls_loss + loc_loss + conf_loss_pos
        ) * factor + conf_loss_noobj * (1 - factor)
        return total_loss


if __name__ == "__main__":
    pass
    # trainer = ModuleTrainer()
    # trainer.run()
