from ultralytics import YOLO
from cfg import config
import cv2
import torch
import numpy as np
import torch.nn.functional as F
import math


def get_arthrosis_info(image):
    model = YOLO("./weights/yolov8m_detect.pt")
    results = model(image, conf=0.5)
    return results[0]


def enhance(image_path):
    img = cv2.imread(image_path)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    image_enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return image_enhanced


def draw_arthrosis(image_path, save_path):
    image_enhanced = enhance(image_path)
    result = get_arthrosis_info(image_enhanced)
    boxes = result.boxes.data.cpu()  # torch.Size([21, 6])

    image = cv2.imread(image_path)
    for i in range(boxes.shape[0]):
        box = boxes[i]
        x1, y1, x2, y2 = box[0:4].int().tolist()
        conf = f"{box[4].item():.2f}"
        cls = result.names[box[-1].int().item()]
        color = config.CLS_COLOR[box[-1].int().item()]
        cv2.rectangle(image, (x1, y1), (x2, y2), color=color, thickness=2)
        cv2.putText(
            image,
            cls + " " + conf,
            (x1, y1 - 5),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.6,
            color=color,
            thickness=2,
        )
    cv2.imwrite(save_path, image)


def get_keyjoint(image_path):
    image_enhanced = enhance(image_path)
    result = get_arthrosis_info(image_enhanced)
    boxes = result.boxes.data.cpu()  # torch.Size([21, 6])
    boxes_keyjoint = {}
    images_keyjoint = {}
    classes = boxes[:, -1].int()
    groups = {}
    unique_cls = classes.unique()
    for c in unique_cls:
        groups[c.item()] = boxes[classes == c]
    for idx, val in groups.items():
        # cls 0: Radius 1: Ulna 2: MCPFirst 不用筛选
        if idx in (0, 1, 2):
            # boxes_keyjoint.append(val[0])
            boxes_keyjoint[config.CLS_ARTHROSIS[idx]] = val[0]
        else:
            sorted_val = val[val[:, 0].argsort()]
            # cls 4: ProximalPhalanx 6: DistalPhalanx 分别筛选出3个
            if idx in (4, 6):
                # for i in (0, 2, -1):
                #     boxes_keyjoint.append(sorted_val[i])
                boxes_keyjoint[config.CLS_ARTHROSIS[idx * 100 + 1]] = sorted_val[0]
                boxes_keyjoint[config.CLS_ARTHROSIS[idx * 100 + 2]] = sorted_val[2]
                boxes_keyjoint[config.CLS_ARTHROSIS[idx * 100 + 3]] = sorted_val[-1]
            # cls 3: MCP 5: MiddlePhalanx 分别筛选出2个
            elif idx in (3, 5):
                # for i in (0, 2):
                #     boxes_keyjoint.append(sorted_val[i])
                boxes_keyjoint[config.CLS_ARTHROSIS[idx * 100 + 1]] = sorted_val[0]
                boxes_keyjoint[config.CLS_ARTHROSIS[idx * 100 + 2]] = sorted_val[2]
            else:
                pass
    orig_img = result[0].orig_img
    for key, box in boxes_keyjoint.items():
        x1, y1, x2, y2 = box[0:4].int().tolist()
        crop_img = orig_img[y1:y2, x1:x2]
        # 转为 torch tensor（CHW, float32, 归一化到 0~1）
        crop_tensor = torch.from_numpy(crop_img).permute(2, 0, 1).float() / 255.0
        # 加 batch 维度（1, 3, H, W）方便后面直接喂给分类模型
        crop_tensor = crop_tensor.unsqueeze(0)
        images_keyjoint[key] = crop_tensor

    return result, boxes_keyjoint, images_keyjoint


def save_image_keyjoint(images_keyjoint):
    for key, crop_tensor in images_keyjoint.items():
        save_img = (
            crop_tensor.squeeze(0)  # 去 batch
            .permute(1, 2, 0)  # CHW → HWC
            .numpy()
            * 255.0
        )
        save_img = save_img.astype(np.uint8)
        cv2.imwrite(f"./crops/{key}.png", save_img)


def draw_arthrosis_keyjoint(image_path, save_path):
    result, boxes_keyjoint, _ = get_keyjoint(image_path)

    image = cv2.imread(image_path)
    for box in boxes_keyjoint.values():
        x1, y1, x2, y2 = box[0:4].int().tolist()
        conf = f"{box[4].item():.2f}"
        cls = result.names[box[-1].int().item()]
        color = config.CLS_COLOR[box[-1].int().item()]
        cv2.rectangle(image, (x1, y1), (x2, y2), color=color, thickness=2)
        cv2.putText(
            image,
            cls + " " + conf,
            (x1, y1 - 5),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.6,
            color=color,
            thickness=2,
        )
    cv2.imwrite(save_path, image)


def get_score(sex, images_keyjoint):
    # 预加载所有模型
    joint_to_model = {
        joint: YOLO(f"./weights/{filename}")
        for joint, filename in config.JOINT_TO_CLS_MODEL.items()
    }
    total_score = 0
    scores = {}
    for key, crop_tensor in images_keyjoint.items():
        # 根据key调用不同的分类模型
        cls_model = joint_to_model[key]

        crop_tensor = F.interpolate(
            crop_tensor,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )
        results = cls_model(crop_tensor)
        cls = results[0].probs.top1
        score = config.SCORE[sex][key][cls]
        scores[key] = (cls + 1, score)
        total_score += score
    return total_score, scores


def export(scores, total_score, boneAge):
    report = """
    第一掌骨骺分级{}级，得{}分；
    第三掌骨骨骺分级{}级，得{}分；
    第五掌骨骨骺分级{}级，得{}分；
    第一近节指骨骨骺分级{}级，得{}分；
    第三近节指骨骨骺分级{}级，得{}分；
    第五近节指骨骨骺分级{}级，得{}分；
    第三中节指骨骨骺分级{}级，得{}分；
    第五中节指骨骨骺分级{}级，得{}分；
    第一远节指骨骨骺分级{}级，得{}分；
    第三远节指骨骨骺分级{}级，得{}分；
    第五远节指骨骨骺分级{}级，得{}分；
    尺骨分级{}级，得{}分；
    桡骨骨骺分级{}级，得{}分。

    RUS-CHN分级计分法，受检儿CHN总得分：{}分，骨龄约为{}岁。""".format(
        scores["MCPFirst"][0],
        scores["MCPFirst"][1],
        scores["MCPThird"][0],
        scores["MCPThird"][1],
        scores["MCPFifth"][0],
        scores["MCPFifth"][1],
        scores["PIPFirst"][0],
        scores["PIPFirst"][1],
        scores["PIPThird"][0],
        scores["PIPThird"][1],
        scores["PIPFifth"][0],
        scores["PIPFifth"][1],
        scores["MIPThird"][0],
        scores["MIPThird"][1],
        scores["MIPFifth"][0],
        scores["MIPFifth"][1],
        scores["DIPFirst"][0],
        scores["DIPFirst"][1],
        scores["DIPThird"][0],
        scores["DIPThird"][1],
        scores["DIPFifth"][0],
        scores["DIPFifth"][1],
        scores["Ulna"][0],
        scores["Ulna"][1],
        scores["Radius"][0],
        scores["Radius"][1],
        total_score,
        boneAge,
    )
    return report


def calcBoneAge(score, sex):
    # 根据总分计算对应的年龄
    if sex == "boy":
        boneAge = (
            2.01790023656577
            + (-0.0931820870747269) * score
            + math.pow(score, 2) * 0.00334709095418796
            + math.pow(score, 3) * (-3.32988302362153e-05)
            + math.pow(score, 4) * (1.75712910819776e-07)
            + math.pow(score, 5) * (-5.59998691223273e-10)
            + math.pow(score, 6) * (1.1296711294933e-12)
            + math.pow(score, 7) * (-1.45218037113138e-15)
            + math.pow(score, 8) * (1.15333377080353e-18)
            + math.pow(score, 9) * (-5.15887481551927e-22)
            + math.pow(score, 10) * (9.94098428102335e-26)
        )
        return round(boneAge, 2)
    elif sex == "girl":
        boneAge = (
            5.81191794824917
            + (-0.271546561737745) * score
            + math.pow(score, 2) * 0.00526301486340724
            + math.pow(score, 3) * (-4.37797717401925e-05)
            + math.pow(score, 4) * (2.0858722025667e-07)
            + math.pow(score, 5) * (-6.21879866563429e-10)
            + math.pow(score, 6) * (1.19909931745368e-12)
            + math.pow(score, 7) * (-1.49462900826936e-15)
            + math.pow(score, 8) * (1.162435538672e-18)
            + math.pow(score, 9) * (-5.12713017846218e-22)
            + math.pow(score, 10) * (9.78989966891478e-26)
        )
        return round(boneAge, 2)


if __name__ == "__main__":
    image_path = "./test_1544.png"
    image_name = image_path.split("/")[-1].split(".")[0]
    # save_path_full = image_path.replace(image_name, image_name + "_detected")
    # save_path_selected = image_path.replace(image_name, image_name + "_selected")
    # draw_arthrosis(image_path, save_path_full)
    # draw_arthrosis_keyjoint(image_path, save_path_selected)
    _, _, images_keyjoint = get_keyjoint(image_path)
    # save_image_keyjoint(images_keyjoint)
    total_score, scores = get_score("boy", images_keyjoint)
    boneAge = calcBoneAge(total_score, "boy")
    report = export(scores, total_score, boneAge)
    print(report)
