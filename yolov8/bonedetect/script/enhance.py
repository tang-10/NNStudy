import cv2
from pathlib import Path
from tqdm import tqdm


def enhance(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def process_folder(src_dir, dst_dir):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(list(src_dir.glob("*.*"))):
        img = cv2.imread(str(img_path))

        if img is None:
            print(f"读取失败: {img_path}")
            continue

        enhanced = enhance(img)

        # ✅ 保存
        cv2.imwrite(str(dst_dir / img_path.name), enhanced)


# 🚀 执行
process_folder("../datasets/bone/train/images", "../datasets/bone/train/images_enhanced")
process_folder("../datasets/bone/val/images", "../datasets/bone/val/images_enhanced")
