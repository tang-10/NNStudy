import os
import numpy as np
from PIL import Image
from tqdm import tqdm


def calculate_mean_std(image_dir):
    # 支持 jpg/jpeg/png
    image_files = [
        f
        for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not image_files:
        print("❌ 文件夹里没有图片！")
        return None, None

    print(f"✅ 找到 {len(image_files)} 张图片，开始计算...")

    # 累计每个通道的 sum 和 sum²
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sum_sq = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    for filename in tqdm(image_files):
        img_path = os.path.join(image_dir, filename)
        try:
            img = Image.open(img_path).convert("RGB")
            img_array = np.array(img) / 255.0  # 转为 [0,1] 范围
            h, w = img_array.shape[:2]
            pixels = h * w

            channel_sum += img_array.sum(axis=(0, 1))
            channel_sum_sq += (img_array**2).sum(axis=(0, 1))
            total_pixels += pixels
        except Exception as e:
            print(f"⚠️ 跳过 {filename}: {e}")
            continue

    # 计算全局 mean 和 std
    mean = channel_sum / total_pixels
    std = np.sqrt(channel_sum_sq / total_pixels - mean**2)

    print("\n🎯 计算完成！")
    print(f"MEAN = [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
    print(f"STD  = [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")

    return mean.tolist(), std.tolist()


if __name__ == "__main__":
    image_dir = r"ImageDatas/Fleurine_frames"

    mean, std = calculate_mean_std(image_dir)
