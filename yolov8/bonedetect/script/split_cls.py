from ultralytics.data.split import split_classify_dataset
from pathlib import Path

# 直接对 DIP 文件夹执行
source = Path("/home/tangl/projects/datasets/arthrosis/Ulna")
split_dir = split_classify_dataset(source, train_ratio=0.8)  # 可改成 0.9、0.7 等

print("分割完成！路径是：", split_dir)
