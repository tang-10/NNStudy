from pathlib import Path

# ================== 配置区 ==================
DATA_ROOT = Path("/home/tangl/projects/datasets/arthrosis/Ulna")
# ===========================================


def rename_classes():
    print(f"开始处理目录: {DATA_ROOT}\n")

    for split in ["train", "val"]:
        split_dir = DATA_ROOT / split
        if not split_dir.exists():
            print(f"⚠️  {split} 文件夹不存在，跳过...")
            continue

        print(f"正在处理 {split}/ 文件夹...")

        for i in range(1, 10):  # 只处理 1~9
            old_name = str(i)
            new_name = f"0{i}"

            old_path = split_dir / old_name
            new_path = split_dir / new_name

            if old_path.exists() and not new_path.exists():
                old_path.rename(new_path)
                print(f"  ✅ 重命名: {split}/{old_name}  →  {split}/{new_name}")
            elif new_path.exists():
                print(f"  ⏭️  {new_name} 已存在，跳过 {old_name}")
            else:
                print(f"  ⚠️  {old_name} 不存在，跳过")

    print("\n🎉 所有文件夹重命名完成！")
    print("现在类别文件夹为：01、02、...、09、10、11（排序完全正确）")
    print(f"路径：{DATA_ROOT}")


if __name__ == "__main__":
    rename_classes()
