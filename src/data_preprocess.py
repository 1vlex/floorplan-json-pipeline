import argparse
import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

# 0 - background, 1 - wall, 2 - door, 3 - window
COCO_TO_CLASS_ID: Dict[int, int] = {
    2: 1,  # wall
    1: 2,  # door
    3: 3,  # window
}


def draw_ann_list(mask: np.ndarray, anns: List[dict], class_id: int) -> None:
    h, w = mask.shape[:2]
    for ann in anns:
        x, y, bw, bh = ann["bbox"]
        x1 = int(round(x))
        y1 = int(round(y))
        x2 = int(round(x + bw))
        y2 = int(round(y + bh))

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(mask, (x1, y1), (x2, y2), color=class_id, thickness=-1)


def create_masks_for_split(split_dir: Path, out_dir: Path) -> None:
    ann_path = split_dir / "_annotations.coco.json"
    with ann_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}

    anns_by_image = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        anns_by_image.setdefault(img_id, []).append(ann)

    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0

    for img_id, img_meta in images.items():
        file_name = img_meta["file_name"]
        width = int(img_meta["width"])
        height = int(img_meta["height"])

        mask = np.zeros((height, width), dtype=np.uint8)

        anns = anns_by_image.get(img_id, [])

        walls = [a for a in anns if a["category_id"] == 2]   # wall
        doors = [a for a in anns if a["category_id"] == 1]   # door
        windows = [a for a in anns if a["category_id"] == 3] # window

        # порядок: сначала стены (1), потом двери (2), потом окна (3)
        draw_ann_list(mask, walls, class_id=1)
        draw_ann_list(mask, doors, class_id=2)
        draw_ann_list(mask, windows, class_id=3)

        out_path = out_dir / (Path(file_name).stem + "_mask.png")
        cv2.imwrite(str(out_path), mask)
        count += 1

    print(f"{split_dir.name}: saved {count} masks to {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="COCO (cubicasa5k-2) -> class-id masks (0 bg, 1 wall, 2 door, 3 window)"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Путь к корню cubicasa5k-2.v1i.coco",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        required=True,
        help="Путь, куда класть маски (будут поддиректории train/valid/test)",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)

    for split in ("train", "valid", "test"):
        split_dir = data_root / split
        if not split_dir.exists():
            print(f"Пропускаю {split} (нет директории {split_dir})")
            continue
        out_dir = out_root / split
        create_masks_for_split(split_dir, out_dir)


if __name__ == "__main__":
    main()
