# src/eval.py
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np


CLASS_INFO = {
    1: "wall",
    2: "door",
    3: "window",
}


def draw_class_mask_from_json(
    json_data: Dict[str, Any],
    key: str,
    shape: Tuple[int, int],
    thickness: int = 5,
) -> np.ndarray:
    """
    Растеризуем линии из json[key] в бинарную маску (H, W), где 1 = линия, 0 = фон.
    key: "walls", "doors", "windows".
    shape: (H, W) по GT-маске.
    """
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)

    items = json_data.get(key, [])
    for item in items:
        pts = item.get("points", None)
        if not pts or len(pts) < 2:
            continue

        pts_arr = np.array(pts, dtype=np.int32)  # (N, 2) с [x, y]

        if pts_arr.shape[0] == 2:
            # просто отрезок
            p1 = tuple(pts_arr[0])
            p2 = tuple(pts_arr[1])
            cv2.line(mask, p1, p2, color=1, thickness=thickness)
        else:
            # полилиния
            cv2.polylines(
                mask,
                [pts_arr],
                isClosed=False,
                color=1,
                thickness=thickness,
            )

    return mask.astype(bool)


def compute_metrics_for_class(
    gt_masks_dir: Path,
    pred_dir: Path,
    class_id: int,
    class_name: str,
    tolerant_radius: int,
    thickness: int,
) -> Dict[str, float]:
    """
    Считает метрики по одному классу (wall/door/window) для всех картинок в pred_dir.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * tolerant_radius + 1, 2 * tolerant_radius + 1),
    )

    tp = fp = fn = 0
    inter_tol = union_tol = 0

    sum_dist = 0.0
    count_dist = 0

    gt_has_pixels = False
    images_used = 0

    # перебираем все json в pred_dir
    for json_path in sorted(pred_dir.glob("*.json")):
        stem = json_path.stem  # например "3209_png.rf.xxx"

        gt_mask_path = gt_masks_dir / f"{stem}_mask.png"
        if not gt_mask_path.exists():
            # нет GT-маски для этой картинки – пропускаем
            continue

        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        gt_mask = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            continue

        h, w = gt_mask.shape[:2]

        gt_bin = (gt_mask == class_id)

        if not gt_bin.any():
            # в GT этого класса нет – этот кадр не учитываем для метрик по этому классу
            continue

        gt_has_pixels = True

        # предикт: ключ в json по имени класса
        if class_name == "wall":
            key = "walls"
        elif class_name == "door":
            key = "doors"
        elif class_name == "window":
            key = "windows"
        else:
            key = "walls"

        pred_bin = draw_class_mask_from_json(
            data,
            key=key,
            shape=(h, w),
            thickness=thickness,
        )

        images_used += 1

        # строгий IoU
        tp_local = np.logical_and(pred_bin, gt_bin).sum()
        fp_local = np.logical_and(pred_bin, np.logical_not(gt_bin)).sum()
        fn_local = np.logical_and(np.logical_not(pred_bin), gt_bin).sum()

        tp += tp_local
        fp += fp_local
        fn += fn_local

        # tolerant IoU (диляция)
        gt_u8 = gt_bin.astype(np.uint8)
        pred_u8 = pred_bin.astype(np.uint8)

        gt_d = cv2.dilate(gt_u8, kernel)
        pred_d = cv2.dilate(pred_u8, kernel)

        inter_tol += np.logical_and(gt_d, pred_d).sum()
        union_tol += np.logical_or(gt_d, pred_d).sum()

        # distance: от предикта до ближайшего GT
        if pred_bin.any():
            gt_inv = 1 - gt_u8
            dist_map = cv2.distanceTransform(gt_inv, cv2.DIST_L2, 3)
            dvals = dist_map[pred_bin]
            if dvals.size > 0:
                sum_dist += float(dvals.sum())
                count_dist += int(dvals.size)

    if not gt_has_pixels or images_used == 0:
        # для этого класса вообще нет GT или нечего считать
        return {
            "iou": 0.0,
            "iou_tolerant": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mean_distance": float("nan"),
            "images_used": float(images_used),
        }

    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    iou_tol = inter_tol / union_tol if union_tol > 0 else 0.0
    mean_dist = sum_dist / count_dist if count_dist > 0 else float("nan")

    return {
        "iou": iou,
        "iou_tolerant": iou_tol,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_distance": mean_dist,
        "images_used": float(images_used),
    }


def eval_backend(
    backend_name: str,
    gt_masks_dir: Path,
    pred_dir: Path,
    tolerant_radius: int,
    thickness: int,
) -> Dict[str, Dict[str, float]]:
    """
    backend_name: "opencv" или "unet".
    Возвращает словарь: class_name -> метрики.
    """
    results: Dict[str, Dict[str, float]] = {}

    if backend_name == "opencv":
        # только стены
        class_ids = [1]
    else:
        # unet: стены/двери/окна
        class_ids = [1, 2, 3]

    for cid in class_ids:
        cname = CLASS_INFO.get(cid, f"class_{cid}")
        metrics = compute_metrics_for_class(
            gt_masks_dir=gt_masks_dir,
            pred_dir=pred_dir,
            class_id=cid,
            class_name=cname,
            tolerant_radius=tolerant_radius,
            thickness=thickness,
        )
        results[cname] = metrics

    # mIoU по классам, где есть GT (для UNet)
    if backend_name != "opencv":
        ious = []
        for cname, m in results.items():
            if m["images_used"] > 0 and m["iou"] > 0 or m["iou_tolerant"] > 0:
                ious.append(m["iou"])
        if ious:
            results["mIoU"] = {"value": float(np.mean(ious))}
        else:
            results["mIoU"] = {"value": 0.0}

    return results


def write_metrics_txt(
    out_path: Path,
    backend_name: str,
    gt_masks_dir: Path,
    pred_dir: Path,
    metrics: Dict[str, Dict[str, float]],
    tolerant_radius: int,
    thickness: int,
) -> None:
    lines: List[str] = []

    lines.append(f"Backend: {backend_name}")
    lines.append(f"GT masks dir: {gt_masks_dir}")
    lines.append(f"Predictions dir: {pred_dir}")
    lines.append(f"Tolerant radius: {tolerant_radius}")
    lines.append(f"Line thickness: {thickness}")
    lines.append("")

    for cname, m in metrics.items():
        if cname == "mIoU":
            continue

        lines.append(f"Class: {cname}")
        lines.append(f"  images_used: {int(m.get('images_used', 0))}")
        lines.append(f"  IoU:          {m['iou']:.4f}")
        lines.append(f"  IoU_tolerant: {m['iou_tolerant']:.4f}")
        lines.append(f"  Precision:    {m['precision']:.4f}")
        lines.append(f"  Recall:       {m['recall']:.4f}")
        lines.append(f"  F1:           {m['f1']:.4f}")
        md = m["mean_distance"]
        if np.isnan(md):
            lines.append(f"  Mean distance (px):  NaN")
        else:
            lines.append(f"  Mean distance (px):  {md:.4f}")
        lines.append("")

    if "mIoU" in metrics:
        lines.append(f"mIoU (over present classes): {metrics['mIoU']['value']:.4f}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved metrics for {backend_name} to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate floorplan predictions (OpenCV / UNet) against GT masks"
    )
    parser.add_argument("--gt_masks_dir", type=str, required=True,
                        help="Папка с GT-масками (*_mask.png)")
    parser.add_argument("--pred_root_dir", type=str, required=True,
                        help="Папка с предсказаниями (содержит opencv_infer/ и/или unet_infer)")
    parser.add_argument("--tolerant_radius", type=int, default=2,
                        help="Радиус (в пикселях) для tolerant IoU (морф. диляция)")
    parser.add_argument("--line_thickness", type=int, default=5,
                        help="Толщина линии при растеризации полилиний в маску")

    args = parser.parse_args()

    gt_masks_dir = Path(args.gt_masks_dir)
    pred_root_dir = Path(args.pred_root_dir)

    backends = []

    opencv_dir = pred_root_dir / "opencv_infer"
    if opencv_dir.exists():
        backends.append(("opencv", opencv_dir))

    unet_dir = pred_root_dir / "unet_infer"
    if unet_dir.exists():
        backends.append(("unet", unet_dir))

    if not backends:
        raise RuntimeError(
            f"В {pred_root_dir} не найдены ни opencv_infer, ни unet_infer"
        )

    for backend_name, pred_dir in backends:
        metrics = eval_backend(
            backend_name=backend_name,
            gt_masks_dir=gt_masks_dir,
            pred_dir=pred_dir,
            tolerant_radius=args.tolerant_radius,
            thickness=args.line_thickness,
        )

        out_txt = pred_root_dir / f"metrics_{backend_name}.txt"
        write_metrics_txt(
            out_path=out_txt,
            backend_name=backend_name,
            gt_masks_dir=gt_masks_dir,
            pred_dir=pred_dir,
            metrics=metrics,
            tolerant_radius=args.tolerant_radius,
            thickness=args.line_thickness,
        )


if __name__ == "__main__":
    main()
