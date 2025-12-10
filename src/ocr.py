# src/ocr.py
from typing import List, Dict, Any
import re

import cv2
import numpy as np
import easyocr


def bbox_iou(a: Dict[str, float], b: Dict[str, float]) -> float:
    """IoU для прямоугольников {x1,y1,x2,y2}."""
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])

    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    inter = w * h
    if inter <= 0:
        return 0.0

    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0

    return inter / union


def invert_transform(
    pts: np.ndarray,
    w: int,
    h: int,
    mode: str,
) -> np.ndarray:
    """
    Переводит координаты bbox из преобразованного изображения
    в базовую систему (масштабированное исходное).

    pts: (N, 2) в координатах трансформированного изображения
    w, h: ширина/высота базового изображения (до трансформации)
    mode: 'orig', 'rot90', 'rot180', 'rot270', 'flip_h', 'flip_v'
    """
    x_p = pts[:, 0]
    y_p = pts[:, 1]

    if mode == "orig":
        x = x_p
        y = y_p

    elif mode == "rot90":
        # src (x, y) -> dst (x' = h-1-y, y' = x)
        # обратное: x = y', y = h-1-x'
        x = y_p
        y = h - 1 - x_p

    elif mode == "rot180":
        # src (x, y) -> dst (x' = w-1-x, y' = h-1-y)
        # обратное то же самое
        x = w - 1 - x_p
        y = h - 1 - y_p

    elif mode == "rot270":
        # src (x, y) -> dst (x' = y, y' = w-1-x)
        # обратное: x = w-1-y', y = x'
        x = w - 1 - y_p
        y = x_p

    elif mode == "flip_h":
        # горизонтальное зеркало:
        # src (x, y) -> dst (x' = w-1-x, y' = y)
        x = w - 1 - x_p
        y = y_p

    elif mode == "flip_v":
        # вертикальное зеркало:
        # src (x, y) -> dst (x' = x, y' = h-1-y)
        x = x_p
        y = h - 1 - y_p

    else:
        # на всякий случай
        x = x_p
        y = y_p

    return np.stack([x, y], axis=1)


class OCREngine:
    def __init__(
        self,
        lang: str = "eng+rus",
        min_conf: float = 0.2, # EasyOCR даёт conf в [0, 1]
        tesseract_cmd: str | None = None, # оставляем для совместимости, не используем
        digits_only: bool = False,
        scale: int = 2,
        gpu: bool = False,
        try_rotations: bool = True, # пробовать 0/90/180/270
        try_flips: bool = True, # пробовать зеркалку (гориз./верт.)
        nms_iou: float = 0.2, # порог "очень близко" для объединения
    ) -> None:
        langs: List[str] = []
        low = lang.lower()
        if "en" in low or "eng" in low:
            langs.append("en")
        if "ru" in low or "rus" in low:
            langs.append("ru")

        self.reader = easyocr.Reader(langs, gpu=gpu)
        self.min_conf = float(min_conf)
        self.digits_only = digits_only
        self.scale = max(1, int(scale))
        self.try_rotations = try_rotations
        self.try_flips = try_flips
        self.nms_iou = float(nms_iou)

    def _build_variants(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """
        Возвращает список вариантов для OCR:
        [
          {"mode": "orig",  "image": img0},
          {"mode": "rot90", "image": img_r90},
          ...
        ]
        """
        h, w = img.shape[:2]

        variants: List[Dict[str, Any]] = []
        variants.append({"mode": "orig", "image": img})

        if self.try_rotations:
            variants.append(
                {
                    "mode": "rot90",
                    "image": cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
                }
            )
            variants.append(
                {
                    "mode": "rot180",
                    "image": cv2.rotate(img, cv2.ROTATE_180),
                }
            )
            variants.append(
                {
                    "mode": "rot270",
                    "image": cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
                }
            )

        if self.try_flips:
            variants.append(
                {
                    "mode": "flip_h",
                    "image": cv2.flip(img, 1),
                }
            )
            variants.append(
                {
                    "mode": "flip_v",
                    "image": cv2.flip(img, 0),
                }
            )

        # w, h нужны как размеры базового (до трансформаций)
        for v in variants:
            v["base_w"] = w
            v["base_h"] = h

        return variants

    def extract_texts(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        h0, w0 = image_bgr.shape[:2]

        img = image_bgr
        if self.scale > 1:
            img = cv2.resize(
                img,
                (w0 * self.scale, h0 * self.scale),
                interpolation=cv2.INTER_CUBIC,
            )

        base_h, base_w = img.shape[:2]
        variants = self._build_variants(img)

        candidates: List[Dict[str, Any]] = []

        for var in variants:
            img_var = var["image"]
            mode = var["mode"]

            results = self.reader.readtext(
                img_var,
                detail=1,
                paragraph=False,
            )

            for res in results:
                if len(res) != 3:
                    continue
                bbox, text, conf = res

                if conf < self.min_conf:
                    continue
                text = text.strip()
                if not text:
                    continue

                if not any(ch.isalnum() for ch in text):
                    continue

                if self.digits_only and not re.fullmatch(r"\d+(\.\d+)?", text):
                    continue

                # bbox: 4 точки в координатах трансформированного изображения
                pts_var = np.array(bbox, dtype=np.float32)  # (4, 2)
                # переводим в координаты базового (масштабированного) изображения
                pts_base = invert_transform(
                    pts_var,
                    w=base_w,
                    h=base_h,
                    mode=mode,
                )

                xs = pts_base[:, 0]
                ys = pts_base[:, 1]

                x1 = float(xs.min())
                x2 = float(xs.max())
                y1 = float(ys.min())
                y2 = float(ys.max())

                # подрезаем по границам
                x1 = max(0.0, min(x1, base_w - 1.0))
                x2 = max(0.0, min(x2, base_w - 1.0))
                y1 = max(0.0, min(y1, base_h - 1.0))
                y2 = max(0.0, min(y2, base_h - 1.0))

                if x2 <= x1 or y2 <= y1:
                    continue

                candidates.append(
                    {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "text": text,
                        "conf": float(conf),
                    }
                )

        # NMS: оставляем самые уверенные детекции
        # дубли (перекрывающиеся боксы) выкидываем
        candidates.sort(key=lambda d: d["conf"], reverse=True)
        selected: List[Dict[str, Any]] = []

        for cand in candidates:
            keep = True
            for s in selected:
                if bbox_iou(cand, s) >= self.nms_iou:
                    keep = False
                    break
            if keep:
                selected.append(cand)

        out: List[Dict[str, Any]] = []
        tid = 1

        for cand in selected:
            x1_s = cand["x1"]
            x2_s = cand["x2"]
            y1_s = cand["y1"]
            y2_s = cand["y2"]

            x1 = x1_s / self.scale
            x2 = x2_s / self.scale
            y1 = y1_s / self.scale
            y2 = y2_s / self.scale

            x = int(round(x1))
            y = int(round(y1))
            w_box = int(round(x2 - x1))
            h_box = int(round(y2 - y1))
            if w_box <= 0 or h_box <= 0:
                continue

            text = cand["text"].strip()
            if not text:
                continue

            if re.fullmatch(r"\d+(\.\d+)?", text):
                ttype = "dimension"
            else:
                ttype = "other"

            out.append(
                {
                    "id": f"t{tid}",
                    "bbox": [x, y, w_box, h_box],
                    "text": text,
                    "confidence": float(cand["conf"] * 100.0),  # 0..100
                    "type": ttype,
                }
            )
            tid += 1


        return out
