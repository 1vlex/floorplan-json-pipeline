# src/cv_walls.py
from typing import List, Dict, Any

import cv2
import numpy as np


def get_wall_mask(image_bgr: np.ndarray) -> np.ndarray:
    """
    Бинарная маска стен: 0 фон, 255 линии/стены.
    Стараемся вытащить ВСЕ тёмные линии (черные/серые).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    #Выравниваем контраст (тонкие серые линии становятся заметнее)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    #Чуть размываем шум
    blur = cv2.GaussianBlur(gray_eq, (3, 3), 0)

    # два бинаризатора: Otsu и adaptive

    _, bin_otsu = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    bin_adapt = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        5,
    )

    #Объединяем
    bin_img = cv2.bitwise_or(bin_otsu, bin_adapt)

    # Морфология: закрываем разрывы и чуть наращиваем стены
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=1)
    dilated = cv2.dilate(closed, kernel, iterations=1)

    mask = np.where(dilated > 0, 255, 0).astype(np.uint8)
    return mask



def walls_from_mask(
    wall_mask: np.ndarray,
    min_length: float = 80.0,
) -> List[Dict[str, Any]]:
    """
    Находим стены как длинные отрезки с помощью HoughLinesP.
    Возвращаем список:
      { "id": "w1", "points": [[x1, y1], [x2, y2]] }
    """
    # Canny по маске
    edges = cv2.Canny(wall_mask, 50, 150, apertureSize=3)

    # Hough-прямая
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=60,
        minLineLength=min_length,
        maxLineGap=8,
    )

    walls: List[Dict[str, Any]] = []
    if lines is None:
        return walls

    wid = 1
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < min_length:
            continue

        walls.append(
            {
                "id": f"w{wid}",
                "points": [[int(x1), int(y1)], [int(x2), int(y2)]],
            }
        )
        wid += 1

    return walls


def walls_from_image(
    image_bgr: np.ndarray,
    min_length: float = 80.0,
) -> List[Dict[str, Any]]:
    mask = get_wall_mask(image_bgr)
    walls = walls_from_mask(mask, min_length=min_length)
    return walls
