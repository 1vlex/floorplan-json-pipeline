# src/pipeline_cv.py
import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

from cv_walls import walls_from_image, walls_from_mask
from ocr import OCREngine

# ===== UNet utils =====



def polygons_from_mask(
    class_mask: np.ndarray,
    min_area: float = 50.0,
    eps_ratio: float = 0.01,
) -> List[Dict[str, Any]]:
    """
    Превращает маску одного класса (0/1) в список полигонов:
        { "id": "w1", "points": [[x1,y1], [x2,y2], ...] }

    Используется для UNet, чтобы не схлопывать всё в одну прямую.
    """
    mask = (class_mask > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    items: List[Dict[str, Any]] = []
    idx = 1

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        peri = cv2.arcLength(cnt, True)
        epsilon = eps_ratio * peri
        approx = cv2.approxPolyDP(cnt, epsilon, True)  # (N,1,2)

        pts = approx.reshape(-1, 2).tolist()
        if len(pts) < 2:
            continue

        items.append(
            {
                "id": f"w{idx}",
                "points": pts,
            }
        )
        idx += 1

    return items


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_unet_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """Загружаем UNet (ResNet34 encoder) с обученными весами."""
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=4,  # 0 bg, 1 wall, 2 door, 3 window
    )
    ckpt = torch.load(model_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def unet_predict_mask(
    image_bgr: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
) -> np.ndarray:
    """BGR-картинка -> предсказанная маска классов 0..3 (H, W)."""
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img = img_rgb.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))  # CHW

    x = torch.from_numpy(img).unsqueeze(0).to(device)  # 1,C,H,W
    with torch.no_grad():
        logits = model(x)  # 1,4,H,W
        pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
    return pred  # H,W in {0,1,2,3}


def unet_segments_from_image(
    image_bgr: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
) -> Dict[str, List[Dict[str, Any]]]:
    pred_mask = unet_predict_mask(image_bgr, model, device)

    walls_mask   = (pred_mask == 1).astype(np.uint8)
    doors_mask   = (pred_mask == 2).astype(np.uint8)
    windows_mask = (pred_mask == 3).astype(np.uint8)

    walls   = polygons_from_mask(walls_mask,   min_area=80.0)
    doors   = polygons_from_mask(doors_mask,   min_area=30.0)
    windows = polygons_from_mask(windows_mask, min_area=30.0)

    return {
        "walls": walls,
        "doors": doors,
        "windows": windows,
    }

# ===== main pipeline =====


def run_folder(
    input_dir: str,
    output_dir: str,
    use_cv: bool,
    use_unet: bool,
    device: torch.device,
    ocr_gpu: bool,
    use_ocr: bool = True,
    max_images: Optional[int] = None,
    unet_model_path: Optional[str] = None,
) -> None:
    in_dir = Path(input_dir)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # собираем список картинок
    exts = {".jpg", ".jpeg", ".png"}
    images = [p for p in in_dir.iterdir() if p.suffix.lower() in exts]

    if not images:
        print(f"Нет изображений в {in_dir}")
        return

    # случайный сабсет, если max_images задан
    if max_images is not None and max_images > 0 and max_images < len(images):
        images = random.sample(images, max_images)

    # OCR-движок один на все картинки
    ocr_engine = None
    if use_ocr:
        ocr_engine = OCREngine(
            lang="eng+rus",
            min_conf=0.4,   # EasyOCR: conf в [0,1], внутри конвертим в проценты
            digits_only=False,
            scale=2,
            gpu=ocr_gpu,
        )

    # UNet-модель одна на все картинки
    unet_model = None
    if use_unet:
        if unet_model_path is None:
            # ищем модель по умолчанию: <project_root>/model/unet_resnet34_best.pth
            project_root = Path(__file__).resolve().parent.parent
            default_model = project_root / "model" / "unet_resnet34_best.pth"
            if not default_model.exists():
                raise FileNotFoundError(
                    f"UNet model path is not specified and default "
                    f"{default_model} does not exist. "
                    f"Укажи --unet_model_path явно."
                )
            unet_model_path = str(default_model)

        unet_model = load_unet_model(unet_model_path, device)

    # создаём только те папки, которые реально нужны
    cv_out_dir = None
    unet_out_dir = None

    if use_cv:
        cv_out_dir = out_root / "opencv_infer"
        cv_out_dir.mkdir(parents=True, exist_ok=True)

    if use_unet:
        unet_out_dir = out_root / "unet_infer"
        unet_out_dir.mkdir(parents=True, exist_ok=True)

    processed = 0

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"ERROR: cannot read image {img_path}")
            continue

        # OCR считаем один раз
        ocr_items: List[Dict[str, Any]] = []
        if ocr_engine is not None:
            try:
                ocr_items = ocr_engine.extract_texts(img)
            except Exception as e:
                print(f"ERROR OCR on {img_path.name}: {e}")
                ocr_items = []

        # CV walls -> JSON (только стены)
        if use_cv and cv_out_dir is not None:
            try:
                walls_cv = walls_from_image(img)
                result_cv = {
                    "meta": {
                        "source": img_path.name,
                        "backend": "opencv",
                    },
                    "walls": walls_cv,
                    "ocr": ocr_items,
                }
                out_json_cv = cv_out_dir / f"{img_path.stem}.json"
                with out_json_cv.open("w", encoding="utf-8") as f:
                    json.dump(result_cv, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"ERROR CV on {img_path.name}: {e}")

        # UNet walls/doors/windows -> JSON
        if use_unet and unet_out_dir is not None and unet_model is not None:
            try:
                seg = unet_segments_from_image(img, unet_model, device)
                result_unet = {
                    "meta": {
                        "source": img_path.name,
                        "backend": "unet_resnet34",
                    },
                    "walls": seg["walls"],
                    "doors": seg["doors"],
                    "windows": seg["windows"],
                    "ocr": ocr_items,
                }
                out_json_unet = unet_out_dir / f"{img_path.stem}.json"
                with out_json_unet.open("w", encoding="utf-8") as f:
                    json.dump(result_unet, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"ERROR UNet on {img_path.name}: {e}")

        processed += 1

    print(f"Готово: обработано {processed} файлов. Результаты в {out_root}")


def main():
    parser = argparse.ArgumentParser(
        description="Floorplan inference: OpenCV walls + UNet walls/doors/windows + shared OCR"
    )
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument(
        "--use_cv",
        action="store_true",
        help="Включить инференс OpenCV-бейзлайна",
    )
    parser.add_argument(
        "--use_unet",
        action="store_true",
        help="Включить инференс UNet (ResNet34)",
    )

    parser.add_argument("--no_ocr", action="store_true")
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Максимальное число изображений для обработки. Если не указано - все.",
    )
    parser.add_argument(
        "--unet_model_path",
        type=str,
        default=None,
        help="Путь к unet_resnet34_best.pth. "
             "Если не указан, ищется в <project_root>/model/unet_resnet34_best.pth",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="На каком девайсе гонять UNet и OCR: auto (по умолчанию), cpu или cuda.",
    )
    args = parser.parse_args()

    # если никто не выбран - по умолчанию только CV
    use_cv = args.use_cv
    use_unet = args.use_unet
    if not use_cv and not use_unet:
        use_cv = True
        use_unet = False

    # выбираем девайс и флаг для OCR
    if args.device == "cpu":
        device = torch.device("cpu")
        ocr_gpu = False
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA недоступна, но указан --device cuda")
        device = torch.device("cuda")
        ocr_gpu = True
    else:  # auto
        if torch.cuda.is_available():
            device = torch.device("cuda")
            ocr_gpu = True
        else:
            device = torch.device("cpu")
            ocr_gpu = False

    run_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        use_cv=use_cv,
        use_unet=use_unet,
        device=device,
        ocr_gpu=ocr_gpu,
        use_ocr=not args.no_ocr,
        max_images=args.max_images,
        unet_model_path=args.unet_model_path,
    )


if __name__ == "__main__":
    main()
