# Floorplan -> JSON (OpenCV + UNet + OCR)

![Original / GT / OpenCV / UNet](assets/output.png)

Небольшой прототип пайплайна, который по изображению плана квартиры строит
структурированный JSON с геометрией стен/дверей/окон и результатами OCR.

Используются два подхода:

- классический CV (OpenCV) для выделения стен в виде сегментов;
- UNet (ResNet34, 4 класса) для сегментации стен / дверей / окон;
- EasyOCR для распознавания текстов и размеров.

Датасет: [CubiCasa5k (COCO-версия)](https://universe.roboflow.com/wall-segmentation-pj9zt/cubicasa5k-2-qpmsa-tppfc/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true)

---

## Структура репозитория

```text
.
├── assets/
│   └── output.png                 # коллаж Original / GT / OpenCV / UNet / OCR
├── notebooks/
│   ├── 01_eda_cubicasa5k.ipynb    # EDA датасета, примеры разметки
│   └── viz_predictions.ipynb      # визуализация GT / OpenCV / UNet / OCR
├── src/
│   ├── data_preprocess.py         # COCO-аннотации -> GT-маски 0/1/2/3
│   ├── train_unet.py              # обучение UNet (ResNet34 encoder, 4 класса)
│   ├── cv_walls.py                # классический CV-бейзлайн по стенам
│   ├── ocr.py                     # EasyOCR + повороты/флипы + NMS
│   ├── pipeline_cv.py             # общий инференс: CV / UNet / OCR -> JSON
│   └── eval.py                    # сравнение JSON-предсказаний с GT-масками
├── data/                          # сюда нужно положить CubiCasa5k COCO
│   └── cubicasa5k-2.v1i.coco/     # train / valid / test / *_annotations.coco.json
├── masks/                         # сгенерированные GT-маски 0/1/2/3
│   ├── train/
│   ├── valid/
│   └── test/
├── model/                         # веса обученного UNet(Слишком много весят, не плоложил), логи 
│   ├── unet_resnet34_best.pth
│   ├── unet_resnet34_last.pth
│   └── train_log.txt
├── outputs/                       # инференс + метрики
│   ├── unet/                      # json-примеры для визуализации (unet + ocr)
│   └── OpenCV/                    # json-примеры для визуализации (OpenCV + ocr)
                                   # + metrics_*.txt
├── README.md
└── .gitignore
```

### Формат GT-масок

После препроцессинга каждая маска имеет 4 класса:

- `0` - фон  
- `1` - стена  
- `2` - дверь  
- `3` - окно  

Маски строятся по COCO-аннотациям на основе bbox, чтобы получить толстые сегменты для обучения UNet.

---

## Установка

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
# source .venv/bin/activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # или под свою CUDA/CPU
pip install segmentation-models-pytorch
pip install opencv-python numpy matplotlib tqdm
pip install easyocr
```

Версия Python использовалась 3.10.

---

## Как запустить (пошагово)

### 0. Подготовить данные

Скачать COCO-версию CubiCasa5k и распаковать в:

```text
data/cubicasa5k-2.v1i.coco/
  train/
  valid/
  test/
  train/_annotations.coco.json
  valid/_annotations.coco.json
  test/_annotations.coco.json
```

Путь можно поменять, но в примерах ниже он именно такой.

---

### 1. COCO -> маски (walls / doors / windows)

```bash
python src/data_preprocess.py \
  --data_root data/cubicasa5k-2.v1i.coco \
  --out_root masks
```

Результат:

```text
masks/train/*_mask.png
masks/valid/*_mask.png
masks/test/*_mask.png
```

---

### 2. Обучение UNet

Используем train / valid разбиение, уже имеющееся в датасете:

```bash
python src/train_unet.py \
  --train_images_dir data/cubicasa5k-2.v1i.coco/train \
  --train_masks_dir  masks/train \
  --val_images_dir   data/cubicasa5k-2.v1i.coco/valid \
  --val_masks_dir    masks/valid \
  --out_dir          model \
  --epochs 30 \
  --batch_size 4 \
  --lr 1e-3 \
  --num_workers 4 \
  --patience 3
```

Скрипт логирует loss и mIoU в `model/train_log.txt` и сохраняет:

- `model/unet_resnet34_best.pth` - лучшая модель по val mIoU (на валидации);
- `model/unet_resnet34_last.pth` - модель после последней эпохи.

---

### 3. Инференс на test (без OCR, только геометрия)

Чтобы посчитать объективные метрики, OCR не нужен:

```bash
python src/pipeline_cv.py \
  --input_dir  data/cubicasa5k-2.v1i.coco/test \
  --output_dir outputs/test_pred \
  --use_cv \
  --use_unet \
  --device cuda \
  --no_ocr
```

Результат:

```text
outputs/test_pred/opencv_infer/*.json   # стены от OpenCV
outputs/test_pred/unet_infer/*.json     # стены / двери / окна от UNet
```

---

### 4. Оценка качества (eval.py)

Сравниваем JSON-предсказания с GT-масками. Скрипт сам понимает, какие папки есть
(`opencv_infer` / `unet_infer`), и считает метрики для каждой.

```bash
python src/eval.py \
  --gt_masks_dir  masks/test \
  --pred_root_dir outputs/test_pred \
  --tolerant_radius 2 \
  --line_thickness 5
```

Результаты сохраняются в:

- `outputs/test_pred/metrics_opencv.txt`
- `outputs/test_pred/metrics_unet.txt`

---

### 5. Инференс + OCR для визуализаций (valid, 100 случайных картинок)

```bash
python src/pipeline_cv.py \
  --input_dir  data/cubicasa5k-2.v1i.coco/valid \
  --output_dir outputs/viz_valid_100 \
  --use_cv \
  --use_unet \
  --device cuda \
  --max_images 100
```

После этого в ноутбуке `notebooks/viz_predictions.ipynb` можно отрисовать
коллажи GT / OpenCV / UNet / OCR (пример - `assets/output.png`).

---

## Результаты

### Обучение UNet (валидация)

Лучший val mIoU по стенам / дверям / окнам на валидации примерно 0.66  
(точное значение и номер эпохи видно в `model/train_log.txt`).

### Тестовый набор (400 изображений)

Метрики считаются на тонких линиях: JSON-полилинии сначала растеризуются
в маску фиксированной толщины (по умолчанию 5 пикселей), затем считаются:

- строгий IoU;
- tolerant IoU с морфологической диляцией (радиус 2 пикселя);
- Precision / Recall / F1;
- средняя дистанция до ближайшего GT-пикселя.

#### UNet (стены + двери + окна)

| metric              | wall   | door   | window |
|---------------------|--------|--------|--------|
| images_used         | 400    | 399    | 398    |
| IoU                 | 0.3074 | 0.3230 | 0.2917 |
| IoU_tolerant        | 0.4567 | 0.4519 | 0.4477 |
| Precision           | 0.4741 | 0.4080 | 0.4782 |
| Recall              | 0.4664 | 0.6078 | 0.4279 |
| F1                  | 0.4702 | 0.4882 | 0.4516 |
| Mean distance (px)  | 3.61   | 12.63  | 11.75 |

mIoU по всем классам, где есть GT: **0.3073**

#### OpenCV-бейзлайн (только стены)

| metric              | wall   |
|---------------------|--------|
| images_used         | 400    |
| IoU                 | 0.0919 |
| IoU_tolerant        | 0.1745 |
| Precision           | 0.1862 |
| Recall              | 0.1536 |
| F1                  | 0.1683 |
| Mean distance (px)  | 21.18  |

---

## Замечание про метрики

Метрики здесь немного отличаются от классического формата CubiCasa5k и  
скорее занижают качество, чем завышают:

- исходная разметка датасета - это толстые области (dense-сегментация стен / дверей / окон);
- в этом прототипе для генерации JSON и GT, и предсказания приводятся к линейному представлению:
  стены / двери / окна описываются как полилинии и отрезки;
- для оценки:
  - JSON-линии растеризуются обратно в маску фиксированной толщины (на тонких стенах получается меньше перекрытие с GT-областью, чем у исходной толщины);
  - считаются строгий IoU по маскам и tolerant IoU с диляцией;
  - дополнительно считаются Precision / Recall / F1 и средняя дистанция.

Поэтому абсолютные значения IoU / F1 нельзя напрямую сравнивать с цифрами из статей по CubiCasa5k.  
Это метрики именно под задачу "линии -> JSON-геометрия", и они скорее занижают оценку, чем раздувают ее.

---

## Идеи для улучшения

UNet / сегментация:

- более сильный энкодер (ResNet50 / EfficientNet) и/или двухэтапная схема:
  грубая сегментация -> refinement вдоль линий;
- аугментации, специфичные для чертежей:
  perspective warp, морфологический шум, инверсия, изменение контраста, легкий blur;
- loss с акцентом на границы (Boundary loss, Lovasz, focal IoU и т.п.).

Пост-обработка / векторизация:

- склеивание соседних отрезков в длинные полилинии;
- "снэппинг" координат к горизонталям / вертикалям;
- фильтрация коротких шумовых сегментов по длине / площади;
- поиск помещений как полигонов по сегментации стен.

OCR / текст и размеры:

- более умная фильтрация OCR:
  - разделение на типы (dimension vs название комнаты);
  - привязка размеров к ближайшей стене / комнате;
- для production-варианта - более современные OCR-модели (PaddleOCR, TrOCR и т.п.) и дообучение на кусочках чертежей.

"Грязные" фото:

- автоворовнявание: поиск горизонта по Hough Lines, коррекция поворота;
- для перспективных искажений - оценка гомографии по внешнему прямоугольнику планировки и нормализация входа.

