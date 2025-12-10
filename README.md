# Floorplan -> JSON (OpenCV + UNet + OCR)

![Original / GT / OpenCV / UNet](assets/output.png)

Небольшой прототип пайплайна, который по изображению плана квартиры строит
структурированный JSON с геометрией стен, дверей, окон и результатами OCR.

Используются два подхода:

- классический CV (OpenCV) для выделения стен в виде сегментов;
- UNet (ResNet34, 4 класса) для сегментации стен, дверей и окон;
- EasyOCR для распознавания текстов и размеров.

Датасет: [CubiCasa5k (COCO версия)](https://universe.roboflow.com/wall-segmentation-pj9zt/cubicasa5k-2-qpmsa-tppfc/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true)

---

## Структура репозитория

```text
.
├── assets/
│   └── overview.png               # пример коллажа Original / GT / OpenCV / UNet / OCR
├── notebooks/
│   ├── 01_eda_cubicasa5k.ipynb    # EDA датасета, примеры разметки (gitignored)
│   └── viz_predictions.ipynb      # визуализация GT / OpenCV / UNet / OCR (gitignored)
├── src/
│   ├── data_preprocess.py         # COCO аннотации -> GT маски 0/1/2/3
│   ├── train_unet.py              # обучение UNet (ResNet34 encoder, 4 класса)
│   ├── cv_walls.py                # классический CV бейзлайн по стенам
│   ├── ocr.py                     # EasyOCR + повороты/флипы + NMS
│   ├── pipeline_cv.py             # общий инференс: CV / UNet / OCR -> JSON
│   └── eval.py                    # сравнение JSON предсказаний с GT масками
├── data/                          # сюда нужно положить CubiCasa5k COCO
│   └── cubicasa5k-2.v1i.coco/     # train / valid / test / *_annotations.coco.json
├── masks/                         # сгенерированные GT маски 0/1/2/3
│   ├── train/
│   ├── valid/
│   └── test/
├── model/                         # сюда сохраняются веса UNet и логи
│   ├── unet_resnet34_best.pth
│   ├── unet_resnet34_last.pth
│   └── train_log.txt
├── outputs/                       # инференс + метрики
│   ├── viz_valid_100/             # json примеры для визуализации (cv + unet + ocr)
│   └── test_pred/                 # json предсказания для теста + metrics_*.txt
├── requirements.txt               # зависимости проекта
├── README.md
└── .gitignore
```

### Формат GT масок

После препроцессинга каждая маска имеет 4 класса:

- `0` - фон  
- `1` - стена  
- `2` - дверь  
- `3` - окно  

Маски строятся по COCO аннотациям на основе bbox, чтобы получить толстые сегменты для обучения UNet.

---

## Установка

Рекомендуется Python 3.10.

```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows:
# .venv\Scriptsctivate

pip install -r requirements.txt
```

Все основные зависимости описаны в `requirements.txt`.  

Если нужен конкретный билд PyTorch под свою CUDA или только CPU, можно:

1. Удалить из `requirements.txt` строки с `torch` и `torchvision`.
2. Поставить PyTorch по инструкции с официального сайта.
3. Установить остальные зависимости:

```bash
pip install -r requirements.txt --no-deps
```

---

## Как запустить

### 0. Подготовить данные

Скачать COCO версию CubiCasa5k и распаковать в:

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

Генерируем GT маски 0,1,2,3 по COCO аннотациям:

```bash
python src/data_preprocess.py   --data_root data/cubicasa5k-2.v1i.coco   --out_root masks
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
python src/train_unet.py   --train_images_dir data/cubicasa5k-2.v1i.coco/train   --train_masks_dir  masks/train   --val_images_dir   data/cubicasa5k-2.v1i.coco/valid   --val_masks_dir    masks/valid   --out_dir          model   --epochs 30   --batch_size 4   --lr 1e-3   --num_workers 4   --patience 3
```

Скрипт логирует loss и mIoU в `model/train_log.txt` и сохраняет:

- `model/unet_resnet34_best.pth` - лучшая модель по val mIoU;
- `model/unet_resnet34_last.pth` - модель после последней эпохи.

---

### 3. Инференс на test для метрик (OpenCV + UNet, без OCR)

Для подсчета метрик по геометрии OCR не нужен, поэтому отключаем его флагом `--no_ocr`.  

Запускаем инференс и CV, и UNet сразу:

```bash
python src/pipeline_cv.py   --input_dir  data/cubicasa5k-2.v1i.coco/test   --output_dir outputs/test_pred   --use_cv   --use_unet   --device cuda   --no_ocr
```

Если нет GPU, можно заменить `--device cuda` на `--device cpu` или просто не указывать этот флаг (режим `auto` сам выберет).

Результат:

```text
outputs/test_pred/opencv_infer/*.json   # стены от OpenCV
outputs/test_pred/unet_infer/*.json     # стены, двери, окна от UNet
```

По умолчанию `pipeline_cv.py` ищет веса UNet в `model/unet_resnet34_best.pth`.  
Если веса лежат в другом месте, можно явно указать путь:

```bash
python src/pipeline_cv.py   --input_dir  data/cubicasa5k-2.v1i.coco/test   --output_dir outputs/test_pred   --use_unet   --device cuda   --no_ocr   --unet_model_path path/to/unet_resnet34_best.pth
```

---

### 4. Оценка качества (eval.py)

Сравниваем JSON предсказания с GT масками.  

Скрипт сам понимает, какие подпапки есть в `pred_root_dir` (`opencv_infer` и `unet_infer`) и считает метрики отдельно для OpenCV и для UNet.

```bash
python src/eval.py   --gt_masks_dir  masks/test   --pred_root_dir outputs/test_pred   --tolerant_radius 2   --line_thickness 5
```

Результаты сохраняются в:

- `outputs/test_pred/metrics_opencv.txt`
- `outputs/test_pred/metrics_unet.txt`

---

### 5. Инференс + OCR для визуализаций (valid, 100 случайных картинок)

Для наглядных примеров с OCR выбираем 100 случайных картинок из валидации:

```bash
python src/pipeline_cv.py   --input_dir  data/cubicasa5k-2.v1i.coco/valid   --output_dir outputs/viz_valid_100   --use_cv   --use_unet   --device cuda   --max_images 100
```

OCR по умолчанию включен (флаг `--no_ocr` не задан), поэтому в JSON добавятся поля с распознанным текстом.  

После этого в ноутбуке `notebooks/viz_predictions.ipynb` можно отрисовать
коллажи GT / OpenCV / UNet / OCR как на картинке в `assets/overview.png`.

---

## Результаты

Метрики посчитаны на тестовом наборе из 400 изображений (CubiCasa5k COCO версия),  
при параметрах `tolerant_radius = 2`, `line_thickness = 5`.

### UNet (стены + двери + окна)

| metric             | wall   | door   | window |
|--------------------|--------|--------|--------|
| images_used        | 400    | 399    | 398    |
| IoU                | 0.3074 | 0.3230 | 0.2917 |
| IoU_tolerant       | 0.4567 | 0.4519 | 0.4477 |
| Precision          | 0.4741 | 0.4080 | 0.4782 |
| Recall             | 0.4664 | 0.6078 | 0.4279 |
| F1                 | 0.4702 | 0.4882 | 0.4516 |
| Mean distance (px) | 3.61   | 12.63  | 11.75  |

Средний mIoU по всем классам, где есть GT: **0.3073**

### OpenCV бейзлайн (только стены)

| metric             | wall   |
|--------------------|--------|
| images_used        | 400    |
| IoU                | 0.1002 |
| IoU_tolerant       | 0.1883 |
| Precision          | 0.1648 |
| Recall             | 0.2036 |
| F1                 | 0.1822 |
| Mean distance (px) | 22.63  |

---

## Почему метрики могут быть занижены

Важно понимать, что эти значения IoU и F1 нельзя напрямую сравнивать с цифрами из статей по CubiCasa5k.  
Причины, по которым метрики здесь немного занижены относительно классической dense сегментации:

1. Исходная разметка CubiCasa5k представляет собой толстые области (стены, двери, окна в виде полигонов).
2. В этом прототипе и GT, и предсказания приводятся к линейному представлению:
   - стены, двери и окна описываются как полилинии и отрезки;
   - при генерации JSON происходит сильное упрощение геометрии.
3. Для оценки JSON сначала обратно растеризуется:
   - линии раскрашиваются в маску фиксированной толщины (`line_thickness`, по умолчанию 5 пикселей);
   - затем по этой тонкой маске считается:
     - строгий IoU;
     - tolerant IoU с морфологической диляцией радиусом `tolerant_radius` (по умолчанию 2 пикселя);
     - Precision, Recall, F1;
     - средняя дистанция от предсказанных пикселей до ближайшего GT пикселя.
4. Так как GT стены обычно толще, чем наша линизация, даже геометрически корректные линии дают меньше перекрытие по площади, чем при оценке по исходным толстым маскам.

Из-за этого метрики скорее занижают качество, чем завышают его.  
Они специально настроены под задачу "линейная геометрия -> JSON", а не под классическую dense сегментацию.

---

## Идеи для улучшения

UNet / сегментация:

- более сильный энкодер (ResNet50, EfficientNet) и возможная двухэтапная схема:
  грубая сегментация -> refinement вдоль линий;
- аугментации, специфичные для чертежей:
  perspective warp, морфологический шум, инверсия, изменение контраста, легкий blur;
- лоссы с акцентом на границы (Boundary loss, Lovasz, focal IoU и т.п.).

Пост обработка / векторизация:

- склеивание соседних отрезков в длинные полилинии;
- "снэппинг" координат к горизонталям и вертикалям;
- фильтрация коротких шумовых сегментов по длине / площади;
- поиск помещений как полигонов по сегментации стен.

OCR / текст и размеры:

- более умная фильтрация OCR:
  - разделение на типы (размеры vs названия комнат);
  - привязка размеров к ближайшей стене или комнате;
- для production варианта - более современные OCR модели (PaddleOCR, TrOCR и т.п.) и дообучение на кусочках чертежей.

"Грязные" фото:

- автовыравнивание: поиск горизонта по Hough Lines, коррекция поворота;
- для перспективных искажений - оценка гомографии по внешнему прямоугольнику планировки и нормализация входа.


