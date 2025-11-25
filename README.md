# Drone Detection: YOLOv8 / YOLOv9 / YOLOv11 + ONNX + TFLite

Репозиторий с исследованием и реализацией пайплайна детекции дронов:

- сравнение **YOLOv8s/m, YOLOv9m, YOLOv11m** на специализированном датасете дронов;
- полное обучение и анализ качества (mAP, Precision, Recall);
- экспорт в **ONNX** и **TensorFlow Lite**;
- бенчмарк скорости инференса по видео (**PyTorch GPU/CPU vs ONNX vs TFLite**);
- подготовка моделей для **мобильных и встраиваемых устройств**.

Проект оформлен как:
- Jupyter/Colab ноутбук для интерактивных экспериментов;
- Python-пакет `drone_yolo` с модульной структурой (обучение, инференс, экспорт, бенчмарки).

---

## Примеры детекции

<p align="center">
  <img src="docs/images/yolov11_drone.png" width="45%" />
  <img src="docs/images/yolov11_nodrone.png" width="45%" />
</p>

<p align="center">
  <i>Слева: YOLOv11m (PyTorch) с детекцией дрона. Справа: YOLOv11m (PyTorch) с детекцией птицы.</i>
</p>

<p align="center">
  <img src="docs/images/yolov11_tflite_drone.png" width="45%" />
  <img src="docs/images/yolov11_tflite_nodrone.png" width="45%" />
</p>

<p align="center">
  <i>Слева: YOLOv11m (TFLite) с детекцией дрона. Справа: YOLOv11m (TFLite) с детекцией птицы.</i>
</p>

---

## Содержание

1. [Описание проекта](#описание-проекта)
2. [Архитектура и стек](#архитектура-и-стек)
3. [Датасет](#датасет)
4. [Обучение моделей](#обучение-моделей)
5. [Метрики и результаты](#метрики-и-результаты)
6. [Экспорт и инференс](#экспорт-и-инференс-pytorch--onnx--tflite)
7. [Как запустить](#как-запустить)
8. [Структура репозитория](#структура-репозитория)
9. [Ограничения и будущая-работа](#ограничения-и-будущая-работа)
10. [Лицензия и благодарности](#лицензия-и-благодарности)
11. [Обратная связь](#обратная-связь)

---

## Описание проекта

Цель работы - исследовать качество и скорость современных моделей семейства YOLO (**YOLOv8, YOLOv9, YOLOv11**) для задачи обнаружения дронов и подготовить лёгкие модели для запуска на мобильных и встраиваемых устройствах.

В рамках проекта реализовано:

- обучение и сравнение моделей **YOLOv8s**, **YOLOv8m**, **YOLOv9m**, **YOLOv11m**
  на датасете дронов;
- сбор и анализ метрик качества (mAP50, mAP50-95, Precision, Recall, потери валидации);
- инференс по одиночным изображениям и видео (в т.ч. реальный FPS);
- экспорт моделей в **ONNX** и **TFLite**;
- сравнение скоростей инференса на **PyTorch (GPU/CPU)**, **ONNX Runtime** и **TFLite**.

Проект может использоваться как прототип системы видеомониторинга воздушного пространства.

---

## Архитектура и стек

Основные компоненты:

- **PyTorch + Ultralytics YOLO** - обучение и базовый инференс;
- **ONNX export + ONNX Runtime** - серверный / кросс-языковой инференс;
- **TensorFlow Lite** - мобильный и edge-инференс;
- **OpenCV** - работа с видео, конвертация AVI -> MP4, визуализация;
- **Roboflow** - загрузка и разметка датасета дронов;
- **Jupyter / Colab** - интерактивные эксперименты;
- **Python-пакет `drone_yolo`** - единый код для обучения, метрик, экспорта и бенчмарков.

Ключевой дизайн:

- вся логика разбита на модули (`train.py`, `metrics.py`, `infer_torch.py`,
  `infer_onnx.py`, `tflite_utils.py`, `video_utils.py`, `config.py`, `utils.py`);
- функции переиспользуются между ноутбуком и чистым Python-кодом;
- минимизация дублирования кода и несовпадающих параметров.

---

## Датасет

В работе используется датасет дронов из _`Roboflow`_:

- 2 класса: `drone`, `not_drone`;
- разметка в формате **YOLO** (bounding boxes + классы);
- обучающая, валидационная и тестовая выборки.

Загрузка в коде осуществляется через `Roboflow API`:

```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("dronenotdrone").project("drone-detection-pq8sj")
version = project.version(2)
dataset = version.download("yolov11")  # структура в стиле Ultralytics
```

Файл `data.yaml` содержит пути к train/val/test и список классов.

---

## Обучение моделей

Для сравнения используются следующие модели:

- `yolov8s.pt` - лёгкая модель (mobile-friendly);
- `yolov8m.pt` - средняя модель (баланс точность/скорость);
- `yolov9m.pt` - архитектура с улучшенными бэкбонами и head;
- `yolo11m.pt` - современная версия с attention-блоками.

Общие настройки обучения:

- `epochs = 150` (ранняя остановка по `patience = 30`);
- `imgsz = 640`;
- индивидуальный `batch_size` для каждой модели (8–32 в зависимости от размера);
- автоматическое сохранение чекпоинтов и `results.csv`.

Пример запуска обучения всех базовых моделей:

```python
from drone_yolo.train import train_all_main_models

train_all_main_models(
    yaml_path="./drone-detection-2/data.yaml",
    epochs=150,
    patience=30,
    img_size=640,
)

```

Отдельно обучается yolov8s как кандидат для TFLite-деплоя.

---

## Метрики и результаты

Для каждой модели из `runs/new_train/<model_name>/results.csv` автоматически собираются лучшие значения по mAP50-95:

- `metrics/mAP50(B)`
- `metrics/mAP50-95(B)`
- `metrics/precision(B)`
- `metrics/recall(B)`
- `val/box_loss`, `val/cls_loss`, `val/dfl_loss`

Пример агрегирования:

```python
from drone_yolo.metrics import collect_best_metrics, plot_map_comparison

results_df = collect_best_metrics(["yolov8s", "yolov8m", "yolov9m", "yolov11m"])
print(results_df)

plot_map_comparison(results_df)
```

Пример результатов:

| model    | best_epoch  | mAP50    | mAP50-95    | precision    | recall    | val_box_loss | val_cls_loss | val_dfl_loss |
| -------- | ---------:  | ----:    | -------:    | --------:    | -----:    | -----:       | -----:       | -----:       |
| yolov8s  |         93  |  0.91703 |     0.57403 |      0.86780 |   0.88704 |   1.43192    |    0.69005   |    1.13487   |
| yolov8m  |         117 |  0.91618 |     0.56285 |      0.86356 |   0.88893 |   1.41441    |    0.66287   |    1.31078   |
| yolov9m  |         107 |  0.90504 |     0.55292 |      0.85721 |   0.85717 |   1.33164    |    0.66751   |    1.33736   |
| yolov11m |         146 |  0.90856 |     0.56953 |      0.85588 |   0.85638 |   1.34088    |    0.65731   |    1.26081   |

---

## Экспорт и инференс (PyTorch / ONNX / TFLite)

### PyTorch (Ultralytics)

Стандартный инференс по изображениям и видео:

```python
from drone_yolo.infer_torch import load_trained_models, infer_random_images, infer_videos

models = load_trained_models()
infer_random_images(models, test_images_dir="./Drones-11/test/images")
infer_videos(models, videos_dir="./Drones-11/test/videos")
```

### ONNX

Экспорт:

```python
from drone_yolo.export_models import export_to_onnx

best_weights = {
    "YOLOv8s": "./runs/new_train/yolov8s/weights/best.pt",
    "YOLOv8m": "./runs/new_train/yolov8m/weights/best.pt",
    "YOLOv9m": "./runs/new_train/yolov9m/weights/best.pt",
    "YOLOv11m": "./runs/new_train/yolov11m/weights/best.pt",
}

onnx_paths = export_to_onnx(best_weights)
```

Инференс через Ultralytics YOLO(onnx):

```python
from drone_yolo.infer_onnx import load_onnx_models, infer_random_images_onnx

onnx_models = load_onnx_models(onnx_paths)
infer_random_images_onnx(onnx_models, test_images_dir="./Drones-11/test/images")
```

### TFLite

Экспорт:
 - для YOLOv8s/YOLOv8m/YOLOv9m/YOLOv11m успешно создаются `best_float32.tflite` и `best_float16.tflite`;
 - для YOLOv9m/YOLOv11m создаётся TFLite без встроенного NMS (требуется внешняя пост-обработка).

Инференс по видео с кастомным препроцессом и NMS:

```python
from drone_yolo.tflite_utils import run_tflite_on_video

stats = run_tflite_on_video(
    model_path="./runs/new_train/yolov8s/weights/best_saved_model/best_float32.tflite",
    video_path="./Drones-11/test/videos/7.mp4",
    save_video=True,
    measure_fps=True
)
print(stats)
```

Сравнение FPS:
 - PyTorch GPU - эталонная максимальная скорость;
 - PyTorch CPU / ONNX / TFLite - сравнение производительности на `CPU;
 - для TFLite дополнительно измеряется `FP32` vs `FP16`.

---

## Как запустить

### 1. Локальная установка

```bash
git clone https://github.com/denchicka/DroneDetection.git
cd drone-detection-yolo

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Запуск обучения

```bash
# Jupyter / VSCode / Colab:
# открыть notebook.ipynb и выполнить ячейки обучения
```

### 3. Бенчмарки и инференс

 - PyTorch / ONNX - через функции в `drone_yolo/infer_torch.py` и `drone_yolo/infer_onnx.py`;
 - TFLite - через `drone_yolo/tflite_utils.py` (run_tflite_on_video, detect_image_tflite).

---

## Структура репозитория

.
├─ drone_yolo/
│   ├─ __init__.py
│   ├─ config.py          # пути, глобальные настройки
│   ├─ train.py           # функции обучения моделей
│   ├─ metrics.py         # сбор и визуализация метрик
│   ├─ infer_torch.py     # инференс PyTorch (изображения и видео)
│   ├─ infer_onnx.py      # инференс ONNX (Ultralytics и/или ORT)
│   ├─ tflite_utils.py    # экспорт TFLite, препроцесс, NMS, инференс
│   ├─ video_utils.py     # работа с видео (конвертация AVI→MP4, вывод)
│   └─ utils.py           # clean_memory, show_image и т.п.
│
├─ notebook.ipynb         # основной исследовательский ноутбук
└─ README.md

---

## Ограничения и будущая работа

Ограничения:

- **YOLOv`x`m TFLite** генерирует "сырые" предсказания без встроенного NMS, что усложняет использование на мобильных устройствах;
- бенчмарки `ONNX/TFLite` проводились в основном на CPU одной машины - результаты зависят от конкретного железа.

Возможные направления развития:

- адаптация модели под реальные видеопотоки с IP-камер/дронов;
- оптимизация моделей через pruning / quantization-aware training;
- интеграция с мобильным приложением (Android / iOS) для демонстрации реального деплоя;
- расширение датасета дронов (разные типы, дальности, условия освещения, количество классов).

## Лицензия и благодарности

- Модели `YOLO` и код обучения основаны на библиотеке [Ultralytics](https://github.com/ultralytics/ultralytics).
- Датасет дронов получен через [Roboflow](https://roboflow.com/).
- Часть идей по экспорту и бенчмарку `ONNX/TFLite` вдохновлена официальной документацией `Ultralytics` и `ONNX Runtime`.

Лицензия проекта: MIT.

## Обратная связь

Если у Вас есть пожелания или вопросы, Вы можете связаться в личных сообщениях в мессенджере [Telegram](t.me/denchicka213)