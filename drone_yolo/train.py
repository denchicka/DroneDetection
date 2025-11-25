import os
from pathlib import Path
from ultralytics import YOLO
from .config import DATA_YAML, TRAIN_MODELS, EXTRA_TFLITE_MODEL, NEW_TRAIN_DIR, IMG_SIZE, EPOCHS, PATIENCE, SAVE_PERIOD
from .utils import clean_memory

def train_single_model(model_name: str, batch_size: int, project_dir: Path, run_name: str):
    """
    Обучение одной модели YOLO по указанным параметрам.

    @param model_name: str
        Имя модели или путь к .pt весам.

    @param batch_size: int
        Размер batch для обучения.

    @param project_dir: Path
        Папка, в которой будет создан каталог обучения Ultralytics.

    @param run_name: str
        Имя подпапки внутри `project_dir`, куда будут записаны логи,
        веса best.pt, results.csv и графики обучения.

    @return:
        Объект результатов обучения.
    """

    print(f"\nОбучаем {model_name} (batch={batch_size}) - {run_name}")

    # Загружаем YOLO модель по имени .pt файла
    model = YOLO(model_name)

    # Запуск обучения
    results = model.train(
        data=str(DATA_YAML),     # путь к data.yaml (классы, пути train/val)
        epochs=EPOCHS,           # максимальное число эпох
        patience=PATIENCE,       # ранняя остановка
        imgsz=IMG_SIZE,          # размер изображений
        batch=batch_size,        # указанный batch size
        save_period=SAVE_PERIOD, # сохранять чекпоинт каждые N эпох
        project=str(project_dir),# корневой каталог обучения
        name=run_name,           # подпапка эксперимента
    )

    # Освобождаем память после обучения
    del model
    clean_memory()

    return results


def train_all_main_models():
    """
    Запуск обучения всех основных моделей, перечисленных в TRAIN_MODELS.

    TRAIN_MODELS имеет формат:
        {
            "yolov8m.pt": 16,
            "yolov9m.pt": 10,
            "yolo11m.pt": 8
        }

    Для каждой модели:
        - создаётся директория runs/new_train/ (если ещё нет)
        - имя эксперимента = stem модели (например, 'yolov8m')
        - запускается train_single_model()

    @return None
    """

    # Убедимся, что директория для экспериментов существует
    NEW_TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    # Перебор моделей для обучения
    for model_name, batch in TRAIN_MODELS.items():

        # Имя подпапки = имя модели без .pt
        run_name = Path(model_name).stem   # 'yolov8m', 'yolov9m', 'yolo11m'

        # Обучаем модель
        train_single_model(
            model_name=model_name,
            batch_size=batch,
            project_dir=NEW_TRAIN_DIR,
            run_name=run_name
        )

def train_v11_model():
    """
    Запуск обучения только модели yolo11m.pt
    """
    NEW_TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    model_name = "yolo11m.pt"
    batch = TRAIN_MODELS[model_name]          # 8
    run_name = Path(model_name).stem          # "yolo11m"

    train_single_model(
        model_name=model_name,
        batch_size=batch,
        project_dir=NEW_TRAIN_DIR,
        run_name=run_name
    )



def train_extra_tflite_models():
    """
    Обучение дополнительных моделей, подготовленных для экспорта в TFLite.

    Для каждой модели:
        - создаётся директория runs/new_train/ (если её ещё нет)
        - run_name = stem(model_name) → напр. 'yolov8s'
        - вызывается train_single_model()

    @return None
    """

    # Создаём директорию для экспериментов, если её ещё нет
    NEW_TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    # Перебираем TFLite-ориентированные модели
    for model_name, batch in EXTRA_TFLITE_MODEL.items():

        # Имя подпапки, совпадающее с моделью без .pt
        run_name = Path(model_name).stem   # 'yolov8s'

        # Запуск обучения
        train_single_model(
            model_name=model_name,
            batch_size=batch,
            project_dir=NEW_TRAIN_DIR,
            run_name=run_name
        )
