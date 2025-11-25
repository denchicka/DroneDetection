from pathlib import Path


# Пути и основные директории проекта

# Корневая директория проекта
ROOT = Path(".").resolve()

# Путь к датасету Drones-11
DATA_DIR = ROOT / "Drones-11"

# Путь к файлу описания датасета (data.yaml)
DATA_YAML = DATA_DIR / "data.yaml"

# Директория для всех результатов обучения
RUNS_DIR = ROOT / "runs"

# Путь к директории, где будут храниться новые эксперименты
NEW_TRAIN_DIR = RUNS_DIR / "new_train"


# Модели для обучения (имя весов, batch_size)
TRAIN_MODELS = {
    "yolov8m.pt": 16,
    "yolov9m.pt": 10,
    "yolo11m.pt": 8,
}

# Дополнительная модель, обучаемая отдельно под TFLite, для сравнения (имя весов, batch_size)
EXTRA_TFLITE_MODEL = {
    "yolov8s.pt": 32,
}

# Конфигурация обучения

# Размер входного изображения (input resolution)
IMG_SIZE = 640

# Максимальное число эпох обучения
EPOCHS = 150

# Раннее прекращение обучения (количество эпох без улучшения метрики, после которых training остановится)
PATIENCE = 30

# Интервал сохранения чекпоинтов (каждые N эпох)
SAVE_PERIOD = 10

# Список имён классов для датасета
CLASS_NAMES = ['not_drone', 'drone']
