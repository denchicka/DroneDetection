from pathlib import Path
import os
import glob
import random
import cv2
from ultralytics import YOLO

from .utils import show_image, clean_memory
from .config import NEW_TRAIN_DIR


def load_onnx_models(onnx_paths: dict):
    """
    Загрузка набора ONNX-моделей Ultralytics YOLO.

    @param onnx_paths: dict
        Словарь с путями до ONNX-моделей.
        
        Пример:
        {
            "YOLOv8m": "exports/onnx/yolov8m.onnx",
            "YOLOv9m": "exports/onnx/yolov9m.onnx"
        }

    @return dict:
        Словарь:
        {
            "имя_модели": YOLO(onnx_path, task="detect"),
            ...
        }
    """

    models = {}  # Здесь будут лежать загруженные модели

    for name, onnx_path in onnx_paths.items():
        print(f"Загружаем ONNX-модель {name} из {onnx_path}")

        # Загружаем ONNX-модель для детекции
        models[name] = YOLO(str(onnx_path), task="detect")

    return models



def infer_random_images_onnx(models: dict, test_images_dir: str, n_files=5, imgsz=640):
    """
    Запуск инференса ONNX-моделей на случайных изображениях из директории.

    @param models: dict
        Словарь из подготовленных моделей.

    @param test_images_dir: str
        Путь к директории с изображениями (обычно .jpg).
        Пример: "dataset/images/val"

    @param n_files: int
        Количество случайных изображений для инференса.
        По умолчанию 5.

    @param imgsz: int
        Размер изображения для инференса (imgsz x imgsz).
        По умолчанию 640.

    @return None
        Функция отображает результаты инференса.
    """

    # Собираем список файлов .jpg
    files = glob.glob(os.path.join(test_images_dir, "*.jpg"))

    # Проверка наличия изображений
    if not files:
        print(f"Нет изображений в {test_images_dir}")
        return

    # Выбираем случайные изображения (не более n_files)
    images = random.sample(files, min(n_files, len(files)))

    # Проходим по каждому выбранному изображению
    for img_path in images:
        print(f"\nИзображение: {os.path.basename(img_path)}")

        # Прогоняем через все ONNX модели
        for model_name, model in models.items():

            # Инференс модели на одном изображении
            pred = model.predict(
                source=img_path,
                imgsz=imgsz,
                conf=0.25,
                verbose=False
            )[0]

            # Генерация картинки с нарисованными детекциями
            result_img = pred.plot()

            # Отображаем картинку
            show_image(
                result_img,
                title=f"{model_name} ONNX - {os.path.basename(img_path)}"
            )



def infer_videos_onnx(models: dict,
                      videos_dir: str,
                      project_root="runs",
                      conf=0.5,
                      n_videos=2,
                      imgsz=640):
    """
    Запуск инференса ONNX-моделей на видеофайлах и сохранение результатов.

    @param models: dict
        Словарь из подготовленных моделей.

    @param videos_dir: str
        Папка, содержащая видео.
        Видео выбираются по индексам от 1 до n_videos.

    @param project_root: str
        Корневая директория для сохранения результатов.
        По умолчанию "runs".

    @param conf: float
        Порог уверенности детекции.
        По умолчанию 0.5.

    @param n_videos: int
        Сколько видео нужно обработать.
        По умолчанию 2.

    @param imgsz: int
        Размер входного изображения для модели.
        По умолчанию 640.

    @return None
    """

    # Перебираем ONNX-модели
    for model_name, model in models.items():
        print(f"\nВидео-инференс ONNX: {model_name}")

        # Обрабатываем видео по номерам 1..n_videos
        for num in range(1, n_videos + 1):

            # Формируем путь к видео
            video_path = os.path.join(videos_dir, f"{num}.mp4")

            # Проверяем существование файла
            if not os.path.exists(video_path):
                print(f"Видео {video_path} не найдено, пропускаем.")
                continue

            print(f"{model_name} - {video_path}")

            # Выполняем инференс модели на видео
            model.predict(
                source=video_path,                     # входное видео
                conf=conf,                             # порог детекции
                imgsz=imgsz,                           # размер входа
                save=True,                             # сохранять результат
                project=f"{project_root}/{model_name}_ONNX",  # директория
                name=f"video_{num}",                   # поддиректория для каждого видео
                exist_ok=True                          # не создавать новые если уже есть
            )

            # Чистим память после обработки видео
            clean_memory()
