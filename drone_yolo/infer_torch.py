import glob
import os
import cv2
import random
from ultralytics import YOLO
from pathlib import Path
from .utils import show_image, clean_memory
from .config import NEW_TRAIN_DIR

def load_trained_models(model_names):
    """
    Загрузка обученных PyTorch-моделей YOLO из директории runs/new_train/.

    @param model_names: list or iterable
        Список имён моделей, которые нужно загрузить.
        Каждый элемент соответствует имени папки обучения.

        Пример:
        ["yolov8m", "yolov9m", "yolo11m"]

    @return dict:
        {
            "model_name": YOLO(model_path),
            ...
        }
    """

    models = {}  # Словарь загруженных моделей

    for name in model_names:

        # Путь к файлу весов:
        # runs/new_train/<model_name>/weights/best.pt
        weights = NEW_TRAIN_DIR / name / "weights" / "best.pt"

        print(f"Загружаем модель {name} из {weights}")

        # Загружаем модель YOLO
        models[name] = YOLO(str(weights))

    return models


def infer_random_images(models: dict, test_images_dir: str, n_files=5, imgsz=640):
    """
    Запуск инференса PyTorch-моделей YOLO на случайных изображениях из директории.

    @param models: dict
        Словарь из подготовленных моделей.

    @param test_images_dir: str
        Путь к директории, где лежат изображения (.jpg).

    @param n_files: int
        Количество случайно выбираемых изображений.
        По умолчанию 5.

    @param imgsz: int
        Размер входного изображения для инференса (imgsz x imgsz).
        По умолчанию 640.

    @return None
        Функция выводит изображения, но ничего не возвращает.
    """

    # Получаем список всех JPG изображений в директории
    files = glob.glob(os.path.join(test_images_dir, "*.jpg"))

    # Выбираем случайные изображения (не больше n_files)
    images = random.sample(files, min(n_files, len(files)))

    # Прогоняем инференс по выбранным изображениям
    for img_path in images:
        print(f"\nИзображение: {os.path.basename(img_path)}")

        for model_name, model in models.items():

            # Предсказание модели на изображении
            pred = model.predict(
                source=img_path,
                imgsz=imgsz
            )[0]

            # Получение изображения с нарисованными боксами
            result_img = pred.plot()

            # Показ результата
            show_image(
                result_img,
                title=f"{model_name} - {os.path.basename(img_path)}"
            )


def infer_videos(models: dict, videos_dir: str, project_root="runs", conf=0.5, n_videos=3):
    """
    Запуск инференса PyTorch-моделей YOLO на видеофайлах и сохранение результатов.

    @param models: dict
        Словарь из подготовленных моделей.

    @param videos_dir: str
        Путь к директории с видеофайлами.
        Обрабатываются первые n_videos.

    @param project_root: str
        Корневая директория для сохранения результатов инференса.
        По умолчанию "runs".

    @param conf: float
        Порог уверенности детекции.
        По умолчанию 0.5.

    @param n_videos: int
        Сколько пронумерованных видео нужно обработать.
        По умолчанию 3.

    @return None
    """

    # Цикл по моделям
    for model_name, model in models.items():
        print(f"\nВидео-инференс: {model_name}")

        # Обработка видео по номерам 1..n_videos
        for num in range(1, n_videos + 1):
            video_path = os.path.join(videos_dir, f"{num}.mp4")

            # Проверка существования файла
            if not os.path.exists(video_path):
                print(f"Видео {video_path} не найдено, пропускаем.")
                continue

            print(f"{model_name} - {video_path}")

            # Запуск инференса
            model.predict(
                source=video_path,          # путь к видео
                conf=conf,                  # порог детекции
                save=True,                  # сохранить результат
                project=f"{project_root}/{model_name}",  # папка сохранения
                name=f"video_{num}",        # подпапка
                exist_ok=True               # разрешить использование существующей папки
            )

            # Чистим память между обработкой файлов
            clean_memory()
