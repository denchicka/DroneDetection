import time
import cv2
from ultralytics import YOLO
from pathlib import Path
from .utils import clean_memory
from .tflite_infer import run_tflite_on_video

def benchmark_video_ultra(model, model_name, video_path,
                          imgsz=640, conf=0.5,
                          device=None,
                          tag="PyTorch"):
    """
    Бенчмарк скорости обработки видео моделью (FPS).

    @param model:       Загруженная модель.
    @param model_name:  Имя модели (для вывода).
    @param video_path:  Путь к видеофайлу, который нужно обработать.
    @param imgsz:       Размер входного изображения для инференса (по умолчанию 640).
    @param conf:        Порог уверенности (confidence threshold) для предсказаний модели.
    @param device:      Устройство инференса ('cpu', 'cuda:0' и т.п.). Если None - используется дефолт.
    @param tag:         Текстовая метка для вывода результатов (например 'PyTorch', 'ONNX', 'TensorRT').

    @return float:      Средний FPS (кадров в секунду).
    """

    # Открываем видео по указанному пути
    cap = cv2.VideoCapture(video_path)

    # Счётчик обработанных кадров
    frame_count = 0

    # Фиксируем время старта
    t0 = time.time()

    # Основной цикл построчного чтения видео
    while True:
        ret, frame = cap.read()          # Считываем кадр
        if not ret:                      # Если кадр не считан - видео закончилось
            break

        # Параметры для model.predict()
        kwargs = dict(
            source=frame,                # Кадр для инференса
            imgsz=imgsz,                 # Размер входного изображения
            conf=conf,                   # Порог детекции
            verbose=False                # Отключаем лишний вывод
        )

        # Добавляем устройство, если оно указано
        if device is not None:
            kwargs["device"] = device

        _ = model.predict(**kwargs)      # Инференс на кадре
        frame_count += 1                 # Инкрементируем количество кадров

    # Освобождаем видеоресурс
    cap.release()

    # Полное время обработки
    total_time = time.time() - t0

    # Подсчёт FPS
    fps = frame_count / total_time if total_time > 0 else 0

    # Печать итоговых результатов
    print(f"[{tag}] {model_name} | Frames: {frame_count}, Time: {total_time:.2f}s, FPS: {fps:.2f}")

    return fps # Возвращаем средний FPS

def benchmark_pytorch_models(best_weights: dict, video_path: str):
    """
    Бенчмарк PyTorch-моделей (YOLO) на CPU и GPU.

    @param best_weights: dict  
        Словарь формата: { "имя_модели": "путь_к_весам" }.  
        Например: { "yolo_n": "weights/yolo_n.pt", "yolo_s": "weights/yolo_s.pt" }

    @param video_path: str  
        Путь к видеофайлу, который будет использоваться для тестирования FPS.

    @return tuple(dict, dict):
        CPU_FPS, GPU_FPS - два словаря вида:
        {
            "имя_модели": fps_value,
            ...
        }
    """

    cpu_fps, gpu_fps = {}, {}     # Словари для хранения FPS на CPU и GPU

    # Перебираем модели по именам и путям
    for name, wpath in best_weights.items():

        # GPU тестирование 
        print(f"\nPyTorch GPU - {name}")
        model = YOLO(str(wpath))   # Загружаем модель

        # Бенчмарк модели на GPU
        gpu_fps[name] = benchmark_video_ultra(
            model,                     # модель
            name,                      # имя (для вывода)
            video_path,                # путь к видео
            imgsz=640,                 # размер входа
            conf=0.5,                  # порог детекции
            device="cuda",             # устройство - GPU
            tag="PyTorch-CUDA"         # тег для печати
        )

        del model                     # Удаляем модель из памяти
        clean_memory()                # Чистим VRAM + RAM

        # CPU тестирование
        print(f"\nPyTorch CPU - {name}")
        model = YOLO(str(wpath))      # Перезагружаем модель

        # Бенчмарк модели на CPU
        cpu_fps[name] = benchmark_video_ultra(
            model,
            name,
            video_path,
            imgsz=640,
            conf=0.5,
            device="cpu",             # устройство - CPU
            tag="PyTorch-CPU"
        )

        del model                     # Удаляем модель
        clean_memory()                # Чистим память

    return cpu_fps, gpu_fps           # Возвращаем FPS на CPU и GPU


def benchmark_onnx_models(onnx_paths: dict, video_path: str):
    """
    Бенчмарк ONNX-моделей (Ultralytics YOLO) на CPU.

    @param onnx_paths: dict  
        Словарь формата:  
        {
            "имя_модели": "путь_к_ONNX_файлу",
            ...
        }  
        Пример:  
        {  
            "yolo_8s": "export/yolo_8s.onnx",  
            "yolo_8m": "export/yolo_8m.onnx"  
        }

    @param video_path: str  
        Путь к видеофайлу, которое будет проигрываться покадрово для измерения FPS.

    @return dict:  
        Словарь вида:
        { "имя_модели": fps }
    """

    onnx_fps = {}   # Словарь для хранения FPS

    # Перебираем все указанные ONNX модели
    for name, onnx_path in onnx_paths.items():

        print(f"\nONNX-Ultralytics - {name}")

        # Загружаем ONNX-модель через Ultralytics YOLO
        model = YOLO(onnx_path, task="detect")

        # Бенчмарк модели (ONNX обычно работает только на CPU у Ultralytics)
        onnx_fps[name] = benchmark_video_ultra(
            model,              # модель
            name,               # имя для вывода
            video_path,         # путь к видео
            imgsz=320,          # размер входа
            conf=0.5,           # порог детекции
            device=None,        # ONNX backend сам выбирает устройство (обычно CPU)
            tag="ONNX-Ultra"    # тег для консольного вывода
        )

        del model              # Удаляем модель из оперативной памяти
        clean_memory()         # Чистим память (RAM)

    return onnx_fps            # Возвращаем словарь FPS


def benchmark_tflite_models(bench_targets: dict, video_path: str):
    """
    Бенчмарк TFLite-моделей (FP32 и FP16) на видео.

    @param bench_targets: dict  
        Структура словаря:  
        {
            "имя_модели": {
                "fp32": "path/to/model_fp32.tflite" или None,
                "fp16": "path/to/model_fp16.tflite" или None
            },
            ...
        }  
        Пример:
        {
            "yolo_n": {
                "fp32": "export/yolo_n_fp32.tflite",
                "fp16": "export/yolo_n_fp16.tflite"
            }
        }

    @param video_path: str  
        Путь к видеофайлу, которое будет использоваться для замера FPS.

    @return dict:  
        Словарь вида:
        {
            "model_name": {
                "fp32": stats_dict,
                "fp16": stats_dict
            }
        }
    """

    tflite_fps = {}   # Словарь для хранения статистики по всем моделям и форматам

    # Перебираем все модели
    for model_name, formats in bench_targets.items():
        tflite_fps[model_name] = {}   # Для каждой модели создаём вложенный словарь

        # FP32
        if formats["fp32"]:
            stats = run_tflite_on_video(
                model_path=formats["fp32"],      # путь к fp32 .tflite
                video_path=video_path,           # путь к видео
                model_name=model_name + "_FP32", # имя модели
                save_video=True,                 # сохранять ли визуализацию
                measure_fps=True                 # измерять FPS
            )
            tflite_fps[model_name]["fp32"] = stats

        # FP16
        if formats["fp16"]:
            stats = run_tflite_on_video(
                model_path=formats["fp16"],   # путь к fp16 .tflite
                video_path=video_path,
                model_name=model_name + "_FP16",
                save_video=True,
                measure_fps=True
            )
            tflite_fps[model_name]["fp16"] = stats

    return tflite_fps
