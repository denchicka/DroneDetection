from ultralytics import YOLO
from pathlib import Path
from .config import NEW_TRAIN_DIR, IMG_SIZE
from .utils import clean_memory

def export_to_onnx(model_weights: dict, opset=12, dynamic=True):
    """
    Экспорт списка PyTorch-моделей YOLO в формат ONNX.

    @param model_weights: dict
        Словарь с весами моделей.

        Пример:
        {
            "yolov8m": "runs/new_train/yolov8m/weights/best.pt",
            "yolov9m": "runs/new_train/yolov9m/weights/best.pt"
        }

    @param opset: int
        Версия ONNX opset.  
        По умолчанию 12 - наиболее совместимая и стабильная
        для экспорта YOLO.

    @param dynamic: bool
        Включает динамические размеры входа.
        Если True - модель принимает произвольные размеры (N, H, W).
        Если False - фиксированный размер imgsz.

    @return dict:
        Словарь вида:
        {
            "имя_модели": "путь_к_onnx_файлу",
            ...
        }
    """

    onnx_paths = {}  # Словарь для сохранения путей к экспортированным ONNX-файлам

    for name, wpath in model_weights.items():
        print(f"\nЭкспорт {name} в ONNX")

        model = YOLO(str(wpath))   # Загружаем PyTorch модель по пути

        # Экспорт в ONNX
        onnx_file = model.export(
            format="onnx",  # формат вывода
            imgsz=IMG_SIZE, # размер входного изображения
            opset=opset,    # версия ONNX opset
            dynamic=dynamic # включить динамический input shape
        )

        onnx_paths[name] = onnx_file  # Сохраняем путь к полученному ONNX

        del model        # Удаляем модель из RAM
        clean_memory()   # Чистим RAM + VRAM (если нужно)

    return onnx_paths    # Возвращаем словарь путей


def export_to_tflite(model_weights: dict, imgsz=320):
    """
    Экспорт PyTorch-моделей YOLO в формат TFLite (FP32 и FP16).

    @param model_weights: dict
        Словарь с весами моделей.

        Пример:
        {
            "yolov8s": "runs/new_train/yolov8s/weights/best.pt"
        }

    @param imgsz: int
        Размер входного изображения для экспорта.
        По умолчанию 320 - оптимально для TFLite устройств.

    @return tuple(dict, dict):
        1) export_status:
           {
               "model": {
                   "fp32": "OK" / "NO FILE" / "ERROR: ...",
                   "fp16": "OK" / "NO FILE" / "ERROR: ..." / "SKIP(FP32 failed)"
               }
           }

        2) tflite_paths:
           {
               "model_fp32": Path(...),
               "model_fp16": Path(...)
           }
    """

    export_status = {}  # Статусы экспортов по моделям
    tflite_paths = {}   # Пути к созданным .tflite файлам

    # Перебираем модели
    for name, wpath in model_weights.items():
        print(f"\nЭкспорт TFLite: {name}")

        export_status[name] = {"fp32": None, "fp16": None}

        # FP32 EXPORT
        try:
            model = YOLO(str(wpath))      # Загружаем модель
            out = model.export(
                format="tflite",          # Экспорт в TFLite
                imgsz=imgsz,              # Размер входа
                half=False,               # FP32
                verbose=False
            )
            tfile = Path(out)             # Путь к результату

            # Проверяем существование файла
            if tfile.exists():
                export_status[name]["fp32"] = "OK"
                tflite_paths[f"{name}_fp32"] = tfile
            else:
                export_status[name]["fp32"] = "NO FILE"

        except Exception as e:
            export_status[name]["fp32"] = f"ERROR: {e}"

        del model
        clean_memory()

        # FP16 EXPORT
        if export_status[name]["fp32"] == "OK":
            try:
                model = YOLO(str(wpath))
                out = model.export(
                    format="tflite",
                    imgsz=imgsz,
                    half=True,            # FP16
                    verbose=False
                )
                tfile = Path(out)

                if tfile.exists():
                    export_status[name]["fp16"] = "OK"
                    tflite_paths[f"{name}_fp16"] = tfile
                else:
                    export_status[name]["fp16"] = "NO FILE"

            except Exception as e:
                export_status[name]["fp16"] = f"ERROR: {e}"

            del model
            clean_memory()

        else:
            # FP16 невозможно экспортировать, если FP32 не собрался
            export_status[name]["fp16"] = "SKIP(FP32 failed)"

    return export_status, tflite_paths
