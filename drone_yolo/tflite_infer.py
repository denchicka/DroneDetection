from pathlib import Path
import cv2
import time
import tensorflow as tf
from .tflite_postprocess import letterbox, decode_tflite_output, yolo_nms, draw_detections
from .config import CLASS_NAMES

def create_tflite_interpreter(model_path: Path):
    """
    Создание и инициализация TFLite-интерпретатора для инференса.

    @param model_path: Path
        Путь к .tflite модели.
        Пример:
            Path("exports/tflite/yolov8s_fp16.tflite")

    @return tf.lite.Interpreter
        Готовый к использованию TFLite-интерпретатор.
        После вызова функции тензоры уже выделены (allocate_tensors).
    """

    # Создаём интерпретатор из TFLite-модели
    interpreter = tf.lite.Interpreter(model_path=str(model_path))

    # Выделяем память под тензоры (обязательно перед инференсом)
    interpreter.allocate_tensors()

    # Возвращаем готовый объект интерпретатора
    return interpreter


def infer_tflite_frame(interpreter, frame, conf_thres=0.25, iou_thres=0.7):
    """
    Выполнение инференса одного кадра через TFLite-интерпретатор.

    @param interpreter: tf.lite.Interpreter
        Инициализированный TFLite-интерпретатор,
        созданный через create_tflite_interpreter().

    @param frame: np.ndarray (BGR)
        Кадр изображения (обычно исходный кадр из видео).

    @param conf_thres: float
        Порог уверенности для NMS.
        По умолчанию 0.25.

    @param iou_thres: float
        Порог IoU для NMS.
        По умолчанию 0.7.

    @return list:
        Список детекций в формате:
        [
            ((x1, y1, x2, y2), score, class_id),
            ...
        ]
        Координаты - в системе оригинального кадра.
    """

    # Получаем входные и выходные тензоры
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Входная форма интерпретатора (H x W), dtype входа
    H, W = input_details[0]['shape'][1:3]
    dtype = input_details[0]['dtype']

    
    # PREPROCESS: letterbox -> RGB -> нормализация

    # Функция letterbox возвращает:
    # resized - кадр с паддингом
    # r       - коэффициент масштабирования
    # pad_x/y - отступы паддинга
    resized, r, (pad_x, pad_y) = letterbox(frame, (H, W))

    # Перевод BGR -> RGB
    resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Нормализация в диапазон [0..1] + приведение к нужному типу
    inp = resized_rgb.astype("float32") / 255.0
    inp = inp[None, ...].astype(dtype)  # добавляем batch dim

    # INFERENCE: передаём вход и запускаем модель

    interpreter.set_tensor(input_details[0]['index'], inp)
    interpreter.invoke()

    # Выход модели - обычно shape: (1, N, 6)
    raw = interpreter.get_tensor(output_details[0]['index'])

    # POSTPROCESS: декодирование + NMS

    # Декодируем YOLO-предсказания TFLite (bbox, score, class)
    pred = decode_tflite_output(raw)  # (N, 6)

    # Выполняем NMS
    detections = yolo_nms(
        pred,
        img_w=W, img_h=H,
        conf_thres=conf_thres,
        iou_thres=iou_thres
    )

    # Восстановление координат из letterbox -> оригинальный кадр

    corrected = []
    for (x1, y1, x2, y2), sc, cl in detections:

        # Убираем паддинг + делим на коэффициент масштабирования
        x1o = int((x1 - pad_x) / r)
        y1o = int((y1 - pad_y) / r)
        x2o = int((x2 - pad_x) / r)
        y2o = int((y2 - pad_y) / r)

        corrected.append(((x1o, y1o, x2o, y2o), sc, cl))

    return corrected


def run_tflite_on_video(model_path,
                        video_path,
                        model_name="TFLite",
                        save_video=False,
                        measure_fps=False,
                        conf_thres=0.25,
                        iou_thres=0.7,
                        out_dir="runs/detect/tflite_custom"):
    """
    Запуск TFLite-модели на видеофайле с опциональным сохранением результата и замером FPS.

    @param model_path: str or Path
        Путь к .tflite модели.

    @param video_path: str
        Путь к входному видео.
        
    @param model_name: str
        Имя модели.

    @param save_video: bool
        Если True - сохраняет видео с прорисованными детекциями.

    @param measure_fps: bool
        Если True - считает FPS и возвращает статистику.

    @param conf_thres: float
        Порог уверенности для NMS.

    @param iou_thres: float
        Порог IoU для NMS.

    @param out_dir: str
        Папка, куда сохраняется итоговое видео.

    @return dict:
        {
            "fps": ...,
            "frames": ...,
            "status": "OK"
        }
    """

    # Загружаем TFLite-модель
    interpreter = create_tflite_interpreter(Path(model_path))

    # Открываем видео и читаем первый кадр
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Не удалось прочитать видео")
        cap.release()
        return None

    orig_h, orig_w = frame.shape[:2]

    # Настраиваем сохранение видео
    if save_video:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"{model_name}_{Path(video_path).stem}_tflite.mp4"

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(out_path),
            fourcc,
            30,
            (orig_w, orig_h)
        )

        print(f"Видео будет сохранено в: {out_path}")
    else:
        out = None

    # Переменные для FPS
    frame_idx = 0
    t0 = time.time()

    # Основной цикл обработки кадров
    while True:

        # Первый кадр уже прочитан -> читаем только со 2-го
        if frame_idx > 0:
            ret, frame = cap.read()
            if not ret:
                break

        frame_idx += 1

        # Инференс модели на кадре
        detections = infer_tflite_frame(
            interpreter,
            frame,
            conf_thres=conf_thres,
            iou_thres=iou_thres
        )

        # Рисуем детекции
        draw_frame = draw_detections(frame, detections, CLASS_NAMES)

        # Если нужно - записываем кадр в выходное видео
        if out is not None:
            out.write(draw_frame)

    # Освобождаем ресурсы
    cap.release()
    if out is not None:
        out.release()

    # FPS метрики
    if measure_fps:
        total_time = time.time() - t0
        fps = frame_idx / (total_time + 1e-9)
        print(f"[TFLite] {model_name} | frames={frame_idx}, fps={fps:.2f}")

        return {
            "fps": fps,
            "frames": frame_idx,
            "status": "OK"
        }

    # Если FPS не считался
    return {
        "frames": frame_idx,
        "status": "OK"
    }


def detect_image_tflite(model_path, image_path, conf_thres=0.25, iou_thres=0.7):
    """
    Детекция объектов на одном изображении с помощью TFLite-модели.

    @param model_path: str or Path
        Путь к .tflite модели.

    @param image_path: str
        Путь к изображению (обычно JPEG/PNG).

    @param conf_thres: float
        Порог уверенности для NMS.
        По умолчанию 0.25.

    @param iou_thres: float
        Порог IoU для NMS.
        По умолчанию 0.7.

    @return np.ndarray (BGR)
        Изображение с нарисованными детекциями.
    """

    # Загружаем TFLite-интерпретатор
    interpreter = create_tflite_interpreter(Path(model_path))

    # Параметры входного тензора
    input_details = interpreter.get_input_details()
    H, W = input_details[0]['shape'][1:3]   # ожидаемый размер модели

    # Загружаем изображение
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")

    img_h, img_w = img.shape[:2]

    # PREPROCESS - resize -> RGB -> normalize
    resized = cv2.resize(img, (W, H))
    resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Масштабируем в [0..1], добавляем batch dim
    inp = resized_rgb.astype("float32") / 255.0
    inp = inp[None, ...]   # shape: (1, H, W, 3)

    # 4. INFERENCE - подаём вход и запускаем модель
    interpreter.set_tensor(input_details[0]['index'], inp)
    interpreter.invoke()

    # Получаем сырое предсказание
    raw = interpreter.get_tensor(
        interpreter.get_output_details()[0]['index']
    )

    # POSTPROCESS - декодирование + NMS
    pred = decode_tflite_output(raw)

    detections = yolo_nms(
        pred,
        img_w=W,
        img_h=H,
        conf_thres=conf_thres,
        iou_thres=iou_thres
    )

    # Масштабирование координат обратно
    scale_x = img_w / W
    scale_y = img_h / H

    corrected = []
    for (x1, y1, x2, y2), sc, cl in detections:
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)
        corrected.append(((x1_scaled, y1_scaled, x2_scaled, y2_scaled), sc, cl))

    # Рисуем финальные боксы
    result_img = draw_detections(img, corrected, CLASS_NAMES)

    return result_img
