import numpy as np
import cv2

def letterbox(im, new_shape=(320, 320), color=(114, 114, 114)):
    """
    Масштабирование изображения с сохранением пропорций (letterbox),
    добавлением паддинга и выравниванием под требуемый размер модели.

    @param im: np.ndarray (BGR)
        Исходное изображение.

    @param new_shape: tuple(int, int)
        Желаемый размер изображения (H, W) после масштабирования и паддинга.

    @param color: tuple(int, int, int)
        Цвет паддинга. По умолчанию (114,114,114), как в YOLO.

    @return:
        resized_image - выходное изображение с паддингами
        r             - коэффициент масштабирования (используется для обратного преобразования координат)
        (dw, dh)      - половина паддинга по ширине и высоте
    """

    # Исходные размеры входного изображения (h, w)
    shape = im.shape[:2]

    # Коэффициент масштабирования - выбираем минимальный,
    # чтобы вписать изображение в new_shape без обрезки
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Размер изображения после масштабирования без паддинга
    new_unpad = (int(shape[1] * r), int(shape[0] * r))  # (new_w, new_h)

    # Количество паддинга по каждой оси
    dw = new_shape[1] - new_unpad[0]   # width padding total
    dh = new_shape[0] - new_unpad[1]   # height padding total

    # Паддинг делим поровну слева/справа и сверху/снизу
    dw /= 2
    dh /= 2

    # Масштабируем изображение
    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    # int(dh) - верхний паддинг
    # int(dh + 0.5) - нижний паддинг (компенсация округления)
    top, bottom = int(dh), int(dh + 0.5)

    # Аналогично для ширины
    left, right = int(dw), int(dw + 0.5)

    # Добавляем бордеры (паддинг)
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color
    )

    return im, r, (dw, dh)


def nms_xyxy(boxes, scores, iou_thres=0.5):
    """
    Простая реализация NMS (Non-Maximum Suppression) в формате XYXY.

    @param boxes: np.ndarray, shape (N, 4)
        Массив боксов в формате (x1, y1, x2, y2).

    @param scores: np.ndarray, shape (N,)
        Оценки (confidence score) для каждого бокса.

    @param iou_thres: float
        Порог IoU для подавления пересекающихся боксов.
        По умолчанию 0.5.

    @return list[int]:
        Индексы боксов, которые прошли NMS (оставлены).
        Можно использовать как boxes[keep], scores[keep].
    """

    # Если боксов нет - сразу выходим
    if len(boxes) == 0:
        return []

    # Приводим тип (ускоряет вычисления)
    boxes = boxes.astype(np.float32)
    scores = scores.astype(np.float32)

    # Сортируем боксы по убыванию score (NMS всегда берет самый высокий)
    order = scores.argsort()[::-1]

    keep = []  # сюда будут записываться индексы оставленных боксов

    while order.size > 0:
        i = order[0]       # индекс бокса с максимальным score
        keep.append(i)

        # Если это был последний бокс - завершаем
        if order.size == 1:
            break

        # Вычисляем IoU текущего бокса с остальными

        # Координаты пересечений
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

        # Ширина и высота пересечения
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        # Площади боксов
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_rest = (
            (boxes[order[1:], 2] - boxes[order[1:], 0]) *
            (boxes[order[1:], 3] - boxes[order[1:], 1])
        )

        # IoU = площадь пересечения / площадь объединения
        union = area_i + area_rest - inter
        iou = inter / (union + 1e-7)

        # Оставляем только те боксы, чьи IoU ниже порога
        inds = np.where(iou <= iou_thres)[0]

        # Компенсируем смещение (order[0] - текущий выбранный бокс)
        order = order[inds + 1]

    return keep


def decode_tflite_output(raw):
    """
    Декодирование сырых предсказаний TFLite YOLO-модели.

    @param raw: np.ndarray
        Исходный выход модели.
        Возможные формы:
            (1, 6, N)
            (1, N, 6)
            (6, N)
            (N, 6)

        Где:
            - первые 4 значения: (x, y, w, h) или (cx, cy, w, h)
            - остальные 2+ значения: логиты классов
              [score_class0, score_class1, ...]
              (модель сама возвращает logits или уже confidence)

    @return np.ndarray shape (N, 6)
        Формат:
            [x, y, w, h, conf, cls_id]
    """

    # Удаляем размерность batch (если была)
    pred = np.squeeze(raw)

    # После squeeze должен быть двумерный массив
    if pred.ndim != 2:
        raise ValueError(f"Ожидал 2D выход, получил pred.shape={pred.shape}")

    # Приведение к (N, 6)
    # Случай: (6, N) -> транспонируем
    if pred.shape[0] == 6 and pred.shape[1] != 6:
        pred = pred.T

    # Случай: уже (N, 6)
    elif pred.shape[1] == 6:
        pass  # пропускаем

    # Если ни один формат не подошёл - ошибка
    else:
        raise ValueError(f"Неожиданная форма выхода: {pred.shape}")

    # Разделение на части
    boxes = pred[:, :4]     # координаты (формат зависит от модели)
    logits = pred[:, 4:]    # классовые логиты (conf per class)

    
    # CLASS CONFIDENCE + CLASS ID
    conf = logits.max(axis=1)            # лучшая уверенность среди классов
    cls_ids = logits.argmax(axis=1)      # ID класса с максимальной уверенностью

    # Формируем выход (N, 6)
    return np.concatenate(
        [boxes, conf[:, None], cls_ids[:, None]],
        axis=1
    )


def yolo_nms(pred, img_w, img_h, conf_thres=0.4, iou_thres=0.5):
    """
    YOLO postprocess + NMS.

    @param pred: np.ndarray shape (N, 6)
        Каждая строка: [x, y, w, h, conf, cls]
        Координаты x,y,w,h - нормализованные (0..1), и заданы в
        формате YOLO (cx, cy, w, h).

    @param img_w: int
        Ширина изображения, в пикселях.

    @param img_h: int
        Высота изображения, в пикселях.

    @param conf_thres: float
        Порог уверенности (confidence threshold).

    @param iou_thres: float
        Порог IoU для NMS.

    @return list:
        Список детекций вида:
        [
            ( [x1, y1, x2, y2], conf, class_id ),
            ...
        ]
        Все координаты - в пикселях, формате XYXY.
    """

    # Приводим к np.array (если был список)
    pred = np.array(pred)

    # Разделяем компоненты
    boxes = pred[:, :4].copy()   # cx cy w h (нормализовано)
    scores = pred[:, 4]          # confidence
    cls = pred[:, 5]             # class id

    # Фильтрация по conf
    mask = scores > conf_thres
    boxes = boxes[mask]
    scores = scores[mask]
    cls = cls[mask]

    if len(boxes) == 0:
        return []

    # Приводим bbox из (cx, cy, w, h) и нормализованных координат в absolute pixels
    boxes[:, 0] *= img_w  # cx
    boxes[:, 1] *= img_h  # cy
    boxes[:, 2] *= img_w  # w
    boxes[:, 3] *= img_h  # h

    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    # Конвертируем в формат XYXY
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    # Non-Maximum Suppression
    keep = nms_xyxy(boxes_xyxy, scores, iou_thres=iou_thres)

    # Формирование итогового списка
    result = []
    for i in keep:
        result.append((
            boxes_xyxy[i],        # координаты бокса
            float(scores[i]),     # score
            int(cls[i])           # class_id
        ))

    return result


def draw_detections(frame, detections, class_names):
    """
    Отрисовка детекций (bbox + метка + score) на изображении.

    @param frame: np.ndarray (BGR)
        Исходный кадр.

    @param detections: list
        Формат:
        [
            ((x1, y1, x2, y2), confidence, class_id),
            ...
        ]

    @param class_names: list[str]
        Список имён классов по индексам.

    @return np.ndarray (BGR)
        Копия изображения с нарисованными боксами.
    """

    img = frame.copy()

    for (x1, y1, x2, y2), sc, cl in detections:

        # Имя класса или "cl" если индекс вне диапазона
        label = class_names[cl] if 0 <= cl < len(class_names) else str(cl)

        # Цвет бокса:
        #  - зелёный для "drone"
        #  - красный для "not_drone"
        color = (0, 255, 0) if label == 'drone' else (0, 0, 255)

        # Рисуем прямоугольник
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Текст: "<label>:<score>"
        cv2.putText(
            img,
            f"{label}:{sc:.2f}",
            (x1, max(0, y1 - 4)),   # смещение над боксом
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,                    # размер шрифта
            color,                  # цвет текста
            1                       # толщина
        )

    return img
