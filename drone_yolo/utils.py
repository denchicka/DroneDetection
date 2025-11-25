import gc
import torch
import cv2
import matplotlib.pyplot as plt
import glob
from pathlib import Path
from IPython.display import Video, display, HTML
from moviepy import VideoFileClip

def clean_memory():
    """
    Освобождение RAM и GPU памяти.

    Выполняет:
      - сборку мусора Python (gc.collect)
      - очистку CUDA кеша (torch.cuda.empty_cache)
      - сборку CUDA IPC объектов (torch.cuda.ipc_collect)

    @return None
    """

    # Чистим Python-объекты из памяти
    gc.collect()

    # Если доступна CUDA - чистим GPU память
    if torch.cuda.is_available():
        torch.cuda.empty_cache()   # освобождает кешированные тензоры
        torch.cuda.ipc_collect()   # собирает IPC память между процессами


def show_image(img, title=None, size=(6, 6)):
    """
    Отображение изображения в matplotlib.

    @param img: np.ndarray (BGR)
        Изображение в формате OpenCV (BGR).  
        Автоматически будет конвертировано в RGB для корректного отображения в matplotlib.

    @param title: str or None
        Заголовок для изображения.  
        Если None - заголовок не выводится.

    @param size: tuple(int, int)
        Размер фигуры matplotlib (width, height).  
        По умолчанию (6, 6).

    @return None
    """

    plt.figure(figsize=size)

    # OpenCV -> matplotlib (BGR -> RGB)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if title:
        plt.title(title)

    plt.axis("off")  # скрываем оси

    plt.show()

def show_saved_train_images(run_dir):
    """
    Отображение сохранённых изображений и графиков обучения для эксперимента YOLO.

    @param run_dir: str or Path
        Путь к директории одного эксперимента YOLO.

    @return None
    """

    run_dir = Path(run_dir)
    
    # Обучающие батчи: train_batch*.jpg
    for p in sorted(run_dir.glob("train_batch*.jpg")):
        img = cv2.imread(str(p))
        if img is not None:
            show_image(
                img,
                title=f"{run_dir.name} — {p.name}",
                size=(8, 8)
            )

    # Confusion Matrix
    cm = run_dir / "confusion_matrix_normalized.png"
    if cm.exists():
        img = cv2.imread(str(cm))
        if img is not None:
            show_image(
                img,
                title=f"{run_dir.name} - Confusion Matrix",
                size=(8, 8)
            )

    # Кривые обучения ("*curve*.png")
    for p in sorted(run_dir.glob("*curve*.png")):
        img = cv2.imread(str(p))
        if img is not None:
            show_image(
                img,
                title=f"{run_dir.name} - {p.name}",
                size=(8, 8)
            )
            

def convert_avi_to_mp4_ffmpeg(src_path, dst_path):
    src_path, dst_path = str(src_path), str(dst_path)
    try:
        clip = VideoFileClip(src_path)
        clip.write_videofile(
            dst_path,
            codec="libx264",
            audio=False,
            fps=clip.fps or 25
        )
        clip.close()
        print(f"[FFMPEG] Готово: {dst_path}")
        return True
    except Exception as e:
        print(f"[FFMPEG] Ошибка: {e}")
        return False



def convert_avi_to_mp4_for_model(model_name, max_videos_to_display=2):
    """
    Конвертация всех AVI-файлов, сохранённых YOLO-инференсом,
    в MP4 и отображение нескольких примеров прямо в ноутбуке.

    @param model_name: str
        Имя модели (папка внутри runs/).

    @param max_videos_to_display: int
        Сколько первых сконвертированных видео показывать через HTML <video>.
        По умолчанию 2.

    @return None
    """

    base_dir = Path(f"./runs/{model_name}")
    if not base_dir.exists():
        print(f"Папка {base_dir} не найдена.")
        return

    avi_files = sorted(base_dir.glob("**/*.avi"))
    if not avi_files:
        print(f"Не найдено AVI для {model_name}")
        return

    print(f"{model_name}: найдено {len(avi_files)} .avi файлов")

    shown = 0
    for avi_path in avi_files:
        out_dir = avi_path.parent / "mp4"
        out_dir.mkdir(exist_ok=True)

        mp4_path = out_dir / (avi_path.stem + ".mp4")

        if not mp4_path.exists():
            print(f"Конвертация: {avi_path} -> {mp4_path}")
            convert_avi_to_mp4_ffmpeg(avi_path, mp4_path)
        else:
            print(f"MP4 уже существует: {mp4_path}")

        if shown < max_videos_to_display:
            display(HTML(f"""
            <h3>{model_name}: {avi_path.relative_to(base_dir)}</h3>
            <video width="640" controls>
                <source src="{mp4_path.as_posix()}" type="video/mp4">
            </video>
            """))
            shown += 1

            
def show_tflite_videos(max_videos=10):
    print("\nПоиск TFLite-бенчмарков...")

    files = sorted(glob.glob("runs/detect/tflite_custom/*.mp4"))
    if not files:
        print("Нет файлов *.mp4 в runs/detect/tflite_custom")
        return

    shown = 0

    for file in files:
        print(f"\nНайден файл: {file}")
        file = Path(file)

        # Если вдруг это не H.264, можно перегнать ещё раз в "fixed"
        fixed_path = file.with_name(file.stem + "_fixed.mp4")

        if not fixed_path.exists():
            print(f"Конвертация {file} -> {fixed_path}")
            convert_avi_to_mp4_ffmpeg(str(file), str(fixed_path))
        else:
            print(f"Файл уже перекодирован: {fixed_path}")

        display(Video(str(fixed_path), embed=True))

        shown += 1
        if shown >= max_videos:
            break

    print("\nГотово!")
            
def convert_video_opencv(src_path: str, dst_path: str, codec="mp4v", fps=None):
    """
    Примитивный конвертер видео через OpenCV.
    src_path: входной .avi / .mp4 и т.п.
    dst_path: выходной .mp4 (желательно)
    codec: 'mp4v' (обычно норм для Jupyter/Colab)
    fps: если None - берём FPS исходного видео
    """
    src_path = str(src_path)
    dst_path = str(dst_path)

    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        print(f"[convert_video_opencv] Не удалось открыть {src_path}")
        return False

    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    if fps is None:
        fps = src_fps if src_fps > 0 else 25.0

    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(dst_path, fourcc, fps, (w, h))

    if not out.isOpened():
        print(f"[convert_video_opencv] Не удалось открыть writer для {dst_path}")
        cap.release()
        return False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    print(f"[convert_video_opencv] Готово: {dst_path}")
    return True

def show_infer_videos(models, max_videos=3):
    """
    Показывает результаты инференса YOLO:
    - ищет .mp4 в runs/<model_name>/video_*/mp4/
    - показывает в Jupyter
    """

    for model_name in models.keys():
        print(f"\nВидео для модели: {model_name}")

        # Путь под вашу структуру:
        pattern = f"runs/{model_name}/video_*/mp4/*.mp4"

        mp4_files = sorted(glob.glob(pattern))
        if not mp4_files:
            print("Нет mp4 файлов по пути:", pattern)
            continue

        print(f"Найдено MP4: {len(mp4_files)}")

        shown = 0
        for mp4_path in mp4_files:
            if shown >= max_videos:
                break

            print("Показываю:", mp4_path)

            display(HTML(f"""
                <video width="640" controls>
                    <source src="{mp4_path}" type="video/mp4">
                </video>
            """))
            shown += 1

        print("-" * 50)
