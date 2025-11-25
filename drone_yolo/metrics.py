import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from .config import NEW_TRAIN_DIR

def collect_best_metrics(model_names):
    """
    Сбор лучших метрик (mAP, precision, recall и др.) из результатов обучения YOLO.

    @param model_names: list or iterable
        Список имён моделей, для которых нужно извлечь метрики.

    @return pandas.DataFrame
    """

    rows = []  # Список словарей — одна строка на модель

    # Обрабатываем каждую модель по очереди
    for model in model_names:

        # Путь к CSV с метриками
        csv_path = NEW_TRAIN_DIR / model / "results.csv"

        # Загружаем CSV как DataFrame
        df = pd.read_csv(csv_path)

        # Индекс строки, где mAP50-95(B) максимальный
        best_idx = df["metrics/mAP50-95(B)"].idxmax()

        # Извлекаем лучшую строку
        best = df.iloc[best_idx]

        # Добавляем её в итоговую таблицу
        rows.append({
            "model": model,
            "best_epoch": int(best["epoch"]),            # номер эпохи
            "mAP50": best["metrics/mAP50(B)"],           # mAP@50
            "mAP50-95": best["metrics/mAP50-95(B)"],     # mAP@50-95
            "precision": best["metrics/precision(B)"],   # точность
            "recall": best["metrics/recall(B)"],         # полнота
            "val_box_loss": best["val/box_loss"],        # validation box loss
            "val_cls_loss": best["val/cls_loss"],        # validation cls loss
            "val_dfl_loss": best["val/dfl_loss"],        # validation dfl loss
        })

    # Возвращаем DataFrame с результатами
    return pd.DataFrame(rows)


def plot_map_comparison(results_df):
    """
    Построение графика сравнения mAP50 и mAP50–95 для набора моделей.

    @param results_df: pandas.DataFrame
        Таблица с результатами. 

        Должны присутствовать колонки:
        - model
        - mAP50
        - mAP50-95

    @return None
    """

    # Столбчатая диаграмма для сравнения метрик
    results_df.plot(
        x="model",                         # ось X - названия моделей
        y=["mAP50", "mAP50-95"],           # столбцы для сравнения
        kind="bar",                        # тип графика
        figsize=(8, 4)                     # размер фигуры
    )

    plt.grid(True)                        # сетка на графике
    plt.title("mAP comparison")           # заголовок
    plt.ylabel("mAP")                     # подпись оси Y
    plt.xlabel("Model")                   # подпись оси X

    plt.tight_layout()                    # корректировка расположения элементов
    plt.show()                            # отображение графика
