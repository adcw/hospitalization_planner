from typing import List, Optional

from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from src.session.utils.save_plots import save_txt, save_plot


def save_report_and_conf_m(
        y_true: List,
        y_pred: List,
        cm_title: Optional[str] = 'Macierz pomyłek',

        report_path: str = "classification_report.txt",
        cm_path: str = "confusion_matrix.png"
):
    n_classes = max([*y_true, *y_pred]) - 1
    save_txt(path=report_path, txt=classification_report(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(n_classes),
                yticklabels=range(n_classes))
    plt.xlabel('Przewidywane etykiety')
    plt.ylabel('Rzeczywiste etykiety')
    plt.title(cm_title)
    save_plot(cm_path)
