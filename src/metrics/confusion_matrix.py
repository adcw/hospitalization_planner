from typing import Literal, LiteralString

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from src.config.seeds import set_seed
import numpy as np


def cm2precision(conf_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate precision for each class from the confusion matrix.

    Args:
        conf_matrix (np.ndarray): Confusion matrix.

    Returns:
        np.ndarray: Array of precision values for each class.
    """
    true_positives = np.diag(conf_matrix)
    predicted_positives = np.sum(conf_matrix, axis=0)
    precision = true_positives / predicted_positives
    precision[np.isnan(precision)] = 0  # Handle division by zero
    return precision


def cm2recall(conf_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate recall for each class from the confusion matrix.

    Args:
        conf_matrix (np.ndarray): Confusion matrix.

    Returns:
        np.ndarray: Array of recall values for each class.
    """
    true_positives = np.diag(conf_matrix)
    actual_positives = np.sum(conf_matrix, axis=1)
    recall = true_positives / actual_positives
    recall[np.isnan(recall)] = 0  # Handle division by zero
    return recall


def show_cm(cm: np.ndarray, n_classes: int, is_float: bool = False, title: str = "Macierz pomyłek",
            font_scale: float = 1.5):
    plt.figure(figsize=(7, 6))
    # sns.set(font_scale=font_scale)
    sns.heatmap(cm, annot=True, fmt=".2f" if is_float else "d", cmap="Blues",
                xticklabels=range(n_classes), yticklabels=range(n_classes),
                annot_kws={"size": 14})  # adjust the size of annotations if needed
    plt.xlabel('Przewidywane etykiety', fontsize=16)
    plt.ylabel('Rzeczywiste etykiety', fontsize=16)
    plt.title(title, fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()


def normalize_confusion_matrix(conf_matrix: np.ndarray, normalize: str = Literal['all', 'true', 'pred']) -> np.ndarray:
    """
    Normalize a confusion matrix.

    Args:
        conf_matrix (np.ndarray): Confusion matrix.
        normalize (str, optional): Normalization mode. Options: 'true', 'pred', 'all'.
                                   Default is 'true'.

    Returns:
        np.ndarray: Normalized confusion matrix.
    """
    if normalize == 'true':
        row_sums = conf_matrix.sum(axis=1, keepdims=True)
        normalized_conf_matrix = conf_matrix / row_sums
    elif normalize == 'pred':
        col_sums = conf_matrix.sum(axis=0, keepdims=True)
        normalized_conf_matrix = conf_matrix / col_sums
    elif normalize == 'all':
        total_sum = conf_matrix.sum()
        normalized_conf_matrix = conf_matrix / total_sum
    else:
        raise ValueError("Invalid normalization mode. Options: 'true', 'pred', 'all'.")

    return normalized_conf_matrix


def generate_random_labels_with_accuracy(num_samples: int, num_classes: int, accuracy: float) -> (
        np.ndarray, np.ndarray):
    """
    Generate random true and predicted labels with a specified accuracy for confusion matrix testing.

    Args:
        num_samples (int): Number of samples to generate.
        num_classes (int): Number of different classes.
        accuracy (float): Desired accuracy percentage (0 to 1).

    Returns:
        np.ndarray: Array of true labels.
        np.ndarray: Array of predicted labels.
    """
    true_labels = np.random.randint(0, num_classes, size=num_samples)
    predicted_labels = np.copy(true_labels)

    num_incorrect = int((1 - accuracy) * num_samples)
    incorrect_indices = np.random.choice(num_samples, num_incorrect, replace=False)

    for index in incorrect_indices:
        predicted_labels[index] = np.random.randint(0, num_classes)
        while predicted_labels[index] == true_labels[index]:
            predicted_labels[index] = np.random.randint(0, num_classes)

    return true_labels, predicted_labels


def weighted_confusion_matrix(cm: np.array, weight_matrix: np.ndarray, normalize: str = 'all'):
    cm_with_weights = np.multiply(cm, weight_matrix)
    return normalize_confusion_matrix(cm_with_weights, normalize)


def pr2f1(precision: np.ndarray, recall: np.ndarray) -> np.ndarray:
    f1 = 2 * (precision * recall) / (precision + recall)
    f1[np.isnan(f1)] = 0  # Handle division by zero
    return f1


def macro_average(metric: np.ndarray) -> float:
    return np.mean(metric)


def weighted_average(metric: np.ndarray, support: np.ndarray) -> float:
    return np.average(metric, weights=support)


def calculate_accuracy(conf_matrix: np.ndarray) -> float:
    return np.trace(conf_matrix) / np.sum(conf_matrix)


def testing():
    set_seed()
    n_classes = 4
    _labels_true, _labels_pred = generate_random_labels_with_accuracy(500, n_classes, 0.7)

    original_cm = confusion_matrix(_labels_true, _labels_pred)
    show_cm(original_cm, n_classes, title="Oryginalna macierz pomyłek")

    original_cm_norm = confusion_matrix(_labels_true, _labels_pred, normalize='all')
    show_cm(original_cm_norm, n_classes, title="Oryginalna znormalizowana macierz pomyłek", is_float=True)

    report = classification_report(_labels_true, _labels_pred, output_dict=True)

    precision = cm2precision(original_cm)
    recall = cm2recall(original_cm)

    precision_report = np.array([report[str(i)]['precision'] for i in range(n_classes)])
    recall_report = np.array([report[str(i)]['recall'] for i in range(n_classes)])

    assert np.allclose(precision, precision_report), "Precision values do not match!"
    assert np.allclose(recall, recall_report), "Recall values do not match!"


if __name__ == '__main__':
    original_conf_matrix = np.array([
        [247, 207, 12, 122],
        [244, 1401, 73, 35],
        [3, 69, 308, 5],
        [150, 65, 7, 3950],
    ])

    n_classes = 4

    # weight_matrix = np.array([
    #     [1, 0.9, 1, 0.9],
    #     [0.9, 1, 0.9, 1],
    #     [0.9, 0.8, 1, 1],
    #     [0.8, 0.9, 1, 1]
    # ])

    # weight_matrix = np.array([
    #     [3, 1, 2, 1],
    #     [1, 3, 1, 2],
    #     [2, 1, 3, 3],
    #     [1, 2, 3, 3]
    # ])

    weight_matrix = np.array([
        [4/3, 1, 2, 1],
        [1, 4/3, 1, 2],
        [2, 1, 2, 3],
        [1, 2, 3, 2]
    ])

    show_cm(original_conf_matrix, n_classes, title="Macierz pomyłek")
    #
    # normalized_original_cm = normalize_confusion_matrix(original_conf_matrix, normalize='all')
    # show_cm(normalized_original_cm, 4, title="Normalizowana macierz pomyłek", is_float=True)

    all_weighted_cm = weighted_confusion_matrix(original_conf_matrix, weight_matrix=weight_matrix, normalize='all')

    true_weighted_cm = weighted_confusion_matrix(original_conf_matrix, weight_matrix=weight_matrix, normalize='true')
    show_cm(true_weighted_cm, n_classes, title="Ważona macierz pomyłek normalizowana\n względem klas rzeczywistych",
            is_float=True)

    pred_weighted_cm = weighted_confusion_matrix(original_conf_matrix, weight_matrix=weight_matrix, normalize='pred')
    show_cm(pred_weighted_cm, n_classes, title="Ważona macierz pomyłek normalizowana\n względem klas przewidywanych",
            is_float=True)

    precision = cm2precision(all_weighted_cm)
    recall = cm2recall(all_weighted_cm)
    accuracy = calculate_accuracy(all_weighted_cm)

    f1 = pr2f1(precision, recall)

    support = np.sum(original_conf_matrix, axis=1)

    macro_precision = macro_average(precision)
    macro_recall = macro_average(recall)
    macro_f1 = macro_average(f1)

    weighted_precision = weighted_average(precision, support)
    weighted_recall = weighted_average(recall, support)
    weighted_f1 = weighted_average(f1, support)

    print("Precyzja z ważonej macierzy pomyłek:")
    for i, p in enumerate(precision):
        print(f"Klasa {i}: {p:.2f}")

    print("Recall z ważonej macierzy pomyłek:")
    for i, r in enumerate(recall):
        print(f"Klasa {i}: {r:.2f}")

    print("F1-Score z ważonej macierzy pomyłek:")
    for i, f in enumerate(f1):
        print(f"Klasa {i}: {f:.2f}")

    print(f"Support z ważonej macierzy pomyłek:")
    for i in range(n_classes):
        print(f"Klasa {i}: {int(support[i])}")

    print(f"Dokładność: {accuracy:.2f}")
    print(f"Średnia makro precyzja: {macro_precision:.2f}")
    print(f"Średnia makro recall: {macro_recall:.2f}")
    print(f"Średnia makro F1-score: {macro_f1:.2f}")
    print(f"Średnia ważona precyzja: {weighted_precision:.2f}")
    print(f"Średnia ważona recall: {weighted_recall:.2f}")
    print(f"Średnia ważona F1-score: {weighted_f1:.2f}")
