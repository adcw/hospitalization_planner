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


def show_cm(cm: np.ndarray, is_float: bool = False):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f" if is_float else "d", cmap="Blues", xticklabels=range(n_classes),
                yticklabels=range(n_classes))
    plt.xlabel('Przewidywane etykiety')
    plt.ylabel('Rzeczywiste etykiety')
    plt.title("Macierz pomyłek")
    plt.show()


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


if __name__ == '__main__':
    set_seed()
    n_classes = 3
    labels_true, labels_pred = generate_random_labels_with_accuracy(500, n_classes, 0.7)

    _cm = confusion_matrix(labels_true, labels_pred)
    show_cm(_cm)
    report = classification_report(labels_true, labels_pred, output_dict=True)

    precision = cm2precision(_cm)
    recall = cm2recall(_cm)

    precision_report = np.array([report[str(i)]['precision'] for i in range(n_classes)])
    recall_report = np.array([report[str(i)]['recall'] for i in range(n_classes)])

    assert np.allclose(precision, precision_report), "Precision values do not match!"
    assert np.allclose(recall, recall_report), "Recall values do not match!"
