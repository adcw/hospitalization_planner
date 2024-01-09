import os
from typing import Optional

from matplotlib import pyplot as plt

_base_dir = ""


def base_dir(path: Optional[str] = None):
    """
    Set or get the base plotting dir
    :param path: The path to be set
    :return: Current/updated path
    """
    global _base_dir

    if path is not None:
        _base_dir = path

    if _base_dir is None:
        raise ValueError("Save path is not set")

    return _base_dir


def save_plot(path: str, close: bool = True):
    global _base_dir

    if path is None or path == "":
        path = "."

    save_path = os.path.join(_base_dir, path)
    save_path = os.path.normpath(save_path)

    folder_path = "\\".join(save_path.split("\\")[:-1])

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    plt.savefig(save_path)

    if close:
        plt.close()
