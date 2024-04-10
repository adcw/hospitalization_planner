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


def _prepare_path(path: str):
    global _base_dir

    if path is None or path == "":
        path = "."

    save_path = os.path.join(_base_dir, path)
    save_path = os.path.normpath(save_path)

    folder_path = "\\".join(save_path.split("\\")[:-1])

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    return save_path


def save_plot(path: str, close: bool = True):
    save_path = _prepare_path(path)
    plt.savefig(save_path)

    if close:
        plt.close()


def save_txt(path: str, txt: str):
    save_path = _prepare_path(path)
    with open(save_path, "w+") as file:
        file.write(txt)


def save_viz(path: str, viz):
    save_path = _prepare_path(path)
    viz.save(save_path)
