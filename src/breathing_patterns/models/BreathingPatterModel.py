import contextlib
import pickle
from copy import deepcopy
from typing import Optional, List

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder
from torch import nn, optim
from tqdm import tqdm

from src.breathing_patterns.data.dataset import BreathingDataset
from src.model_selection.stratified import train_test_split_safe
from src.nn.archs.lazy_mlc import ConvLayerData as CLD
from src.nn.archs.window_lstm import WindowedConvLSTM
from src.nn.callbacks.early_stopping import EarlyStopping
from src.nn.callbacks.schedules import LrSchedule
from src.session.utils.save_plots import save_plot, save_txt, base_dir
from src.tools.iterators import batch_iter, shuffled


def calculate_class_weights(class_counts, strength: float = 1):
    weights = 1 / np.array(class_counts)
    weights = (weights ** strength) / np.sum(weights ** strength)

    return weights


def one_hot_encode(labels: np.ndarray | List, n_classes: int) -> np.ndarray:
    """
    One-hot encodes the given labels.

    Args:
        labels (np.ndarray): Array of integer labels.
        n_classes (int): Number of classes.

    Returns:
        np.ndarray: One-hot encoded matrix.
    """
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)

    categories = [np.arange(n_classes)]
    encoder = OneHotEncoder(sparse_output=False, categories=categories)
    labels_reshaped = labels.reshape(-1, 1)
    return encoder.fit_transform(labels_reshaped)


class BreathingPatternModel:
    def __init__(self,
                 window_size: int,
                 device: torch.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
                 use_pattern_in_input: bool = False,
                 n_classes: Optional[int] = None
                 ):

        if use_pattern_in_input and n_classes is None:
            raise AttributeError("The use_pattern_in_input is True, but there is no information about n_classes")

        self.window_size = window_size

        self.n_classes = None
        self.n_attr = None
        self.device = device
        self.__optimizer = None
        self.__criterion = None
        self.__net: Optional[WindowedConvLSTM] = None

        self.__use_pattern_in_input = use_pattern_in_input
        self.__n_classes = n_classes

    def __forward(self, xs, ys, batch_size: int, is_eval=False, xs_c: Optional[List] = None):
        if is_eval:
            self.__net.eval()
        else:
            self.__net.train()

        pbar = tqdm(desc="Forwarding", total=len(xs))
        total_loss = 0.0
        total_batches = 0  # New variable to keep track of total batches

        sh_ind = shuffled([i for i in range(len(xs))], p=1)
        xs = [xs[i] for i in sh_ind]
        ys = [ys[i] for i in sh_ind]

        if is_eval:
            batch_size = len(xs)

        with_c = True
        if xs_c is None:
            with_c = False
            xs_c = deepcopy(xs)

        for xs_batch, xs_c_batch, ys_batch in batch_iter(xs, xs_c, ys, batch_size=batch_size):
            xs_batch = torch.stack(xs_batch).to(self.device)
            ys_batch = torch.Tensor(ys_batch).long().to(self.device)

            if with_c:
                xs_c_batch = one_hot_encode(xs_c_batch, self.n_classes)
                xs_c_batch = torch.Tensor(xs_c_batch).long().to(self.device)

            self.__optimizer.zero_grad()

            with (torch.no_grad() if is_eval else contextlib.nullcontext()):
                ys_pred = self.__net.forward(xs_batch, xs_c_batch if with_c else None)

            loss = self.__criterion(ys_pred, ys_batch)
            total_loss += loss.item()
            total_batches += 1  # Increment batch count

            if not is_eval:
                loss.backward()
                self.__optimizer.step()

            pbar.update(len(xs_batch))
            pbar.set_postfix({"Loss": total_loss / total_batches})

        pbar.close()

        avg_loss = total_loss / total_batches  # Calculate average loss
        return avg_loss

    def __setup_net(self):
        cldata = [
            CLD(channels=64, kernel_size=3, activation=nn.SELU),
            CLD(channels=128, kernel_size=3, activation=nn.SELU),
            CLD(channels=256, kernel_size=3, activation=nn.SELU),
            CLD(channels=512, kernel_size=3, activation=nn.SELU),
        ]

        max_conv_depth = int(np.ceil(self.window_size / 2 - 1))

        if max_conv_depth < len(cldata):
            print(
                f"{len(cldata)} conv layers is too much for window size = {self.window_size}. Truncating to {max_conv_depth} layers.")
            cldata = cldata[-max_conv_depth:]
            print(f"{len(cldata)=}")

        # w = 10
        # 1: 10 - 3 + 1 = 8
        # 2: 7 - 3 + 1 = 6
        # 3: 6 - 3 + 1 = 4

        # n: w - k*(3 - 1)
        # w - k (3 - 1) > 0
        # 2k < w
        # k < w / 2

        # k = ceil(w / 2 - 1)

        self.__net = WindowedConvLSTM(n_attr=self.n_attr + (self.__n_classes if self.__use_pattern_in_input else 0),
                                      output_size=self.n_classes,
                                      conv_layers_data=cldata,

                                      # lstm_hidden_size=512,
                                      # lstm_dropout=0.3,
                                      conv_channel_dropout=0.5,
                                      mlp_dropout=0.5,
                                      mlp_arch=[256, 256, 128, 32],

                                      final_activation=None) \
            .to(self.device)

    def dump(self, path: str):
        data = self.n_attr, self.n_classes, self.__net.state_dict()

        with open(path, "wb+") as file:
            pickle.dump(data, file)

    def load(self, path: str):
        with open(path, "rb") as file:
            self.n_attr, self.n_classes, state_dict = pickle.load(file)
            self.__setup_net()
            self.__net.load_state_dict(state_dict)

    def fit(self,
            dataset: BreathingDataset,

            n_epochs: int = 400,
            batch_size: int = 64,
            es_patience: int = 12,
            ):
        self.n_attr = dataset.xs[0].shape[1]
        self.n_classes = len(np.unique(dataset.ys_classes))

        xs_train, xs_c_train, ys_train, xs_val, xs_c_val, ys_val = train_test_split_safe(dataset.xs, dataset.xs_classes,
                                                                                         dataset.ys_classes,
                                                                                         test_size=0.2,
                                                                                         stratify=dataset.ys_classes)

        train_classes_counts = np.unique(ys_train, return_counts=True)[1]
        weight = torch.Tensor(calculate_class_weights(train_classes_counts, strength=1)).to(self.device)

        self.__setup_net()
        self.__criterion = nn.CrossEntropyLoss(weight=weight)
        self.__optimizer = optim.Adam(self.__net.parameters(), weight_decay=0.01, lr=0.0002)

        early_stopping = EarlyStopping(self.__net, patience=es_patience)

        lr_schedule = LrSchedule(optimizer=self.__optimizer, early_stopping=early_stopping, verbose=1)

        print(f"Saving training results to {base_dir()}")

        train_losses = []
        val_losses = []
        for e in range(n_epochs):
            print(f"Epoch {e + 1}/{n_epochs}")

            # Train
            train_loss = self.__forward(xs_train, ys_train, batch_size=batch_size,
                                        xs_c=xs_c_train if self.__use_pattern_in_input else None)
            train_losses.append(train_loss)

            # Eval
            val_loss = self.__forward(xs_val, ys_val, batch_size=batch_size, is_eval=True,
                                      xs_c=xs_c_val if self.__use_pattern_in_input else None)
            val_losses.append(val_loss)

            if early_stopping(val_loss):
                print("Early stopping")
                break

            lr_schedule.step()
        early_stopping.retrieve()

        plt.plot(train_losses, label="Train losses")
        plt.plot(val_losses, label="Val losses")
        plt.title("Wartości straty")
        plt.legend()
        save_plot("losses.png")

        self.__net.eval()
        with torch.no_grad():
            xs_val_tensor = torch.stack(xs_val).to(self.device)
            ys_val_tensor = torch.Tensor(ys_val).long().to(self.device)

            xs_c_val_tensor = None

            if self.__use_pattern_in_input:
                xs_c_val_tensor = one_hot_encode(xs_c_val, self.n_classes)
                xs_c_val_tensor = torch.Tensor(xs_c_val_tensor).long().to(self.device)

            ys_pred = self.__net.forward(xs_val_tensor, xs_c_val_tensor)
            ys_pred_class = torch.argmax(ys_pred, dim=1)

            ys_val_np = ys_val_tensor.cpu().numpy()
            ys_pred_np = ys_pred_class.cpu().numpy()

            save_txt(path='classification_report.txt',
                     txt=classification_report(ys_val_np, ys_pred_np, zero_division=0))

            cm = confusion_matrix(ys_val_np, ys_pred_np)

            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(self.n_classes),
                        yticklabels=range(self.n_classes))
            plt.xlabel('Przewidywane etykiety')
            plt.ylabel('Rzeczywiste etykiety')
            plt.title('Macierz pomyłek - dane walidacyjne')
            save_plot("validation_confusion_matrix.png")

            pass

    def predict(self, xs: List[pd.DataFrame], x_classes: Optional[List[int]] = None):
        if x_classes is not None:
            x_classes = one_hot_encode(x_classes, self.n_classes)
            x_classes = torch.Tensor(x_classes).long().to(self.device)

        tensor_list = []
        for x in xs:
            tensor_list.append(torch.Tensor(x.values))

        x_tens = torch.stack(tensor_list).to(self.device)

        self.__net.eval()
        with torch.no_grad():
            results = self.__net.forward(x_tens, x_classes)
            results = torch.argmax(results, dim=1).cpu().numpy()

        return results

        pass
