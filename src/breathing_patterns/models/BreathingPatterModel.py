import contextlib
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.optim import Adam
from tqdm import tqdm

from src.breathing_patterns.data.dataset import BreathingDataset
from src.nn.archs.lazy_mlc import ConvLayerData as CLD
from src.nn.archs.window_lstm import WindowedConvLSTM
from src.nn.callbacks.early_stopping import EarlyStopping
from src.nn.callbacks.schedules import LrSchedule
from src.tools.iterators import batch_iter

from matplotlib import pyplot as plt

import seaborn as sns


class BreathingPatternModel:
    def __init__(self,
                 device: torch.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
        self.device = device
        self.__optimizer = None
        self.__criterion = None
        self.__net: Optional[WindowedConvLSTM] = None

    def __forward(self, xs, ys, batch_size: int, is_eval=False):

        if is_eval:
            self.__net.eval()
        else:
            self.__net.train()

        pbar = tqdm(desc="Forwarding", total=len(xs))
        total_loss = 0.0
        for xs_batch, ys_batch in batch_iter(xs, ys, batch_size=batch_size):
            xs_batch = torch.stack(xs_batch).to(self.device)
            ys_batch = torch.Tensor(ys_batch).long().to(self.device)

            self.__optimizer.zero_grad()

            with (torch.no_grad() if is_eval else contextlib.nullcontext()):
                ys_pred = self.__net.forward(xs_batch, None)

            loss = self.__criterion(ys_pred, ys_batch)
            total_loss += loss.item()

            if not is_eval:
                loss.backward()
                self.__optimizer.step()

            pbar.update(len(xs_batch))
            pbar.set_postfix({"Loss": total_loss / pbar.n})

        pbar.close()

        avg_loss = total_loss / len(xs)
        return avg_loss

    def fit(self,
            dataset: BreathingDataset,

            n_epochs: int = 400,
            batch_size: int = 64,
            es_patience: int = 12
            ):
        n_attr = dataset.xs[0].shape[1]
        n_classes = len(np.unique(dataset.ys))

        cldata = [
            CLD(channels=16, kernel_size=3, activation=nn.SELU),
            CLD(channels=32, kernel_size=3, activation=nn.SELU),
            CLD(channels=64, kernel_size=3, activation=nn.SELU),
        ]

        self.__net = WindowedConvLSTM(n_attr=n_attr,
                                      output_size=n_classes,
                                      conv_layers_data=cldata,
                                      final_activation=None) \
            .to(self.device)

        self.__criterion = nn.CrossEntropyLoss()
        self.__optimizer = optim.Adam(self.__net.parameters(), weight_decay=0.001, lr=0.001)

        early_stopping = EarlyStopping(self.__net, patience=es_patience)

        lr_schedule = LrSchedule(optimizer=self.__optimizer, early_stopping=early_stopping, verbose=2)

        xs_train, xs_val, ys_train, ys_val = train_test_split(dataset.xs, dataset.ys, test_size=0.1,
                                                              stratify=dataset.ys)

        train_losses = []
        val_losses = []
        for e in range(n_epochs):
            print(f"Epoch {e + 1}/{n_epochs}")

            # Train
            train_loss = self.__forward(xs_train, ys_train, batch_size)
            train_losses.append(train_loss)

            # Eval
            val_loss = self.__forward(xs_val, ys_val, batch_size, is_eval=True)
            val_losses.append(val_loss)

            if early_stopping(val_loss):
                print("Early stopping")
                break

            lr_schedule.step()
        early_stopping.retrieve()

        plt.plot(train_losses, label="Train losses")
        plt.plot(val_losses, label="Val losses")
        plt.title("Training performance")
        plt.show()

        self.__net.eval()
        with torch.no_grad():
            xs_val_tensor = torch.stack(xs_val).to(self.device)
            ys_val_tensor = torch.Tensor(ys_val).long().to(self.device)
            ys_pred = self.__net.forward(xs_val_tensor, None)
            ys_pred_class = torch.argmax(ys_pred, dim=1)

            ys_val_np = ys_val_tensor.cpu().numpy()
            ys_pred_np = ys_pred_class.cpu().numpy()

            print(classification_report(ys_val_np, ys_pred_np))
            cm = confusion_matrix(ys_val_np, ys_pred_np)

            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(n_classes),
                        yticklabels=range(n_classes))
            plt.xlabel('Predicted labels')
            plt.ylabel('True labels')
            plt.title('Confusion matrix - Validation set')
            plt.show()

        pass
