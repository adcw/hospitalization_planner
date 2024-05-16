import contextlib
import pickle
from typing import Optional, List

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
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


def calculate_class_weights(class_counts, strength=1.0):
    weights = 1 / np.array(class_counts)
    weights = (weights ** strength) / np.sum(weights ** strength)

    return weights


class BreathingPatternModel:
    def __init__(self,
                 device: torch.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
        self.n_classes = None
        self.n_attr = None
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

        sh_ind = shuffled([i for i in range(len(xs))], p=0.2)
        xs = [xs[i] for i in sh_ind]
        ys = [ys[i] for i in sh_ind]

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

    def __setup_net(self):
        cldata = [
            CLD(channels=64, kernel_size=3, activation=nn.SELU),
            CLD(channels=128, kernel_size=3, activation=nn.SELU),
            CLD(channels=256, kernel_size=3, activation=nn.SELU),
            CLD(channels=512, kernel_size=3, activation=nn.SELU)
        ]

        self.__net = WindowedConvLSTM(n_attr=self.n_attr,
                                      output_size=self.n_classes,
                                      conv_layers_data=cldata,

                                      lstm_hidden_size=256,
                                      lstm_layers=2,
                                      lstm_dropout=0.2,
                                      mlp_dropout=0.2,
                                      mlp_arch=[128, 64, 32],

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
            es_patience: int = 12
            ):
        self.n_attr = dataset.xs[0].shape[1]
        self.n_classes = len(np.unique(dataset.ys_classes))

        xs_train, ys_train, xs_val, ys_val = train_test_split_safe(dataset.xs, dataset.ys_classes, test_size=0.2,
                                                                   stratify=dataset.ys_classes)

        train_classes_counts = np.unique(ys_train, return_counts=True)[1]
        weight = torch.Tensor(calculate_class_weights(train_classes_counts, strength=1.5)).to(self.device)

        self.__setup_net()
        self.__criterion = nn.CrossEntropyLoss(weight=weight)
        self.__optimizer = optim.Adam(self.__net.parameters(), weight_decay=0.01, lr=0.0001)

        early_stopping = EarlyStopping(self.__net, patience=es_patience)

        lr_schedule = LrSchedule(optimizer=self.__optimizer, early_stopping=early_stopping, verbose=1)

        print(f"Saving training results to {base_dir()}")

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
        save_plot("losses.png")

        self.__net.eval()
        with torch.no_grad():
            xs_val_tensor = torch.stack(xs_val).to(self.device)
            ys_val_tensor = torch.Tensor(ys_val).long().to(self.device)
            ys_pred = self.__net.forward(xs_val_tensor, None)
            ys_pred_class = torch.argmax(ys_pred, dim=1)

            ys_val_np = ys_val_tensor.cpu().numpy()
            ys_pred_np = ys_pred_class.cpu().numpy()

            save_txt(path='metrics', txt=classification_report(ys_val_np, ys_pred_np, zero_division=0))

            cm = confusion_matrix(ys_val_np, ys_pred_np)

            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(self.n_classes),
                        yticklabels=range(self.n_classes))
            plt.xlabel('Predicted labels')
            plt.ylabel('True labels')
            plt.title('Confusion matrix - Validation set')
            save_plot("validation_confusion_matrix.png")

            pass

    def predict(self, xs: List[pd.DataFrame]):

        tensor_list = []
        for x in xs:
            tensor_list.append(torch.Tensor(x.values))

        x_tens = torch.stack(tensor_list).to(self.device)

        self.__net.eval()
        with torch.no_grad():
            results = self.__net.forward(x_tens, None)
            results = torch.argmax(results, dim=1).cpu().numpy()

        return results

        pass
