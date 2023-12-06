from typing import Optional, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.preprocessing.encoding.onehotencoder import OneHotEncoder


def split_and_norm_sequences(sequences: list[np.ndarray], val_perc: float = 0.2):
    """
    Normalize train and validation data.
    Scaler is fit on training data and used both on training and validation data.
    :param val_perc:
    :param sequences:
    :return: Normalized train, normalized validation data, scaler.
    """
    scaler = MinMaxScaler()

    split_point = round(len(sequences) * (1 - val_perc))
    train_seq = sequences[:split_point]
    val_seq = sequences[split_point:]

    train_cat = np.concatenate(train_seq)
    train_cat_norm = scaler.fit_transform(train_cat)
    train_seq_norm = np.split(train_cat_norm, np.cumsum([seq.shape[0] for seq in train_seq])[:-1])

    if len(val_seq) > 0:
        val_seq_norm = transform_sequences(val_seq, scaler)
    else:
        val_seq_norm = []

    return train_seq_norm, val_seq_norm, scaler


def transform_sequences(val_seq: Optional[List[np.ndarray]], scaler: MinMaxScaler):
    val_cat = np.concatenate(val_seq)
    val_cat_norm = scaler.transform(val_cat)
    val_seq_norm = np.split(val_cat_norm, np.cumsum([seq.shape[0] for seq in val_seq])[:-1])

    return val_seq_norm


if __name__ == '__main__':
    df = pd.read_csv("../../../data/input.csv")

    onehot_cols = [c.POSIEW_SEPSA, c.TYPE_RDS]

    encoder = OneHotEncoder(onehot_cols)

    enc = encoder.fit_transform(df)
    dec = encoder.inverse_transform(enc)

    pass
