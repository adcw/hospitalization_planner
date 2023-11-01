import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import data.raw.colnames as c
from src.preprocessing.encoding.onehotencoder import OneHotEncoder


def normalize_split(train_seq: list[np.ndarray], val_seq: list[np.ndarray] | None):
    """
    Normalize train and validation data.
    Scaler is fit on training data and used both on training and validation data.
    :param train_seq: Train sequence
    :param val_seq: Validation sequence
    :return: Normalized train, normalized validation data, scaler.
    """
    scaler = MinMaxScaler()

    train_cat = np.concatenate(train_seq)
    train_cat_norm = scaler.fit_transform(train_cat)
    train_seq_norm = np.split(train_cat_norm, np.cumsum([seq.shape[0] for seq in train_seq])[:-1])

    if val_seq is not None:
        val_cat = np.concatenate(val_seq)
        val_cat_norm = scaler.transform(val_cat)
        val_seq_norm = np.split(val_cat_norm, np.cumsum([seq.shape[0] for seq in val_seq])[:-1])
    else:
        val_seq_norm = None

    return train_seq_norm, val_seq_norm, scaler


if __name__ == '__main__':
    df = pd.read_csv("../../data/clean/input.csv")

    onehot_cols = [c.SEPSIS_CULTURE, c.RDS_TYPE]

    encoder = OneHotEncoder(onehot_cols)

    enc = encoder.fit_transform(df)
    dec = encoder.inverse_transform(enc)

    pass
