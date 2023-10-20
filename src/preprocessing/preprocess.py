import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

from data.chosen_colnames import colnames as COLS


class Preprocessor:
    def __init__(self):
        self.df = None
        self.onehot_cols = None

    def transform(self, input_df: pd.DataFrame, onehot_cols: list[str], impute: bool = True):
        self.df = input_df
        self.onehot_cols = onehot_cols

        processed_df = self.df.copy()
        processed_df = processed_df[COLS]

        # onehot encode
        for col in self.onehot_cols:
            unique_vals = np.unique(processed_df[col].values)

            for val in unique_vals:
                colname = f"{col}_{val}"

                value_mask = processed_df[col] == val

                processed_df[colname] = 0.
                processed_df.loc[value_mask, colname] = 1.

        processed_df.drop(columns=self.onehot_cols, inplace=True)

        # replace literals with values
        processed_df.replace("YES", 1., inplace=True)
        processed_df.replace("NO", 0., inplace=True)
        processed_df.replace("MISSING", np.NAN, inplace=True)

        if impute:
            # perform KNN imputing.
            # TODO: Verify if this makes sense.

            imputer = KNNImputer(n_neighbors=5)
            imputed_vals = imputer.fit_transform(processed_df)
            processed_df = pd.DataFrame(imputed_vals, columns=processed_df.columns)

        return processed_df

    def inverse_transform(self, processed_df):
        # TODO: Verify if it works
        for col in self.onehot_cols:
            onehot_columns = [colname for colname in processed_df.columns if colname.startswith(f"{col}_")]

            col_vals = []
            for colname in onehot_columns:
                val = colname.split("_")[-1]
                col_vals.append((colname, val))

            for colname, val in col_vals:
                processed_df[val] = processed_df[colname]
                processed_df.drop(columns=[colname], inplace=True)

        # processed_df.rename(columns=c.REVERSE_COLUMN_NAMES, inplace=True)

        # processed_df.replace(1.0, "YES", inplace=True)
        # processed_df.replace(0.0, "NO", inplace=True)
        processed_df.replace(np.NAN, "MISSING", inplace=True)

        return processed_df


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
