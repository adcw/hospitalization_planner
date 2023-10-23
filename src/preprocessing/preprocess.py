import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

from data.chosen_colnames import colnames as COLS

import data.colnames as c


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


class OneHotEncoder:
    def __init__(self, columns: list[str]):
        """
        Create encoder class.\n
        **Note**: It can be used to encode and decode any dataframe,
        as long as the column names and encoded labels are same.

        :param columns: List of column names to be encoded
        """
        self.columns = columns
        self.encoded_columns = []
        self.mapping = {}
        self.inverse_mapping = {}
        self.original_column_order = None

    def transform(self, dataframe: pd.DataFrame):
        transformed_df = dataframe.copy()

        self.original_column_order = list(dataframe.columns)

        for column in self.columns:
            encoded = pd.get_dummies(dataframe[column], prefix=column, prefix_sep='__', dtype=float)
            self.encoded_columns.extend(encoded.columns)
            self.mapping[column] = encoded.columns
            transformed_df = pd.concat([transformed_df, encoded], axis=1)
            transformed_df = transformed_df.drop(column, axis=1)

        return transformed_df

    def inverse_transform(self, dataframe: pd.DataFrame):
        original_df = dataframe.copy()

        for column in self.encoded_columns:
            original_column = column.split('__')[0]

            if original_column not in self.inverse_mapping:
                self.inverse_mapping[original_column] = [col for col in dataframe.columns if
                                                         col.startswith(original_column + '__')]

                original_df[original_column] = original_df[self.inverse_mapping[original_column]].idxmax(axis=1).apply(
                    lambda x: x.split("__")[1])

                original_df = original_df.drop(self.inverse_mapping[original_column], axis=1)

        original_df = original_df[self.original_column_order]

        return original_df


if __name__ == '__main__':
    df = pd.read_csv("../../data/input.csv")

    onehot_cols = [c.SEPSIS_CULTURE, c.RDS_TYPE]

    encoder = OneHotEncoder(onehot_cols)

    enc = encoder.transform(df)
    dec = encoder.inverse_transform(enc)

    pass
