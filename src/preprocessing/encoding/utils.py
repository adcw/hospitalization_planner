import pandas as pd
import numpy as np

from dataclasses import dataclass
from typing import Optional


@dataclass()
class OH2CData:
    """
    Dataclass containing parameters for onehot2categorical for specific target column.
    :var input_colnames: List of colnames to be used for encoding
    :var threshold: The minimum value of membership for object to be considered as a part of specific class. While setting value higher than 0, please provide also outlier_class value
    :var outlier_class: The name of class for object that haven't reached threshold of any class.
    """
    input_colnames: list[str]
    outlier_class: Optional[str] = None
    threshold: float = 0.

    def __hash__(self):
        return hash((tuple(self.input_colnames), self.outlier_class, self.threshold))


def onehot2categorical(df: pd.DataFrame, onehot_dict: dict[str, OH2CData], drop: bool = False):
    """
    Convert from oe-hot encoding to categorical encoding.
    :param df:
    :param onehot_dict:
    :param drop:
    :return:
    """
    processed_df = df.copy()

    for output_colname, data in onehot_dict.items():
        input_colnames = data.input_colnames.copy()

        if data.outlier_class is not None:
            processed_df[data.outlier_class] = data.threshold
            input_colnames.append(data.outlier_class)

        input_cols = processed_df[input_colnames]

        max_indexes = np.argmax(input_cols, axis=1)
        processed_df[output_colname] = processed_df[input_colnames].columns[max_indexes]

        # Delete temporary columns
        if data.outlier_class is not None:
            # Delete one hot encoded columns
            processed_df.drop(columns=[data.outlier_class], inplace=True)

    if drop:
        for data in onehot_dict.values():
            processed_df.drop(columns=data.input_colnames, inplace=True)

    return processed_df


if __name__ == '__main__':
    """
    Test
    """
    _data = {
        'A': [0, 0, 0, 1, 1],
        'B': [0, 1, 0, 0, 0],
        'C': [1, 0, 1, 0, 0],

        'D': [1, 0.5, 0.6, 1, 0],
        'E': [0, 0.2, 1, 0, 0],
        'F': [0.5, 1, 3, 1, 3],

        'G': [0, 0.1, 0.3, 0.2, 1],
        'H': [0, 0.1, 0.7, 0.2, 0.8],

        'other': [0.5, 1, 3, 1, 3]
    }

    _df = pd.DataFrame(_data)

    _onehot_dict = {
        'First letters': OH2CData(input_colnames=['A', 'B', 'C']),
        'Second letters': OH2CData(input_colnames=['D', 'E', 'F']),
        'Last letters': OH2CData(input_colnames=['G', 'H'], threshold=0.2, outlier_class='Outlier')
    }

    _processed = onehot2categorical(_df, onehot_dict=_onehot_dict, drop=True)

    pass
