from typing import Optional

import numpy as np
import pandas as pd
import torch

from src.preprocessing.transform import TransformData, transform


class Preprocessor:
    def __init__(self,
                 group_cols: list[str],
                 group_sort_col: str,

                 onehot_cols: Optional[list[str]] = None,
                 impute_dict: dict[[str], list[str]] | None = None,
                 ):
        self.transform_data: Optional[TransformData] = None

        self.group_cols = group_cols
        self.group_sort_col = group_sort_col
        self.onehot_cols = onehot_cols
        self.impute_dict = impute_dict

    def fit_transform(
            self,
            input_df: pd.DataFrame,
    ) -> list[np.ndarray]:
        exclude_cols = self.group_cols.copy()
        exclude_cols.append(self.group_sort_col)

        df = input_df[exclude_cols]

        transformed, data = transform(input_df=input_df.drop(columns=exclude_cols).copy(), onehot_cols=self.onehot_cols,
                                      impute_dict=self.impute_dict)

        df = pd.concat([df, transformed], axis=1)
        # df.dropna(inplace=True)

        self.transform_data = data

        groups = df.groupby(self.group_cols, sort=False)

        sequences = []
        for _, g in groups:
            g.sort_values(by=self.group_sort_col, inplace=True)
            sequences.append(g.drop(columns=exclude_cols).values.astype(float))

        return sequences

    def inverse_transform(self, tensors: list[torch.Tensor]) -> list[pd.DataFrame]:
        original_cols = self.transform_data.onehot_encoder.original_column_order.copy()
        original_cols.extend(self.transform_data.onehot_encoder.encoded_columns)

        drop_cols = self.transform_data.onehot_encoder.columns
        drop_cols.extend(self.group_cols)

        real_cols = [col for col in original_cols if (col not in drop_cols)]

        dfs = []
        for t in tensors:
            df = pd.DataFrame(t, columns=real_cols)
            dfs.append(self.transform_data.onehot_encoder.inverse_transform(df))

        return dfs
