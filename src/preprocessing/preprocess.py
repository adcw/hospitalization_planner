from typing import Optional

import numpy as np
import pandas as pd
import torch

from src.preprocessing.transform import TransformData, transform


class Preprocessor:
    def __init__(self):
        self.transform_data: Optional[TransformData] = None

    def transform(
            self,
            input_df: pd.DataFrame,
            group_cols: list[str],

            onehot_cols: Optional[list[str]] = None,
            drop_cols: Optional[list[str]] = None,

            impute_dict: dict[[str], list[str]] | None = None,
    ) -> list[torch.Tensor]:
        df, data = transform(input_df=input_df, onehot_cols=onehot_cols, drop_cols=drop_cols, impute_dict=impute_dict)
        self.transform_data = data

        groups = df.groupby(group_cols)

        tensors = []
        for _, g in groups:
            tensors.append(torch.Tensor(g.drop(columns=group_cols).values))

        return tensors

    def inverse_transform(self, tensors: list[torch.Tensor]):
        tensors = torch.concat(tensors)
        original_cols = set(self.transform_data.onehot_encoder.columns)
        real_cols = original_cols.difference(self.transform_data.drop_cols)
        df = pd.DataFrame(tensors, columns=list(real_cols))

        # onehot decode
        if self.transform_data.onehot_cols is not None:
            df = self.transform_data.onehot_encoder.inverse_transform(df)

        df.replace(1.0, "YES", inplace=True)
        df.replace(0.0, "NO", inplace=True)
        df.replace(np.NAN, "MISSING", inplace=True)

        return df
