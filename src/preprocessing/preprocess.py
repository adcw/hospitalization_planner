from typing import Optional

import numpy as np
import pandas as pd
import torch

from src.preprocessing.transform import TransformData, transform


class Preprocessor:
    def __init__(self):
        self.transform_data: Optional[TransformData] = None
        self.group_cols: Optional[list[str]] = None

    def transform(
            self,
            input_df: pd.DataFrame,
            group_cols: list[str],
            group_sort_col: str,

            onehot_cols: Optional[list[str]] = None,

            impute_dict: dict[[str], list[str]] | None = None,
    ) -> list[torch.Tensor]:
        exclude_cols = group_cols.copy()
        exclude_cols.append(group_sort_col)

        df = input_df[exclude_cols]

        transformed, data = transform(input_df=input_df.drop(columns=exclude_cols).copy(), onehot_cols=onehot_cols,
                                      impute_dict=impute_dict)

        df = pd.concat([df, transformed], axis=1)

        self.transform_data = data
        self.group_cols = group_cols

        groups = df.groupby(self.group_cols, sort=False)

        tensors = []
        for _, g in groups:
            g.sort_values(by=group_sort_col, inplace=True)
            tensors.append(torch.Tensor(g.drop(columns=exclude_cols).values.astype(float)))

        return tensors

    def inverse_transform(self, tensors: list[torch.Tensor]):
        original_cols = self.transform_data.onehot_encoder.original_column_order.copy()
        original_cols.extend(self.transform_data.onehot_encoder.encoded_columns)

        drop_cols = self.transform_data.onehot_encoder.columns
        drop_cols.extend(self.group_cols)

        real_cols = [col for col in original_cols if (col not in drop_cols)]

        dfs = []
        for t in tensors:
            df = pd.DataFrame(t, columns=real_cols)
            dfs.append(self.transform_data.onehot_encoder.inverse_transform(df))

        # tensors = torch.concat(tensors)
        #
        # original_cols = self.transform_data.onehot_encoder.original_column_order.copy()
        # original_cols.extend(self.transform_data.onehot_encoder.encoded_columns)
        #
        # drop_cols = self.transform_data.drop_cols
        # drop_cols.extend(self.transform_data.onehot_encoder.columns)
        # drop_cols.extend(self.group_cols)
        #
        # real_cols = [col for col in original_cols if (col not in drop_cols)]
        # df = pd.DataFrame(tensors, columns=list(real_cols))
        #
        # # onehot decode
        # if self.transform_data.onehot_cols is not None:
        #     df = self.transform_data.onehot_encoder.inverse_transform(df)
        #
        # df.replace(1.0, "YES", inplace=True)
        # df.replace(0.0, "NO", inplace=True)
        # df.replace(np.NAN, "MISSING", inplace=True)

        return dfs
