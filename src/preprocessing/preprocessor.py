from typing import Optional

import pandas as pd
import torch

from src.preprocessing.utils.transform import TransformData, transform


class Preprocessor:
    def __init__(self,
                 group_cols: list[str],
                 group_sort_col: str,

                 onehot_cols: Optional[list[str]] = None,
                 impute_dict: dict[[str], list[str]] | None = None,
                 rank_dict: dict[[str], list[str]] | None = None,
                 drop_na: bool = False
                 ):
        self.transform_data: Optional[TransformData] = None

        self.group_cols = group_cols
        self.group_sort_col = group_sort_col
        self.onehot_cols = onehot_cols
        self.impute_dict = impute_dict
        self.rank_dict = rank_dict
        self.drop_na = drop_na

        self.original_column_order = None

    def fit_transform(
            self,
            input_df: pd.DataFrame,
    ) -> list[pd.DataFrame]:
        """
        Encode values and group
        :param input_df:
        :return:
        """

        exclude_cols = self.group_cols.copy()
        exclude_cols.append(self.group_sort_col)

        df = input_df[exclude_cols]
        input_df = input_df.drop(columns=exclude_cols)

        self.original_column_order = list(input_df.columns)

        transformed, data = transform(input_df=input_df, onehot_cols=self.onehot_cols,
                                      impute_dict=self.impute_dict, rank_dict=self.rank_dict)

        df = pd.concat([df, transformed], axis=1)

        if self.drop_na:
            df.dropna(inplace=True, axis='rows')

        self.transform_data = data

        groups = df.groupby(self.group_cols, sort=False)

        sequences = []
        for _, g in groups:
            g.sort_values(by=self.group_sort_col, inplace=True)
            sequences.append(g.drop(columns=exclude_cols).astype(float).reset_index(drop=True))

        return sequences

    def inverse_transform(self,
                          tensors: list[torch.Tensor],
                          col_indexes: Optional[list[int]] = None
                          ) -> list[pd.DataFrame]:

        real_cols = self.original_column_order
        if col_indexes is not None:
            real_cols = [real_cols[i] for i in col_indexes]

        if self.transform_data.onehot_encoder is not None:
            original_cols = self.transform_data.onehot_encoder.original_column_order.copy()
            original_cols.extend(self.transform_data.onehot_encoder.encoded_columns)

            drop_cols = self.transform_data.onehot_encoder.columns
            drop_cols.extend(self.group_cols)

            real_cols = [col for col in original_cols if (col not in drop_cols)]

        dfs = []
        for t in tensors:
            df = pd.DataFrame(t, columns=real_cols)

            if self.transform_data.onehot_encoder is not None:
                df = self.transform_data.onehot_encoder.inverse_transform(df)

            if self.transform_data.rank_encoder is not None:
                df = self.transform_data.rank_encoder.inverse_transform(df)

            dfs.append(df)

        return dfs
