from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from src.preprocessing.encoding import OneHotEncoder


@dataclass
class TransformData:
    df_cols: list[str]
    onehot_cols: Optional[list[str]] = None
    onehot_encoder: Optional[OneHotEncoder] = None


def transform(
        input_df: pd.DataFrame,
        onehot_cols: Optional[list[str]] = None,
        impute_dict: dict[[str], list[str]] | None = None,
) -> tuple[pd.DataFrame, TransformData]:
    """
    :param return_tensor:
    :param drop_cols:
    :param input_df:
    :param onehot_cols:
    :param impute_dict: Dictionary containing keys and values. Key denotes imputation target,
    values are colum names used to impute the target
    :return:
    """
    df_cols = input_df.columns
    onehot_encoder = OneHotEncoder()

    processed_df = input_df.copy()

    # onehot encode
    if onehot_cols is not None:
        processed_df = onehot_encoder.fit_transform(processed_df, onehot_cols)

    processed_df = processed_df.astype(float)

    # TODO: How to chose columns for imputation?
    if impute_dict is not None:
        for impute_target, impute_values in impute_dict.items():
            if impute_target not in impute_values:
                impute_values.append(impute_target)

            sub_df = processed_df.copy()
            sub_df_slice = sub_df[impute_values]
            target_loc = sub_df_slice.columns.get_loc(impute_target)

            # imputer = KNNImputer(n_neighbors=10)
            imputer = SimpleImputer()
            imputed_values = imputer.fit_transform(sub_df_slice)
            sub_df[impute_target] = imputed_values[:, target_loc]

            processed_df = pd.DataFrame(sub_df, columns=processed_df.columns)
        pass

    return processed_df, TransformData(df_cols=df_cols, onehot_cols=onehot_cols,
                                       onehot_encoder=onehot_encoder)
