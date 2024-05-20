from typing import List, Optional, Literal

import numpy as np
import pandas as pd
from scipy.stats import linregress


def _df_time_features(dataframe: pd.DataFrame, input_cols: List[str]) -> pd.DataFrame:
    """
    Analyzes time series features in the dataframe and returns a new DataFrame.

    Parameters:
        dataframe (pd.DataFrame): Input dataframe containing time series features.
        input_cols (List[str]): List of column names to use for analysis.

    Returns:
        pd.DataFrame: New dataframe containing standard deviation, slope, and mean
        for each selected column.
    """

    new_dataframe = pd.DataFrame()

    if input_cols is None:
        input_cols = dataframe.columns.tolist()

    for col in input_cols:
        # Standard deviation
        # std_dev_col = dataframe[col].std()
        # new_dataframe[col + '_std_dev'] = [std_dev_col]

        # Slope
        x = np.arange(len(dataframe))
        y = dataframe[col].values

        A = np.vstack([x, np.ones(len(x))]).T

        (slope, intercept), _, _, _ = np.linalg.lstsq(A, y, rcond=None)

        # slope, intercept, _, slope_err, _ = linregress(x, y)

        new_dataframe[col + '_slope'] = [slope]
        # new_dataframe[col + '_initial'] = [y[0]]
        # new_dataframe[col + '_final'] = [y[-1]]
        new_dataframe[col + '_std'] = [np.std(y)]

        # Mean
        mean_vals = dataframe[col].mean()
        new_dataframe[col + '_mean'] = [mean_vals]

    # new_dataframe['len'] = len(dataframe)

    return new_dataframe


def _3m_features(df: pd.DataFrame, target_col: str):
    data_sequence = df[target_col].values.flatten()
    result = pd.DataFrame()

    result[f'{target_col}_mean'] = [np.mean(data_sequence)]
    result[f'{target_col}_min'] = [np.min(data_sequence)]
    result[f'{target_col}_max'] = [np.max(data_sequence)]
    return result


def _proportion_features(
        df: pd.DataFrame, target_col: str, target_col_values_unique: np.ndarray
):
    data_sequence = df[target_col].values.flatten()
    result = pd.DataFrame()
    count_dict = {}
    unique, counts = np.unique(data_sequence, return_counts=True)

    for u, c in zip(unique, counts):
        count_dict[u] = c

    if len(unique) > 10:
        raise ValueError(f"In this column, there are {len(unique)} unique values, which is unexpected")

    for v in target_col_values_unique:
        count = count_dict.get(v, 0)
        result[f'{target_col}_val_{v}'] = [count / len(data_sequence)]

    return result


def extract_seq_features(sequences: List[pd.DataFrame],
                         input_cols: Optional[List[str]] = None,
                         mode: Literal['3m', 'prop', 'old'] = '3m',
                         ) -> pd.DataFrame:
    """
    Extracts time series features from a list of sequences.

    Parameters:
        sequences (List[pd.DataFrame]): List of DataFrames representing time series sequences.
        input_cols (List[str]): List of column names to use for analysis.

    Returns:
        pd.DataFrame: DataFrame containing extracted features from all sequences.
        :param mode:
    """
    results = None

    if mode == 'old':
        results = [_df_time_features(seq, input_cols) for seq in sequences]
    elif mode == '3m':
        results = [_3m_features(seq, input_cols[0]) for seq in sequences]

    elif mode == 'prop':
        if len(input_cols) > 1:
            raise ValueError("For breath mode, specify only ine input column (respiration)")

        target_col_values_unique = np.unique(
            np.concatenate([s[input_cols[0]].values.flatten() for s in sequences])
        )

        results = [_proportion_features(seq, input_cols[0], target_col_values_unique) for seq in sequences]

    return pd.concat(results)
