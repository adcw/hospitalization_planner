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


def _breath_time_features(df: pd.DataFrame, target_col: str, target_col_values_unique: List):
    """
    a. kod sposobu wentylacji w pierwszym elemencie sekwencji (podzielone przez 5),
    b. kod sposobu wentylacji w ostatnim elemencie sekwencji (podzielone przez 5),
    d. liczba OW w sekwencji (normalizacja przez liczbę wezłów w sekwencji)
    e. liczba CPAP w sekwencji (normalizacja przez liczbę wezłów w
    sekwencji)
    f. liczba MAP1 w sekwencji (normalizacja przez liczbę wezłów w sekwencji)
    g. liczba MAP2 w sekwencji (normalizacja przez liczbę wezłów w sekwencji)
    h. liczba MAP3 w sekwencji (normalizacja przez liczbę wezłów w sekwencji)
    i. liczba przejść tzw. pogorszeń sposobów wentylacji  (normalizacja przez liczbę przejść w sekwencji)
    j. liczba przejść tzw. polepszeń sposobów wentylacji  (normalizacja przez liczbę przejść w sekwencji)
    k. czy ostatnie przejście to pogorszenie? (1-TAK, 0-NIE)
    l. czy ostatnie przejście to polepszenie? (1-TAK, 0-NIE)


    :param breath:
    :return:
    """
    data_sequence = df[target_col].values.flatten()
    result = pd.DataFrame()

    result[f'{target_col}_mean'] = [np.mean(data_sequence)]
    result[f'{target_col}_min'] = [np.min(data_sequence)]
    result[f'{target_col}_max'] = [np.max(data_sequence)]

    # result[f'{target_col}_init'] = [data_sequence[0]]
    # result[f'{target_col}_final'] = [data_sequence[-1]]
    #
    # count_dict = {}
    # unique, counts = np.unique(data_sequence, return_counts=True)
    #
    # for u, c in zip(unique, counts):
    #     count_dict[u] = c
    #
    # if len(unique) > 10:
    #     raise ValueError(f"In this column, there are {len(unique)} unique values, which is unexpected")
    #
    # for v in target_col_values_unique:
    #     count = count_dict.get(v, 0)
    #     result[f'{target_col}_val_{v}'] = [count / len(data_sequence)]
    #
    # breath_diff = np.diff(data_sequence)
    # diff_mask = breath_diff != 0
    # all_diffs = diff_mask.sum()
    #
    # if all_diffs == 0:
    #     imps = 0
    #     # dets = 0
    # else:
    #     imps = (breath_diff > 0).sum()
    #     # dets = all_diffs - imps
    #
    # result[f'{target_col}_imps'] = [imps / all_diffs] if all_diffs != 0 else [0]
    # # result[f'{target_col}_dets'] = [dets / all_diffs] if all_diffs != 0 else [0]
    #
    # if all_diffs == 0:
    #     last_change_is_improv = last_change_is_det = 0
    # else:
    #     last_diff = breath_diff[diff_mask][-1]
    #     last_change_is_improv = 1 if last_diff > 0 else 0
    #     last_change_is_det = 1 if last_diff < 0 else 0
    #
    # result[f'{target_col}_is_last_change_imp'] = [last_change_is_improv]
    # result[f'{target_col}_is_last_change_det'] = [last_change_is_det]

    return result


def extract_seq_features(sequences: List[pd.DataFrame],
                         input_cols: Optional[List[str]] = None,
                         mode: Literal['old', 'breath'] = 'breath',
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
    elif mode == 'breath':
        if len(input_cols) > 1:
            raise ValueError("For breath mode, specify only ine input column (respiration)")

        target_col_values_unique = np.unique(
            np.concatenate([s[input_cols[0]].values.flatten() for s in sequences])
        )

        results = [_breath_time_features(seq, input_cols[0], target_col_values_unique) for seq in sequences]

    return pd.concat(results)
