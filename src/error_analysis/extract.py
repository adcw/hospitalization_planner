from typing import List

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
        std_dev_col = dataframe[col].std()
        new_dataframe[col + '_std_dev'] = [std_dev_col]

        # Slope
        slope_vals = []
        x = np.arange(len(dataframe))
        y = dataframe[col].values
        slope, _, _, _, _ = linregress(x, y)
        slope_vals.append(slope)
        new_dataframe[col + '_slope'] = slope_vals

        # Mean
        mean_vals = dataframe[col].mean()
        new_dataframe[col + '_mean'] = [mean_vals]

    new_dataframe['len'] = len(dataframe)

    return new_dataframe


def extract_seq_features(sequences: List[pd.DataFrame], input_cols: List[str]) -> pd.DataFrame:
    """
    Extracts time series features from a list of sequences.

    Parameters:
        sequences (List[pd.DataFrame]): List of DataFrames representing time series sequences.
        input_cols (List[str]): List of column names to use for analysis.

    Returns:
        pd.DataFrame: DataFrame containing extracted features from all sequences.
    """

    results = [_df_time_features(seq, input_cols) for seq in sequences]

    return pd.concat(results)
