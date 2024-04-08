import random
from typing import List, Tuple
import pandas as pd
import numpy as np
from scipy.stats import linregress
from sklearn.tree import DecisionTreeClassifier, export_text

from data.chosen_colnames import COLS
from src.error_analysis.core.tree_utils import print_top_rules
from src.preprocessing.preprocessor import Preprocessor
import data.colnames_original as c
from src.session.model_manager import _get_sequences


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


if __name__ == '__main__':
    seqs, _ = _get_sequences(path="../../data/input.csv")

    feats = extract_seq_features(seqs, input_cols=[c.RESPIRATION])

    dummy_y = [round(random.random()) for row in feats.itertuples(index=False)]

    # Inicjalizuj klasyfikator drzewiasty
    clf = DecisionTreeClassifier()

    # Dopasuj klasyfikator do danych
    clf = clf.fit(feats, dummy_y)

    print(export_text(clf))

    print_top_rules(tree=clf, dataframe=feats)

    pass
