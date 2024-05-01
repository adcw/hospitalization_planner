from typing import List, Tuple

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# import data.colnames_original as c

def scale(dfs: List[pd.DataFrame]) -> Tuple[List[pd.DataFrame], MinMaxScaler]:
    """
    Scale each DataFrame in the input list to the range (0, 1) using MinMaxScaler.

    Args:
        dfs (List[pd.DataFrame]): List of Pandas DataFrames to be scaled.

    Returns:
        Tuple[List[pd.DataFrame], MinMaxScaler]: A tuple containing the list of scaled Pandas DataFrames
        and the MinMaxScaler object used for scaling.
    """
    # Concatenate DataFrames into one DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)

    # Initialize MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Scale the combined DataFrame
    scaled_data = scaler.fit_transform(combined_df)

    # Split the scaled data back into individual DataFrames
    scaled_dfs = []
    start_index = 0
    for df in dfs:
        num_rows = len(df)
        scaled_df = pd.DataFrame(scaled_data[start_index:start_index + num_rows], columns=df.columns)
        scaled_dfs.append(scaled_df)
        start_index += num_rows

    return scaled_dfs, scaler
