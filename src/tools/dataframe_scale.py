from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# import data.colnames_original as c

def scale(data: Union[List[pd.DataFrame], List[np.ndarray]]) -> Tuple[
    Union[List[pd.DataFrame], List[np.ndarray]], MinMaxScaler]:
    """
    Scale each DataFrame or NumPy array in the input list to the range (0, 1) using MinMaxScaler.

    Args:
        data (Union[List[pd.DataFrame], List[np.ndarray]]): List of Pandas DataFrames or NumPy arrays to be scaled.

    Returns:
        Tuple[Union[List[pd.DataFrame], List[np.ndarray]], MinMaxScaler]: A tuple containing the list of scaled Pandas DataFrames
        or NumPy arrays, and the MinMaxScaler object used for scaling.
    """
    if not data:
        return data, MinMaxScaler()

    if isinstance(data[0], pd.DataFrame):
        combined_df = pd.concat(data, ignore_index=True)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(combined_df)

        scaled_data_list = []
        start_index = 0
        for df in data:
            num_rows = len(df)
            scaled_df = pd.DataFrame(scaled_data[start_index:start_index + num_rows], columns=df.columns)
            scaled_data_list.append(scaled_df)
            start_index += num_rows

    elif isinstance(data[0], np.ndarray):
        combined_data = np.vstack(data)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(combined_data)

        scaled_data_list = []
        start_index = 0
        for array in data:
            num_rows = array.shape[0]
            scaled_array = scaled_data[start_index:start_index + num_rows]
            scaled_data_list.append(scaled_array)
            start_index += num_rows

    else:
        raise ValueError("Unsupported data type. The function accepts only lists of DataFrames or NumPy arrays.")

    return scaled_data_list, scaler
