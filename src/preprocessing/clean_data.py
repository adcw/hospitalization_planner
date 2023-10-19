import os

import numpy as np
import pandas as pd

from src.preprocessing.sequences import preprocess_data, make_sequences


def strip_data(data_raw_path: str, data_stripped_path: str):
    with open(data_raw_path, "r") as file_in:
        lines = file_in.readlines()
        stripped_lines = [line.strip() + "\n" for line in lines]

    with open(data_stripped_path, "w+") as file_out:
        file_out.writelines(stripped_lines)


def create_header_constants(data_path: str, colnames_path: str):
    with open(data_path, "r") as file:
        headers: list[str] = file.readline().strip().split(" ")

    constants = [f"{name.upper()} = '{name}'\n" for name in headers]

    with open(colnames_path, "w+") as file:
        file.writelines(constants)


def get_sequences(data_path: str, train_perc: float = 0.7):
    if not 0 < train_perc <= 1:
        raise ValueError(f"Train percentage must be in range (0, 1], the value is {train_perc}")

    # if not os.path.isfile(data_path):
    #     strip_data(DATA_STRIPPED_PATH, DATA_STRIPPED_PATH)
    #
    # if not os.path.isfile(COLNAMES_PATH):
    #     create_header_constants(DATA_STRIPPED_PATH, COLNAMES_PATH)
    #     return

    whole_df = pd.read_csv(data_path, sep=" ", dtype=object)

    preprocessed_df, scaler, imputer = preprocess_data(whole_df)
    sequences = make_sequences(preprocessed_df)[:10]
    np.random.shuffle(sequences)

    return sequences
