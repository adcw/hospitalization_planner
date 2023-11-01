import os.path

import numpy as np
import pandas as pd
from data.raw.chosen_colnames import COLS, c
from src.preprocessing import onehot2categorical, OH2CData

DATA_RAW_PATH = '../data/raw/neonatologia_raw.txt'
DATA_STRIPPED_PATH = '../data/raw/neonatologia_stripped.txt'
COLNAMES_ORIGINAL_FILE = '../data/raw/colnames_original.py'
COLNAMES_RENAMED_FILE = '../data/raw/colnames_renamed.py'
DATA_CLEAN_PATH = "../data/clean/input.csv"


# COLNAMES_PATH


def strip_data(data_raw_path: str, data_stripped_path: str):
    with open(data_raw_path, "r") as file_in:
        lines = file_in.readlines()
        stripped_lines = [line.strip() + "\n" for line in lines]

    with open(data_stripped_path, "w+") as file_out:
        file_out.writelines(stripped_lines)


def create_header_constants(data_path: str, colnames_path: str):
    with open(data_path, "r") as file:
        headers: list[str] = file.readline().strip().split(",")

    constants = [f"{name.upper()} = '{name}'\n" for name in headers]

    with open(colnames_path, "w+") as file:
        file.writelines(constants)


if __name__ == '__main__':
    # strip whitespaces
    if not os.path.exists(DATA_STRIPPED_PATH):
        strip_data(DATA_RAW_PATH, DATA_STRIPPED_PATH)

    df = pd.read_csv(DATA_STRIPPED_PATH, sep=" ", usecols=COLS)
    df = df[COLS]

    df.replace("MISSING", np.nan, inplace=True)
    df.replace("NO", 0., inplace=True)
    df.replace("YES", 1., inplace=True)

    onehot_dict = {
        'respiration': OH2CData(input_colnames=[c.CPAP, c.MAP1, c.MAP2, c.MAP3], threshold=0.1, outlier_class='WLASNY')
    }

    df = onehot2categorical(df, onehot_dict=onehot_dict, drop=True)

    df.to_csv(DATA_CLEAN_PATH, index=False)

    if not os.path.exists(COLNAMES_ORIGINAL_FILE):
        create_header_constants(DATA_CLEAN_PATH, COLNAMES_ORIGINAL_FILE)

    # Choose columns
    # Change their values

    pass
