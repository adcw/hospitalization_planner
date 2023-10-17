import os

import pandas as pd

from src.preprocessing.clean_data import strip_data, create_header_constants
from src.preprocessing.sequences import preprocess_data, make_sequences

DATA_RAW_PATH = "../data/neonatologia.txt"
DATA_STRIPPED_PATH = "../data/neonatologia_stripped.txt"
COLNAMES_PATH = "../data/colnames.py"


def get_sequences():
    if not os.path.isfile(DATA_STRIPPED_PATH):
        strip_data(DATA_STRIPPED_PATH, DATA_STRIPPED_PATH)

    if not os.path.isfile(COLNAMES_PATH):
        create_header_constants(DATA_STRIPPED_PATH, COLNAMES_PATH)
        return

    whole_df = pd.read_csv(DATA_STRIPPED_PATH, sep=" ", dtype=object)

    preprocessed_df, scaler, imputer = preprocess_data(whole_df)

    return make_sequences(preprocessed_df)


def main():
    sequences = get_sequences()

    pass


if __name__ == '__main__':
    main()
