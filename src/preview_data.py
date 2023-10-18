import os

import numpy as np
import pandas as pd
import torch

from src.experimental.triple_arch.archs import StepTimeLSTM
from src.experimental.triple_arch.training import StatePredictionModule
from src.preprocessing.clean_data import strip_data, create_header_constants
from src.preprocessing.sequences import preprocess_data, make_sequences

DATA_RAW_PATH = "../data/neonatologia.txt"
DATA_STRIPPED_PATH = "../data/neonatologia_stripped.txt"
COLNAMES_PATH = "../data/colnames.py"


def get_sequences(train_perc: float = 0.7):
    if not 0 < train_perc <= 1:
        raise ValueError(f"Train percentage must be in range (0, 1], the value is {train_perc}")

    if not os.path.isfile(DATA_STRIPPED_PATH):
        strip_data(DATA_STRIPPED_PATH, DATA_STRIPPED_PATH)

    if not os.path.isfile(COLNAMES_PATH):
        create_header_constants(DATA_STRIPPED_PATH, COLNAMES_PATH)
        return

    whole_df = pd.read_csv(DATA_STRIPPED_PATH, sep=" ", dtype=object)

    preprocessed_df, scaler, imputer = preprocess_data(whole_df)
    sequences = make_sequences(preprocessed_df)[:10]
    np.random.shuffle(sequences)

    return sequences


def main():
    sequences = get_sequences()

    n_attr = sequences[0].shape[1]

    pred_model = StatePredictionModule(n_attr=n_attr, hidden_size=256, device="cuda")
    pred_model.train(sequences=sequences, es_patience=3, epochs=30, n_splits=5)

    # train_state_model(train_sequences=train_seq, val_sequences=val_seq, model=state_model, device="cuda")

    pass


if __name__ == '__main__':
    main()
