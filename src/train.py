import os

import numpy as np
import pandas as pd
import torch

from src.experimental.triple_arch.archs import StepTimeLSTM
from src.experimental.triple_arch.state_predict_module import StatePredictionModule
from src.preprocessing.clean_data import strip_data, create_header_constants, get_sequences
from src.preprocessing.sequences import preprocess_data, make_sequences

DATA_RAW_PATH = "../data/neonatologia.txt"
DATA_STRIPPED_PATH = "../data/neonatologia_stripped.txt"
COLNAMES_PATH = "../data/colnames.py"


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sequences = get_sequences()

    n_attr = sequences[0].shape[1]

    pred_model = StatePredictionModule(n_attr=n_attr, hidden_size=256, device=device)
    pred_model.train(sequences=sequences, es_patience=2, epochs=30, n_splits=3)

    pass


if __name__ == '__main__':
    main()
