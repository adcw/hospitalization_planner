import numpy as np
import pandas as pd
import torch

import data.raw.colnames_original as c
from src.nn import StatePredictionModule
from src.preprocessing import Preprocessor

CSV_PATH = '../data/clean/input.csv'


def main():
    torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    device = torch_device

    # read data
    whole_df = pd.read_csv(CSV_PATH, dtype=object)

    # replace literals with values
    whole_df.replace("YES", 1., inplace=True)
    whole_df.replace("NO", 0., inplace=True)
    whole_df.replace("MISSING", np.NAN, inplace=True)

    impute_dict = {
        c.CREATININE: [c.LEVONOR, c.TOTAL_BILIRUBIN, c.HEMOSTATYCZNY],
        c.TOTAL_BILIRUBIN: [c.RTG_PDA, c.ANTYBIOTYK, c.PENICELINA1, c.STERYD],
        c.PTL: [c.TOTAL_BILIRUBIN, c.ANTYBIOTYK, c.KARBAPENEM, c.GENERAL_PDA_CLOSED]
    }

    rankings = {
        c.RESPIRATION: ["WLASNY", "CPAP", "MAP1", "MAP2", "MAP3"]
    }

    preprocessor = Preprocessor(group_cols=[c.PATIENTID],
                                group_sort_col=c.DATEID,
                                rank_dict=rankings,
                                impute_dict=impute_dict)

    sequences = preprocessor.fit_transform(whole_df)

    n_attr = sequences[0].shape[1]
    #
    pred_model = StatePredictionModule(n_attr=n_attr, hidden_size=64, device=device, lstm_n_layers=2)
    pred_model.train(sequences=sequences[:12], es_patience=2, epochs=30, kfold_n_splits=5)

    pass


if __name__ == '__main__':
    main()
