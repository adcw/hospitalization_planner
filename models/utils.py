import numpy as np
import pandas as pd
import torch

import data.raw.colnames_original as c
from src.preprocessing import Preprocessor

CSV_PATH = '../data/clean/input.csv'


def get_sequences():
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

    return sequences, preprocessor
