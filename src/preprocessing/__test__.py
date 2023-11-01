import numpy as np
import pandas as pd
import torch

import data.raw.colnames as c
from src.preprocessing import Preprocessor

CSV_PATH = '../../data/clean/input.csv'


def main():
    """
    Contents of inv_transformed and whole_df should have identical values
    """
    # read data
    whole_df = pd.read_csv(CSV_PATH, dtype=object)

    # replace literals with values
    whole_df.replace("YES", 1., inplace=True)
    whole_df.replace("NO", 0., inplace=True)
    whole_df.replace("MISSING", np.NAN, inplace=True)

    # preprocess: one hot, impute
    onehot_cols = [c.SEPSIS_CULTURE, c.UREAPLASMA, c.RDS, c.RDS_TYPE, c.PDA, c.RESPCODE]
    group_cols = [c.PATIENT_ID]

    preprocessor = Preprocessor(group_cols=group_cols,
                                group_sort_col=c.DATE_ID,
                                onehot_cols=onehot_cols,
                                impute_dict=None, )

    tensors = preprocessor.fit_transform(whole_df)
    whole_df = whole_df.iloc[:, 2:].reset_index(drop=True)

    inv_transformed = preprocessor.inverse_transform(tensors)
    inv_transformed = pd.concat(inv_transformed, ignore_index=True)

    pass


if __name__ == '__main__':
    main()
