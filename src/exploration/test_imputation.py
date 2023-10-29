import pandas as pd
import numpy as np
from numpy.ma import where
from matplotlib import pyplot as plt

import data.colnames as c
from src.preprocessing import transform
from src.preprocessing import Preprocessor

CSV_PATH = '../../data/input.csv'


def groupby_split(df: pd.DataFrame, col: str, perc: float = 0.2):
    groups = df.groupby(col)

    indexes = np.random.choice(groups.ngroups, int(perc * groups.ngroups))

    first_part = []
    second_part = []

    for i, (_, group) in enumerate(groups):
        if i in indexes:
            second_part.append(group)
        else:
            first_part.append(group)

    first_part = pd.concat(first_part)
    second_part = pd.concat(second_part)

    return first_part, second_part


if __name__ == '__main__':
    """
    Test how much error the imputation has
    """
    # read data
    df = pd.read_csv(CSV_PATH, dtype=object)
    df.replace('MISSING', np.nan, inplace=True)
    df.dropna(inplace=True)

    train, test = groupby_split(df, c.PATIENT_ID)
    test_incomplete = test.copy()
    test_incomplete[c.CREATININE] = np.nan

    whole_df = pd.concat([train, test])
    incomplete_df = pd.concat([train, test_incomplete])

    whole_df.reset_index(drop=True, inplace=True)
    incomplete_df.reset_index(drop=True, inplace=True)

    # preprocess: one hot, impute
    onehot_cols = [c.SEPSIS_CULTURE, c.UREAPLASMA, c.RDS, c.RDS_TYPE, c.PDA, c.RESPCODE]
    impute_dict = {c.CREATININE: [c.LEVONOR, c.DOPAMINE, c.PO2, c.BIRTHWEIGHT]}

    preprocessor = Preprocessor()
    preprocessed_df, _ = transform(incomplete_df,
                                             onehot_cols=onehot_cols,
                                             impute_dict=impute_dict,
                                             )

    imputed_indexes = incomplete_df[incomplete_df[c.CREATININE].isna()].index

    real_vals = whole_df.iloc[imputed_indexes]
    imputed_vals = preprocessed_df.iloc[imputed_indexes]

    real_vals_groups = real_vals.groupby(c.PATIENT_ID)
    imputed_vals_groups = imputed_vals.groupby(c.PATIENT_ID)

    i = 0
    for (_, g1), (_, g2) in zip(real_vals_groups, imputed_vals_groups):
        if i > 10:
            break

        plt.plot(g1[c.CREATININE], label="Real creatinine values")
        plt.plot(g2[c.CREATININE], label="Imputed creatinine values")

        plt.legend()
        plt.show()

        i += 1

    real_vals_creatinine = real_vals[c.CREATININE].astype(float)
    imputed_vals_creatinine = imputed_vals[c.CREATININE].astype(float)

    mean_err = np.mean(np.abs((real_vals_creatinine - imputed_vals_creatinine)) / real_vals_creatinine)

    print(f"Mean error: {mean_err}")

    pass
