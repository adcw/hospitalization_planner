import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import src.colnames_original as c
from src.preprocessing.utils.transform import transform

CSV_PATH = '../data/input.csv'


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

    train, test = groupby_split(df, c.PATIENTID)
    test_incomplete = test.copy()
    test_incomplete[c.CREATININE] = np.nan

    whole_df = pd.concat([train, test])
    incomplete_df = pd.concat([train, test_incomplete])

    whole_df.reset_index(drop=True, inplace=True)
    incomplete_df.reset_index(drop=True, inplace=True)

    # preprocess: one hot, impute
    impute_dict = {
        c.CREATININE: [c.LEVONOR, c.TOTAL_BILIRUBIN, c.HEMOSTATYCZNY],
        c.TOTAL_BILIRUBIN: [c.RTG_PDA, c.ANTYBIOTYK, c.PENICELINA1, c.STERYD],
        c.PTL: [c.TOTAL_BILIRUBIN, c.ANTYBIOTYK, c.KARBAPENEM, c.GENERAL_PDA_CLOSED]
    }

    rankings = {
        c.RESPIRATION: ["WLASNY", "CPAP", "MAP1", "MAP2", "MAP3"]
    }

    # preprocessor = Preprocessor(impute_dict=impute_dict)
    preprocessed_df, _ = transform(incomplete_df, impute_dict=impute_dict, rank_dict=rankings)

    imputed_indexes = incomplete_df[incomplete_df[c.CREATININE].isna()].index

    real_vals = whole_df.iloc[imputed_indexes]
    imputed_vals = preprocessed_df.iloc[imputed_indexes]

    real_vals_groups = real_vals.groupby(c.PATIENTID)
    imputed_vals_groups = imputed_vals.groupby(c.PATIENTID)

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
