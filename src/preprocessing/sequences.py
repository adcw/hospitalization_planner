import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

import data.colnames as c
from data.chosen_colnames import colnames as COLS


def preprocess_data(input_df: pd.DataFrame):
    # params
    cols_to_onehot = [c.POSIEW_SEPSA, c.UREOPLAZMA, c.RDS, c.TYPE_RDS, c.PDA, c.RESPCODE]
    cols_to_impute = [c.PTL, c.TOTAL_BILIRUBIN, c.CREATININE]
    cols_to_scale = [c.FIO2, c.PO2, c.BIRTHWEIGHT]

    cols_to_scale.extend(cols_to_impute)

    processed_df = input_df.copy()
    processed_df = processed_df[COLS]

    # onehot encode
    for col in cols_to_onehot:
        unique_vals = np.unique(processed_df[col].values)

        for val in unique_vals:
            colname = f"{col}_{val}"

            value_mask = processed_df[col] == val

            processed_df[colname] = 0.
            processed_df.loc[value_mask, colname] = 1.

    processed_df.drop(columns=cols_to_onehot, inplace=True)

    # replace literals with values
    processed_df.replace("YES", 1., inplace=True)
    processed_df.replace("NO", 0., inplace=True)
    processed_df.replace("MISSING", np.NAN, inplace=True)

    # impute missing values
    imputer = KNNImputer(n_neighbors=3)
    imputer.columns = cols_to_impute
    processed_df[cols_to_impute] = imputer.fit_transform(processed_df[cols_to_impute])

    # scale values
    scaler = MinMaxScaler()
    scaler.columns = cols_to_scale
    processed_df[cols_to_scale] = scaler.fit_transform(processed_df[cols_to_scale])

    return processed_df, scaler, imputer


def process_group(group: pd.DataFrame):
    group.drop(columns=[c.PATIENTID, c.DATEID], inplace=True)
    return group.values


def make_sequences(input_df: pd.DataFrame):
    group_object = input_df.groupby([c.PATIENTID])

    sequences = [process_group(gr[1]) for gr in group_object]

    return sequences
