import pandas as pd

import data.colnames as c


def process_group(group: pd.DataFrame):
    group.drop(columns=[c.PATIENTID, c.DATEID], inplace=True)
    return group.values


def make_sequences(input_df: pd.DataFrame):
    group_object = input_df.groupby([c.PATIENTID])

    sequences = [process_group(gr[1]) for gr in group_object]

    return sequences
