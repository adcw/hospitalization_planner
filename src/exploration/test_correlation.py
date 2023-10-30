import numpy as np
import pandas as pd
import data.colnames as c
from data.chosen_colnames import colnames as COLS
from src.preprocessing import transform
from src.preprocessing.preprocess import Preprocessor


def correlation_ranking(dataframe):
    correlation_matrix = dataframe.corr()

    ranking = []

    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            column1 = correlation_matrix.columns[i]
            column2 = correlation_matrix.columns[j]
            correlation = correlation_matrix.iat[i, j]
            ranking.append((column1, column2, correlation))

    ranking.sort(key=lambda x: abs(x[2]), reverse=True)

    ranking_df = pd.DataFrame(ranking, columns=['Kolumna_1', 'Kolumna_2', 'Korelacja'])

    return ranking_df


if __name__ == '__main__':
    df = pd.read_csv("../../data/input.csv", usecols=COLS)

    df.replace("YES", 1., inplace=True)
    df.replace("NO", 0., inplace=True)
    df.replace("MISSING", np.NAN, inplace=True)

    onehot_cols = [c.SEPSIS_CULTURE, c.RDS_TYPE, c.RESPCODE]

    preprocessed_df, _ = transform(df, onehot_cols=onehot_cols)
    preprocessed_df.drop(columns=[c.DATE_ID, c.PATIENT_ID], inplace=True)

    corr_ranking = correlation_ranking(preprocessed_df)

    print(corr_ranking.head(20))

    pass
