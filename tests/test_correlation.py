import pandas as pd
import data.colnames_original as c
from data.chosen_colnames import COLS
from src.preprocessing.utils.transform import transform


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

    ranking_df = pd.DataFrame(ranking, columns=['col1', 'col2', 'corr'])

    return ranking_df
"""
ptl
po2
FiO2
ANTYBIOTYK
RTG_RDS
AMINA_PRESYJNA
dopamina
dobutamina
AMINOGLIKOZYD
STERYD
RTG_PDA
GENERAL_PDA_CLOSED
adrenalina
PENICELINA1
GENERAL_SURFACTANT
KARBAPENEM
"""
if __name__ == '__main__':
    df = pd.read_csv("../data/input.csv", usecols=COLS)

    onehot_cols = None

    ranking = {
        c.RESPIRATION: ["WLASNY", "CPAP", "MAP1", "MAP2", "MAP3"]
    }

    preprocessed_df, _ = transform(df, onehot_cols=onehot_cols, rank_dict=ranking)
    preprocessed_df.drop(columns=[c.DATEID, c.PATIENTID], inplace=True)

    corr_ranking = correlation_ranking(preprocessed_df)

    print(corr_ranking.head(20))

    respiration_ranking = corr_ranking[corr_ranking[corr_ranking.columns.values[1]] == c.RESPIRATION]

    vals = respiration_ranking[respiration_ranking.columns.values[2]]
    respiration_ranking = respiration_ranking[abs(vals) > 0.1]

    corr_colnames = respiration_ranking[respiration_ranking.columns.values[0]]


    pass
