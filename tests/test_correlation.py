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

    ranking_df = pd.DataFrame(ranking, columns=['Kolumna_1', 'Kolumna_2', 'Korelacja'])

    return ranking_df


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

    pass
