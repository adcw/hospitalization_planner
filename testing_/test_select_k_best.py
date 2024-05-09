from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

import pandas as pd
import src.colnames_original as c
from src.chosen_colnames import COLS
from src.preprocessing.utils.transform import transform

CSV_PATH = '../data/input.csv'


def select_features(df, k=15):
    X = df.iloc[:, :-1]  # Wszystkie kolumny oprócz ostatniej
    y = df.iloc[:, -1]  # Ostatnia kolumna jako wartość rozróżniająca

    # Wybór najlepszych k cech
    selector = SelectKBest(score_func=f_classif, k=k)
    fit = selector.fit(X, y)

    # Indeksy wybranych cech
    selected_features_indices = fit.get_support(indices=True)

    # Nazwy wybranych cech
    selected_feature_names = df.columns[selected_features_indices]

    return selected_feature_names


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
    df.dropna(inplace=True)

    ranking = {
        c.RESPIRATION: ["WLASNY", "CPAP", "MAP1", "MAP2", "MAP3"]
    }

    preprocessed_df, _ = transform(df, rank_dict=ranking)
    preprocessed_df.drop(columns=[c.DATEID, c.PATIENTID], inplace=True)

    # # Wybór cech
    selected_features = select_features(preprocessed_df)
    print("Wybrane cechy do modelu:", selected_features)
