import pandas as pd
from matplotlib import pyplot as plt

import data.colnames as c
from src.preprocessing.preprocess import Preprocessor
from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':
    """
    For each patient, either there is complete information about CREATININE, PTL or TOTAL_BILIRUBIN,
    or there is no info about these parameters.
    """
    DATA_STRIPPED_PATH = "../../data/neonatologia_stripped.txt"

    whole_df = pd.read_csv(DATA_STRIPPED_PATH, sep=" ", dtype=object)

    # preprocess: one hot, impute
    onehot_cols = [c.POSIEW_SEPSA, c.UREOPLAZMA, c.RDS, c.TYPE_RDS, c.PDA, c.RESPCODE]

    preprocessor = Preprocessor()
    preprocessed_df = preprocessor.transform(whole_df, onehot_cols=onehot_cols)

    scaler = MinMaxScaler()
    scaled_vals = scaler.fit_transform(preprocessed_df)
    scaled_df = pd.DataFrame(scaled_vals)
    scaled_df.columns = preprocessed_df.columns

    patients = scaled_df.groupby([c.PATIENTID])

    for _, pat in patients:
        if pat[c.CREATININE].isna().sum() == 0:
            plt.plot(pat[c.CREATININE].values)

    plt.title("Tendencies of creatinine")
    plt.show()

    for _, pat in patients:
        if pat[c.PTL].isna().sum() == 0:
            plt.plot(pat[c.PTL].values)

    plt.title("Tendencies of PTL")
    plt.show()

    for _, pat in patients:
        if pat[c.TOTAL_BILIRUBIN].isna().sum() == 0:
            plt.plot(pat[c.TOTAL_BILIRUBIN].values)

    plt.title("Tendencies of bilirubine")
    plt.show()

    pass
