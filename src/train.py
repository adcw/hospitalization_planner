import pandas as pd
import torch

import data.colnames as c
from src.preprocessing import Preprocessor

CSV_PATH = '../data/input.csv'


def main():
    torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    device = torch_device

    # read data
    whole_df = pd.read_csv(CSV_PATH, dtype=object)

    # preprocess: one hot, impute
    onehot_cols = [c.SEPSIS_CULTURE, c.UREAPLASMA, c.RDS, c.RDS_TYPE, c.PDA, c.RESPCODE]
    impute_dict = {c.CREATININE: [c.LEVONOR, c.TOTAL_BILIRUBIN, c.PO2]}

    preprocessor = Preprocessor()
    tensors = preprocessor.transform(whole_df,
                                     group_cols=[c.PATIENT_ID],
                                     group_sort_col=c.DATE_ID,
                                     onehot_cols=onehot_cols,
                                     impute_dict=impute_dict,
                                     )

    inv_transformed = preprocessor.inverse_transform(tensors[:1])

    # create sequences
    # sequences = make_sequences(preprocessed_df)

    # n_attr = sequences[0].shape[1]
    #
    # pred_model = StatePredictionModule(n_attr=n_attr, hidden_size=64, device=device, lstm_n_layers=2)
    # pred_model.train(sequences=sequences[:12], es_patience=2, epochs=30, kfold_n_splits=5)

    pass


if __name__ == '__main__':
    main()
