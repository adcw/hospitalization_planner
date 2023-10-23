import pandas as pd
import torch
import data.colnames as c

from src.experimental.triple_arch.state_predict_module import StatePredictionModule
from src.preprocessing.preprocess import Preprocessor
# from src.preprocessing.preprocess import preprocess_data
from src.preprocessing.sequences import make_sequences

CSV_PATH = '../data/input.csv'


def main(torch_device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    device = torch_device

    # read data
    whole_df = pd.read_csv(CSV_PATH, dtype=object)

    # preprocess: one hot, impute
    onehot_cols = [c.SEPSIS_CULTURE, c.UREAPLASMA, c.RDS, c.RDS_TYPE, c.PDA, c.RESPCODE]

    preprocessor = Preprocessor()
    preprocessed_df = preprocessor.transform(whole_df, onehot_cols=onehot_cols, impute=True)

    # create sequences
    sequences = make_sequences(preprocessed_df)

    n_attr = sequences[0].shape[1]

    pred_model = StatePredictionModule(n_attr=n_attr, hidden_size=64, device=device, lstm_n_layers=2)
    pred_model.train(sequences=sequences[:12], es_patience=2, epochs=30, kfold_n_splits=5)

    pass


if __name__ == '__main__':
    main()
