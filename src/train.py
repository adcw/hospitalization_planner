import pandas as pd
import torch
import data.colnames as c

from src.experimental.triple_arch.state_predict_module import StatePredictionModule
from src.preprocessing.preprocess import Preprocessor
# from src.preprocessing.preprocess import preprocess_data
from src.preprocessing.sequences import make_sequences

DATA_RAW_PATH = "../data/neonatologia.txt"
DATA_STRIPPED_PATH = "../data/neonatologia_stripped.txt"
COLNAMES_PATH = "../data/colnames.py"


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # if not os.path.isfile(data_path):
    #     strip_data(DATA_STRIPPED_PATH, DATA_STRIPPED_PATH)
    #
    # if not os.path.isfile(COLNAMES_PATH):
    #     create_header_constants(DATA_STRIPPED_PATH, COLNAMES_PATH)
    #     return

    # read data
    whole_df = pd.read_csv(DATA_STRIPPED_PATH, sep=" ", dtype=object)

    # preprocess: one hot, impute
    onehot_cols = [c.POSIEW_SEPSA, c.UREOPLAZMA, c.RDS, c.TYPE_RDS, c.PDA, c.RESPCODE]

    preprocessor = Preprocessor()
    preprocessed_df = preprocessor.transform(whole_df, onehot_cols=onehot_cols, impute=True)

    # create sequences
    sequences = make_sequences(preprocessed_df)

    n_attr = sequences[0].shape[1]

    pred_model = StatePredictionModule(n_attr=n_attr, hidden_size=256, device=device)
    pred_model.evaluate(sequences=sequences[:20], es_patience=2, epochs=30, n_splits=5)

    pass


if __name__ == '__main__':
    main()
