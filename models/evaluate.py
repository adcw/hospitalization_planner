import torch

from models.utils import get_sequences
from src.nn import StatePredictionModule


def main():
    torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sequences, preprocessor = get_sequences()

    n_attr = sequences[0].shape[1]
    #
    pred_model = StatePredictionModule(n_attr=n_attr, hidden_size=64, device=torch_device, lstm_n_layers=2)
    pred_model.train(sequences=sequences[:12], es_patience=2, epochs=30, kfold_n_splits=5)

    pass


if __name__ == '__main__':
    main()
