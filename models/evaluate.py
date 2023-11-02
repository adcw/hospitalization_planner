import torch

from models.utils import get_sequences
from src.nn import StatePredictionModule, NetParams


def main():
    torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sequences, preprocessor = get_sequences()

    n_attr = sequences[0].shape[1]

    params = NetParams(n_attr=n_attr, device=torch_device, hidden_size=64, n_lstm_layers=2)
    pred_model = StatePredictionModule(net_params=params)

    pred_model.train(sequences=sequences[:10], es_patience=2, epochs=30, kfold_n_splits=5, mode='eval')

    pass


if __name__ == '__main__':
    main()
