import torch

from models.utils import get_sequences
from src.nn import StatePredictionModule, NetParams

from sklearn.model_selection import train_test_split


def main():
    torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sequences, preprocessor = get_sequences()
    sequences = sequences[:6]

    train_seq = sequences[:-1]
    test_seq = sequences[-1]

    n_attr = sequences[0].shape[1]

    params = NetParams(n_attr=n_attr, device=torch_device, hidden_size=64, n_lstm_layers=2)
    pred_model = StatePredictionModule(net_params=params)

    pred_model.train(sequences=train_seq, es_patience=5, epochs=30, mode='train')

    pass


if __name__ == '__main__':
    main()
