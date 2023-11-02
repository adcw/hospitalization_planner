import torch

from models.utils import get_sequences
from src.nn import StatePredictionModule

from sklearn.model_selection import train_test_split


def main():
    torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sequences, preprocessor = get_sequences()
    sequences = sequences[:20]

    train_seq = sequences[:-1]
    test_seq = sequences[-1]

    n_attr = sequences[0].shape[1]

    pred_model = StatePredictionModule(n_attr=n_attr, hidden_size=64, device=torch_device, lstm_n_layers=2)
    pred_model.train(sequences=train_seq, es_patience=2, epochs=30)

    pass


if __name__ == '__main__':
    main()
