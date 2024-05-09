import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

if __name__ == '__main__':
    # Przykładowe dane
    sequences = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0], [6, 0, 0, 0, 0]])
    lengths = [3, 2, 1]
    window_size = 3

    # Tworzenie okienek z pacjentów
    windows = []
    for patient, pat_len in zip(sequences, lengths):
        for i in range(pat_len):
            windows.append(patient[max(i - window_size + 1, 0): i + 1])

    # Padowanie lewej strony krótszych okienek
    padded_windows = [torch.cat([torch.zeros(window_size - len(window)), window]) for window in windows]

    # Tworzenie maski uwagi
    mask = torch.tensor([[0] * (window_size - len(window)) + [1] * len(window) for window in windows])

    # Konwersja na tensor
    padded_windows_tensor = torch.stack(padded_windows)
    mask_tensor = torch.tensor(mask, dtype=torch.bool)

    # Sortowanie okienek według długości (malejąco)
    sorted_lengths, sorted_idx = torch.sort(torch.tensor([len(window) for window in padded_windows]), descending=True)
    sorted_windows = padded_windows_tensor[sorted_idx]
    sorted_mask = mask_tensor[sorted_idx]

    # Pakowanie okienek
    packed_windows = pack_padded_sequence(sorted_windows, sorted_lengths, batch_first=True, enforce_sorted=False)

    # Tworzenie modelu LSTM
    input_size = 1
    hidden_size = 3
    num_layers = 1

    lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

    # Przekazywanie packed_windows jako wejścia do LSTM
    lstm_out, lstm_hidden = lstm(packed_windows)

    # Odpakowywanie wyników po przejściu przez LSTM
    unpacked_lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)

    # Wydrukowanie wyników
    print("Packed Windows:")
    print(packed_windows)

    print("\nLSTM Output (unpacked):")
    print(unpacked_lstm_out)
