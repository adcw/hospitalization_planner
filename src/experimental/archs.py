import torch
import torch.nn as nn
import torch.optim as optim

# Przykładowe dane treningowe
# Zakładamy, że mamy sekwencje obserwacji i odpowiadających akcji na każdym kroku czasowym
# W rzeczywistości dane te powinny być odpowiednio przygotowane
observations = torch.randn(5, 2)  # Przykładowe obserwacje
actions = torch.randint(0, 2, (5,))  # Przykładowe akcje (0 lub 1) w odpowiedzi na obserwacje

# Parametry modelu
input_size = 2
hidden_size = 3
output_size = 2

# Definicja modelu LSTM na poziomie kroku czasowego
class StepTimeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(StepTimeLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0, c0):
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out)
        return out, (hn, cn)

# Inicjalizacja modelu
model = StepTimeLSTM(input_size, hidden_size, output_size)

# Funkcja kosztu i optymalizator
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Przygotowanie danych treningowych na poziomie kroku czasowego
for t in range(observations.size(0)):
    observation = observations[t].view(1, 1, -1)
    action = actions[t].view(1, -1)

    if t == 0:
        # Inicjalizacja stanu ukrytego na początku sekwencji
        h0 = torch.zeros(1, 1, hidden_size)
        c0 = torch.zeros(1, 1, hidden_size)
    else:
        # Przekaż stan ukryty z poprzedniego kroku czasowego
        h0 = hn
        c0 = cn

    # Zerowanie gradientów
    optimizer.zero_grad()

    # Przetwarzanie na poziomie kroku czasowego
    outputs, (hn, cn) = model(observation, h0, c0)

    # Obliczanie straty
    loss = criterion(outputs.view(1, -1), action)

    # Wsteczna propagacja i aktualizacja wag
    loss.backward()
    optimizer.step()

# Teraz model jest wytrenowany na poziomie kroku czasowego
pass