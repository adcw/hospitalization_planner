import torch
import torch.nn as nn

# Przykładowe dane
X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=torch.float32)
y = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0], dtype=torch.float32)


# Tworzenie modelu LSTM w PyTorch
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


model = LSTMModel(input_size=1, hidden_size=50)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Proces treningu
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    y_output = model(X.view(1, len(X), 1))
    loss = criterion(y_output.view(-1), y)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Prognozowanie kolejnych kroków
output_sequence = []
X_input = X[0].view(1, 1, 1)  # Początkowy punkt danych

for i in range(len(X)):
    y_output = model(X_input)
    output_sequence.append(y_output.item())

    # Aktualizacja danych wejściowych na podstawie wyniku poprzedniego kroku
    if i < len(X) - 1:
        X_input = torch.tensor([[y_output.item()]], dtype=torch.float32).view(1, 1, 1)

print("Wygenerowana sekwencja:", output_sequence)
