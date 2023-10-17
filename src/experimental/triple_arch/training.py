import torch
from torch import nn, optim
from tqdm import tqdm

from src.experimental.triple_arch.archs import StepTimeLSTM
import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True)


def train_state_model(sequences: list[torch.Tensor], model: StepTimeLSTM, epochs: int = 5):
    seq_len = len(sequences)
    plt_margin = min(seq_len // 2, 3)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    loss_history = []

    for epoch in range(epochs):
        progress = tqdm(enumerate(sequences), desc=f"Epoch {epoch}")
        for seq_i, seq in progress:
            h0 = torch.randn((1, model.hidden_size))
            c0 = torch.randn((1, model.hidden_size))

            step_loss_history = None
            if seq_i < plt_margin + 1 or seq_len - plt_margin - 1 < seq_i:
                step_loss_history = []

            for step_i in range(len(seq) - 1):
                input_step: torch.Tensor = seq[step_i].clone()
                output_step: torch.Tensor = seq[step_i + 1].clone()

                input_step = input_step.expand((1, -1))
                output_step = output_step.expand((1, -1))

                optimizer.zero_grad()

                outputs, (hn, cn) = model(input_step, h0, c0)

                loss = criterion(outputs, output_step)
                progress.set_postfix({"Loss": loss.item()})

                if step_loss_history is not None:
                    step_loss_history.append(loss.item())

                h0, c0 = hn.detach(), cn.detach()

                loss.backward()
                optimizer.step()

            if step_loss_history is not None:
                loss_history.append(step_loss_history)

    max_plots = min(len(loss_history), 6)  # Maksymalnie rysujemy 6 wykresów
    num_rows = max_plots // 2 + max_plots % 2  # Liczba wierszy na podwykresy
    plt.figure(figsize=(10, 8))

    for i in range(max_plots):
        plt.subplot(num_rows, 2, i + 1)  # dwie kolumny, obliczona liczba wierszy
        loss_data = loss_history[i]
        plt.plot(loss_data)
        plt.xlabel('Kroki treningowe')
        plt.ylabel('Wartość straty')
        plt.title(f'Wykres Straty {i + 1}')

    plt.tight_layout()
    plt.show()

    pass
