from typing import Tuple

import torch
from tqdm import tqdm

from src.config.dataclassess import ModelParams
from src.nn.archs.window_lstm import WindowedConvLSTM
from src.nn.callbacks.metrics import MAECounter


def pad_sequences(
        sequences: list[torch.Tensor],
        window_size: int
):
    return [torch.cat([
        torch.zeros(window_size - len(seq),
                    sequences[0].shape[1],
                    device=sequences[0].device),
        seq
    ]) for seq in sequences]


def windows_and_masks_generator(sequences,
                                window_size: int = 10,
                                batch_size: int = 16,
                                y_columns: list = None,
                                n_predictions: int = 1,

                                ):
    windows = []
    ys = []
    seq_lens = [s.shape[0] for s in sequences]

    def batch():
        nonlocal windows, ys

        masks = [torch.cat([
            torch.ones(window_size - w.shape[0], w.shape[1], device=sequences[0].device, dtype=torch.bool),
            torch.zeros(w.shape[0], w.shape[1], device=sequences[0].device, dtype=torch.bool)
        ]) for w in windows]

        xs = pad_sequences(windows, window_size=window_size)

        return tuple((torch.stack(xs), torch.stack(ys), torch.stack(masks)))

    for seq, seq_len in zip(sequences, seq_lens):
        for i in range(seq_len - n_predictions):
            windows.append(seq[max(i - window_size + 1, 0): i + 1])
            ys.append(seq[i + 1:i + 1 + n_predictions, y_columns])

            if len(windows) == batch_size:
                yield batch()
                windows = []
                ys = []
    if len(windows) != 0:
        yield batch()
        windows = []
        ys = []


def forward_sequences(
        sequences: list[torch.Tensor],

        # TODO: Fill typing
        model: WindowedConvLSTM,
        model_params: ModelParams,
        optimizer,
        criterion,

        is_eval: bool = False,
        target_indexes: list[int] | None = None,
        window_size: int = 10
) -> Tuple[float, float]:
    # train_progress = tqdm(sequences, total=sum([len(s) - model_params.n_steps_predict for s in sequences]))

    # Track overall loss
    loss_sum = 0
    mae_counter = MAECounter()

    generator = windows_and_masks_generator(sequences, window_size,
                                            n_predictions=model_params.n_steps_predict,
                                            batch_size=64,
                                            y_columns=sequences[0].shape[1] - 1)

    # Select proper mode
    if is_eval:
        model.eval()
    else:
        model.train()

    pbar = tqdm(desc="Training...")
    for x, y, m in iter(generator):

        if is_eval:
            with torch.no_grad():
                y_pred = model.forward(x, m)
        else:
            optimizer.zero_grad()
            y_pred = model.forward(x, m)

        loss = criterion(y_pred, y)
        last_loss = loss.item()
        pbar.set_postfix({"Loss": last_loss})
        loss_sum += last_loss

        mae_counter.publish(y_pred, y)

        if not is_eval:
            loss.backward()
            optimizer.step()

        pbar.update(1)
        pass

    total = pbar.n
    pbar.close()

    pass

    # Return mean loss
    mean_loss = loss_sum / total
    mean_mae_loss = mae_counter.retrieve()

    return mean_loss, mean_mae_loss
