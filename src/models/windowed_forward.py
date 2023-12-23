from typing import Tuple

import torch

from src.config.dataclassess import ModelParams
from src.nn.archs.windowed_lstm import WindowedLSTM
from src.nn.callbacks.metrics import MAECounter


def windows_and_masks_generator(sequences,
                                window_size: int = 10,
                                batch_size: int = 16,
                                y_columns: list = None,
                                n_predictions: int = 1):
    windows = []
    ys = []
    seq_lens = [s.shape[0] for s in sequences]

    def batch():
        nonlocal windows, ys

        masks = [torch.cat([
            torch.zeros(window_size - w.shape[0], w.shape[1]),
            torch.ones(w.shape[0], w.shape[1])
        ]) for w in windows]

        xs = [
            torch.cat([
                torch.zeros(window_size - len(window),
                            sequences[0].shape[1],
                            device=sequences[0].device),
                window
            ]) for window in windows]

        if y_columns is not None:
            ys = [
                seq[win_index + 1:win_index + 1 + n_predictions, y_columns] for win_index in range(len(windows))
            ]

        return tuple((xs, ys, masks))

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


def windowed_forward(
        sequences: list[torch.Tensor],

        # TODO: Fill typing
        model: WindowedLSTM,
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

    generator = windows_and_masks_generator(sequences[:2], window_size, n_predictions=3,
                                            y_columns=sequences[0].shape[1] - 1)

    for x, y, m in iter(generator):
        pass

    pass

    # FORWARD

    # FORWARD END

    """
    if is_eval:
        with torch.no_grad():
            outputs, (hn, cn) = model(input_step, h0, c0)
    else:
        outputs, (hn, cn) = model(input_step, h0, c0)

    outputs = outputs.view(model_params.n_steps_predict,
                           round(outputs.shape[1] / model_params.n_steps_predict))

    # Calculate losses
    loss = criterion(outputs, output_step)
    last_loss = loss.item()
    train_progress.set_postfix({"Loss": last_loss})
    loss_sum += last_loss

    mae_counter.publish(outputs, output_step)

    # preserve internal LSTM states
    h0, c0 = hn.detach(), cn.detach()

    # Back-propagation
    if not is_eval:
        loss.backward()
        optimizer.step()

    train_progress.update(1)
    """

    # Return mean loss
    mean_loss = loss_sum / ...  # train_progress.total
    mean_mae_loss = mae_counter.retrieve()

    return mean_loss, mean_mae_loss


pass
