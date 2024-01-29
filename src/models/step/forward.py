from typing import Tuple, List

import torch
from tqdm import tqdm

from src.config.dataclassess import MainParams
from src.nn.callbacks.metrics import MAECounter


def forward_sequences(
        sequences: list[torch.Tensor],

        model: torch.nn.Module,
        main_params: MainParams,
        optimizer,
        criterion,

        is_eval: bool = False,
        target_indexes: list[int] | None = None,
        y_cols_in_x: bool = False,
        verbose: bool = True
) -> Tuple[float, float, List[torch.Tensor]]:
    total = sum([len(s) - main_params.n_steps_predict for s in sequences])

    n_feats = sequences[0].shape[1]
    x_cols = set(range(0, n_feats))

    if not y_cols_in_x:
        x_cols = x_cols.difference(target_indexes)

    x_cols = list(x_cols)

    train_progress = tqdm(sequences, total=total) if verbose else None

    # Track overall loss
    loss_sum = 0
    mae_counter = MAECounter()
    preds = []

    # Select proper mode
    if is_eval:
        model.eval()
    else:
        model.train()

    # Forward all sequences
    for _, seq in enumerate(sequences):
        h0 = None
        c0 = None

        # Iterate over sequence_df
        for step_i in range(len(seq) - main_params.n_steps_predict):

            # Get input and output data
            input_step: torch.Tensor = seq[step_i, x_cols].clone()
            output_step: torch.Tensor = seq[step_i + 1:step_i + 1 + main_params.n_steps_predict].clone()

            if target_indexes is not None:
                output_step = output_step[:, target_indexes]

            input_step = input_step.expand((1, -1)).to(main_params.device)

            if main_params.n_steps_predict == 1:
                output_step = output_step.expand((1, -1)).to(main_params.device)

            if not is_eval:
                optimizer.zero_grad()

            if is_eval:
                with torch.no_grad():
                    outputs, (hn, cn) = model(input_step, h0, c0)
            else:
                outputs, (hn, cn) = model(input_step, h0, c0)

            outputs = outputs.view(main_params.n_steps_predict,
                                   round(outputs.shape[1] / main_params.n_steps_predict))

            preds.append(outputs)

            # Calculate losses
            loss = criterion(outputs, output_step)
            last_loss = loss.item()

            train_progress and train_progress.set_postfix({"Loss": last_loss})
            loss_sum += last_loss

            mae_counter.publish(outputs, output_step)

            # preserve internal LSTM states
            h0, c0 = hn.detach(), cn.detach()

            # Back-propagation
            if not is_eval:
                loss.backward()
                optimizer.step()

            train_progress and train_progress.update(1)

    # Return mean loss
    mean_loss = loss_sum / total
    mean_mae_loss = mae_counter.retrieve()

    return mean_loss, mean_mae_loss, preds
