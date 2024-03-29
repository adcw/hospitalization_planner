import random
from copy import deepcopy
from typing import Tuple, List

import numpy as np
import torch
from tqdm import tqdm
from torch.nn.utils.rnn import pack_sequence

from src.config.dataclassess import MainParams
from src.nn.callbacks.metrics import MAECounter


def batch_iter(data, batch_size):
    """
    Iterator yielding batches of specified size from data.

    Args:
    - data: list, array, or any iterable to be batched
    - batch_size: int, size of each batch

    Yields:
    - batch: list, batch of specified size from data
    """
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


def pack_and_iter(data: np.iterable):
    sorted_data = sorted(data, key=lambda x: len(x), reverse=True)
    packed = pack_sequence(sorted_data)

    start = 0
    for size in packed.batch_sizes:
        size = int(size)
        sl = slice(start, start + size)
        start += size
        yield packed.data[sl]


def prev_and_curr(iterator):
    prev = next(iterator)
    for curr in iter(iterator):
        yield prev, curr
        prev = curr


def shuffled(array, p: float = 1):
    p = min(1., max(0., p))

    if p == 0:
        return array

    num_to_shuffle = int(len(array) * p)

    src_indices = random.sample(range(len(array)), num_to_shuffle)
    shuffled_indices = random.sample(src_indices, len(src_indices))

    cp = deepcopy(array)
    for src, dest in zip(src_indices, shuffled_indices):
        cp[dest] = array[src]

    return cp


def forward_sequences(
        sequences: list[torch.Tensor],

        model: torch.nn.Module,
        main_params: MainParams,
        optimizer,
        criterion,

        is_eval: bool = False,
        target_indexes: list[int] | None = None,
        y_cols_in_x: bool = False,
        verbose: bool = True,

        batch_size: int = 16
) -> Tuple[float, float, List[torch.Tensor]]:
    total = sum([len(s) - main_params.n_steps_predict for s in sequences])

    n_feats = sequences[0].shape[1]
    x_cols = set(range(0, n_feats))

    if not y_cols_in_x:
        x_cols = x_cols.difference(target_indexes)

    x_cols = list(x_cols)

    train_progress = tqdm(total=round(total / batch_size)) if verbose else None

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
    for seq_batch in batch_iter(sequences, batch_size):
        h0 = None
        c0 = None

        for input_step, output_step in prev_and_curr(pack_and_iter(seq_batch)):
            input_step: torch.Tensor = input_step[:, x_cols].clone().unsqueeze(1).to(main_params.device)

            if target_indexes is not None:
                output_step = output_step[:, target_indexes].clone().unsqueeze(1).to(main_params.device)

            out_size = output_step.size(0)
            if input_step.size(0) > out_size:
                input_step = input_step[:out_size, :]
                c0 = c0[:, :out_size, :].contiguous()
                h0 = h0[:, :out_size, :].contiguous()

            if not is_eval:
                optimizer.zero_grad()

            if is_eval:
                with torch.no_grad():
                    outputs, (hn, cn) = model(input_step, h0, c0)
            else:
                outputs, (hn, cn) = model(input_step, h0, c0)

            outputs = outputs.unsqueeze(1)

            preds.append(outputs.flatten().cpu().detach().numpy())

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
