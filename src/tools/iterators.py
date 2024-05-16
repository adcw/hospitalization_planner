import random
from copy import deepcopy
from typing import Iterable, List, Any, Sized

import numpy as np
from torch.nn.utils.rnn import pack_sequence

import numpy as np


def batch_iter(*arrays, batch_size):
    """
    Iterator yielding batches of specified size from multiple arrays.

    Args:
    - *arrays: Multiple arrays to be batched
    - batch_size: int, size of each batch

    Yields:
    - batch: tuple, batch of specified size from each array
    """
    # Check if all arrays have the same length
    lens = [len(arr) for arr in arrays]
    if not np.all(np.array(lens) == lens[0]):
        raise ValueError(f"Expected all arrays lengths to be equal, received {lens}")

    # Iterate over the arrays and yield batches
    for i in range(0, lens[0], batch_size):
        batches = [data[i:i + batch_size] for data in arrays]
        yield tuple(batches)


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
    if p == 0:
        return array

    p = min(1., max(0., p))

    num_to_shuffle = int(len(array) * p)

    src_indices = random.sample(range(len(array)), num_to_shuffle)
    shuffled_indices = random.sample(src_indices, len(src_indices))

    cp = deepcopy(array)
    for src, dest in zip(src_indices, shuffled_indices):
        cp[dest] = array[src]

    return cp


def windowed(array: Sized,
             window_size: int,
             stride: int) -> Iterable[List[Any]]:
    """
    Generate sliding windows over the given array with the specified window size and stride.

    Args:
        array (Iterable): The input array.
        window_size (int): The size of the window.
        stride (int): The stride between consecutive windows.

    Yields:
        Iterable: A generator yielding windows of size `window_size` over the input array.
    """
    array_len = len(array)

    if array_len <= window_size:
        yield array
    else:
        for i in range(0, len(array) - window_size + 1, stride):
            yield array[i:i + window_size]
