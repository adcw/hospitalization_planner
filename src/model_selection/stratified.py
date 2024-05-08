import random
from collections import defaultdict
from typing import Iterable

from sklearn.model_selection import train_test_split


def stratified_sampling(classes, sample_size):
    class_indices = defaultdict(list)

    for idx, class_label in enumerate(classes):
        class_indices[class_label].append(idx)

    sampled_indices = []
    remaining_sample_size = sample_size

    while remaining_sample_size > 0 and any(class_indices.values()):
        for class_label, indices in list(class_indices.items()):
            if indices:
                sampled_index = random.choice(indices)
                sampled_indices.append(sampled_index)
                indices.remove(sampled_index)
                remaining_sample_size -= 1

                if remaining_sample_size <= 0:
                    break
            else:
                del class_indices[class_label]

    return sampled_indices


def train_test_split_safe(*arrays: Iterable,
                          stratify: Iterable,
                          test_size: float):
    indexes_by_class = {k: [] for k in set(stratify)}

    for i, cls in enumerate(stratify):
        indexes_by_class[cls].append(i)

    train = [[] for _ in arrays]
    test = [[] for _ in arrays]

    for cls, indices in indexes_by_class.items():
        if len(indices) == 1:
            for i, arr in enumerate(arrays):
                train[i].append(arr[indices[0]])
                # test[i].append(arr[indices[0]])
            continue

        tr_indices, ts_indices = train_test_split(indices, test_size=test_size,
                                                  stratify=[stratify[idx] for idx in indices])

        for i, arr in enumerate(arrays):
            train[i].extend([arr[idx] for idx in tr_indices])
            test[i].extend([arr[idx] for idx in ts_indices])

    return *train, *test
