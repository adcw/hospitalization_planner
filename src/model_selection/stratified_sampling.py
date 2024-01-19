import random
from collections import defaultdict


def stratified_sampling(classes, sample_size):
    class_indices = defaultdict(list)

    for idx, class_label in enumerate(classes):
        class_indices[class_label].append(idx)

    sampled_indices = []
    remaining_sample_size = sample_size

    while remaining_sample_size > 0:
        for class_label, indices in class_indices.items():
            if indices:
                sampled_index = random.choice(indices)
                sampled_indices.append(sampled_index)
                indices.remove(sampled_index)
                remaining_sample_size -= 1

    return sampled_indices
