from collections import Counter
from typing import Sequence

from sklearn.model_selection import GroupShuffleSplit


def grouped_train_validation_split(
    samples: Sequence[object],
    labels: Sequence[int],
    groups: Sequence[str],
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    n_splits: int = 1000,
) -> tuple[list[int], list[int]]:
    """Choose a grouped holdout with both classes and balanced label counts."""
    if not len(samples) == len(labels) == len(groups):
        raise ValueError("samples, labels, and groups must have the same length")
    if len(set(labels)) < 2:
        raise ValueError("Grouped validation requires samples from both classes.")

    splitter = GroupShuffleSplit(
        n_splits=n_splits,
        test_size=test_size,
        random_state=random_state,
    )
    total_groups = len(set(groups))
    candidates = []
    for train_indices, validation_indices in splitter.split(samples, labels, groups):
        train_labels = [labels[index] for index in train_indices]
        validation_labels = [labels[index] for index in validation_indices]
        if len(set(train_labels)) < 2 or len(set(validation_labels)) < 2:
            continue

        counts = Counter(validation_labels)
        label_imbalance = abs(counts[0] - counts[1]) / len(validation_labels)
        group_fraction = len({groups[index] for index in validation_indices}) / total_groups
        split_size_error = abs(group_fraction - test_size)
        score = label_imbalance + 0.25 * split_size_error
        candidates.append((score, train_indices, validation_indices))

    if not candidates:
        raise ValueError(
            "Cannot create a grouped train/validation split with both labels in "
            "each partition; add more labeled performer groups."
        )

    _, train_indices, validation_indices = min(candidates, key=lambda item: item[0])
    return train_indices.tolist(), validation_indices.tolist()
