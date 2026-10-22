"""Plant-level split construction, with repeated observations kept together."""

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold


def grouped_folds(y, plants, n_splits=5, seed=42):
    y, plants = np.asarray(y), np.asarray(plants)
    if y.ndim != 1 or plants.shape != y.shape:
        raise ValueError("One label and globally unique plant ID required per row")
    unique = np.unique(plants)
    labels = []
    for plant in unique:
        found = np.unique(y[plants == plant])
        if len(found) != 1:
            raise ValueError("A plant must have one treatment label across all days")
        labels.append(found[0])
    labels = np.asarray(labels)
    if np.min(np.unique(labels, return_counts=True)[1]) < n_splits:
        raise ValueError("Too few independent plants per class for requested folds")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train, test in cv.split(unique, labels):
        yield (
            np.flatnonzero(np.isin(plants, unique[train])),
            np.flatnonzero(np.isin(plants, unique[test])),
        )


def cycle_folds(plants, cycles):
    plants, cycles = np.asarray(plants), np.asarray(cycles)
    if plants.ndim != 1 or cycles.shape != plants.shape or len(np.unique(cycles)) < 2:
        raise ValueError("At least two cycles with matching plant IDs required")
    for plant in np.unique(plants):
        if len(np.unique(cycles[plants == plant])) != 1:
            raise ValueError("Plant IDs must be globally unique across cycles")
    yield from LeaveOneGroupOut().split(plants, groups=cycles)
