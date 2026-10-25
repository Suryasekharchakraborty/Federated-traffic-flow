"""
data_loader.py
--------------
Loads METR-LA, builds sliding-window (X -> next-step) samples, and
partitions sensors/windows across simulated federated clients.

Expects: data/metr-la.h5  (standard DCRNN-format file: a single
pandas DataFrame, shape ~ (34272, 207), index = 5-min timestamps,
columns = sensor IDs). Download from the DCRNN repo:
https://github.com/liyaguang/DCRNN  (data/metr-la.h5)
~6 MB, not 2-3 GB like UTD-19.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEQ_LEN    = 12   # 12 steps × 5 min = 1 hour of history
HORIZON    = 1    # predict 1 step (5 min) ahead
TRAIN_FRAC = 0.7
VAL_FRAC   = 0.1  # remainder is test


def load_raw(h5_path: str = "data/metr-la.h5") -> pd.DataFrame:
    df = pd.read_hdf(h5_path)
    df = df.ffill().bfill()   # ffill/bfill without deprecated method= kwarg
    return df


def make_windows(arr: np.ndarray,
                 seq_len: int = SEQ_LEN,
                 horizon: int = HORIZON):
    """arr: (T, N)  ->  X: (samples, seq_len, N),  y: (samples, N)"""
    X, y = [], []
    for t in range(seq_len, len(arr) - horizon + 1):
        X.append(arr[t - seq_len : t])
        y.append(arr[t + horizon - 1])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def chronological_split(X: np.ndarray, y: np.ndarray):
    n       = len(X)
    n_train = int(n * TRAIN_FRAC)
    n_val   = int(n * VAL_FRAC)
    return (
        (X[:n_train],              y[:n_train]),
        (X[n_train : n_train + n_val], y[n_train : n_train + n_val]),
        (X[n_train + n_val :],     y[n_train + n_val :]),
    )


def normalize(train_X: np.ndarray, *others, mean=None, std=None):
    if mean is None:
        mean = train_X.mean()
        std  = train_X.std() + 1e-6
    out = [(train_X - mean) / std] + [(o - mean) / std for o in others]
    return out, mean, std


def partition_sensors_for_clients(num_sensors: int, num_clients: int,
                                   mode: str, seed: int):
    """
    IID     -> sensors shuffled then split evenly.
    Non-IID -> sensors kept in column order and chunked, so each client
               owns a contiguous geographic block (traffic patterns differ).
    Returns list[np.ndarray], length = num_clients.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(num_sensors)
    if mode == "IID":
        rng.shuffle(idx)
    return np.array_split(idx, num_clients)


def build_federated_dataset(h5_path: str,
                             num_clients: int,
                             mode: str,
                             seed: int,
                             max_sensors: int | None = 60):
    """
    Returns
    -------
    clients  : list of dicts {X_train, y_train, X_val, y_val, mean, std, n_sensors}
    test     : (X_test, y_test)  – shared global test set
    meta     : {num_sensors, seq_len}
    """
    df  = load_raw(h5_path)
    arr = df.values                          # (T, 207)
    if max_sensors:
        arr = arr[:, :max_sensors]

    sensor_groups = partition_sensors_for_clients(arr.shape[1], num_clients, mode, seed)

    clients: list[dict] = []
    all_test_X, all_test_y = [], []

    for sensors in sensor_groups:
        sub = arr[:, sensors]                # (T, k)
        X, y = make_windows(sub)
        (Xtr, ytr), (Xval, yval), (Xte, yte) = chronological_split(X, y)

        (Xtr, Xval, Xte), mean, std = normalize(Xtr, Xval, Xte)
        ytr  = (ytr  - mean) / std
        yval = (yval - mean) / std
        yte  = (yte  - mean) / std

        clients.append(dict(X_train=Xtr, y_train=ytr,
                            X_val=Xval,   y_val=yval,
                            mean=mean,    std=std,
                            n_sensors=len(sensors)))
        all_test_X.append(Xte)
        all_test_y.append(yte)

    # global test set – truncate to shortest client then concatenate
    min_len = min(len(x) for x in all_test_X)
    test_X  = np.concatenate([x[:min_len] for x in all_test_X], axis=-1)
    test_y  = np.concatenate([y[:min_len] for y in all_test_y], axis=-1)

    meta = dict(num_sensors=arr.shape[1], seq_len=SEQ_LEN)
    return clients, (test_X, test_y), meta