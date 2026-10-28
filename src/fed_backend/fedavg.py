"""
fedavg.py
---------
Core federated-learning logic: local client training, FedAvg
aggregation, a single-round driver (used by the Streamlit app in
place of simulate_round), and a centralized baseline trainer.

Patched:
  * Removed the self-import (`import fedavg` from inside fedavg.py) —
    harmless but fragile; this module already has direct access to its
    own functions.
  * `run_federated_round` now uses ONE `num_sensors` (the round's global
    model) for every client instead of each client's own `n_sensors`.
    FedAvg averages state_dict tensors directly with no shape check, so
    if clients had different input sizes, fed_avg() would either throw a
    shape-mismatch error or silently misalign weights. All clients must
    share the same model architecture for a valid average.
  * The per-round dict key is now `comm_bytes` (not `comm`), matching the
    column name `app.py`'s `load_experiment()` reads from SQLite
    (`row["comm_bytes"]`) — one less translation step for build_db.py.
  * `train_centralized_baseline(..., rounds=...)` — renamed from `epochs`
    and now REQUIRED to equal the federated run's round count. app.py
    zips federated and centralized values round-for-round (Tab 1's
    dashed lines, Tab 3's comparison); a mismatched length there either
    crashes the zip or silently truncates the comparison partway through
    the chart. Passing `rounds=total_rounds` explicitly makes that
    requirement visible at the call site instead of silently defaulting
    to 5 epochs regardless of how many federated rounds were run.
"""

from __future__ import annotations

import copy
from typing import List

import numpy as np
import torch
import torch.nn as nn

from model import TrafficGRU, count_params

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _to_tensor(arr):
    return torch.tensor(arr, dtype=torch.float32, device=DEVICE)


def client_update(global_state_dict, client_data: dict, num_sensors: int,
                   local_epochs: int = 1, lr: float = 1e-2, batch_size: int = 64):
    """Train one client's local copy of the global model for a few epochs.
    Returns (updated_state_dict, num_samples, local_val_mae).

    `num_sensors` must be the SAME value used to build the global model
    for this round (see run_federated_round) — every client's TrafficGRU
    has to share the global model's input shape or fed_avg() cannot
    average the resulting state_dicts."""
    model = TrafficGRU(num_sensors).to(DEVICE)
    model.load_state_dict(global_state_dict)
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X, y = _to_tensor(client_data["X_train"]), _to_tensor(client_data["y_train"])
    n = len(X)
    for _ in range(local_epochs):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            pred = model(X[idx])
            loss = loss_fn(pred, y[idx])
            loss.backward()
            opt.step()

    # local validation MAE (de-normalized back to real units)
    model.eval()
    with torch.no_grad():
        Xv, yv = _to_tensor(client_data["X_val"]), _to_tensor(client_data["y_val"])
        pred = model(Xv)
        mae = torch.mean(torch.abs(pred - yv)).item() * client_data["std"]

    return model.state_dict(), n, mae


def fed_avg(state_dicts: List[dict], sample_counts: List[int]) -> dict:
    """Weighted average of client state_dicts by number of local samples.
    Requires every state_dict to share identical shapes — i.e. every
    client's model was built with the same num_sensors (see client_update
    docstring)."""
    total = sum(sample_counts)
    avg = copy.deepcopy(state_dicts[0])
    for key in avg:
        avg[key] = sum(sd[key] * (n / total) for sd, n in zip(state_dicts, sample_counts))
    return avg


def evaluate_global(state_dict, test_X, test_y, num_sensors, std, mean=0.0):
    model = TrafficGRU(num_sensors).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        X, y = _to_tensor(test_X), _to_tensor(test_y)
        pred = model(X)
        mae = torch.mean(torch.abs(pred - y)).item() * std
        rmse = torch.sqrt(torch.mean((pred - y) ** 2)).item() * std
    return round(mae, 3), round(rmse, 3)


def run_federated_round(global_state_dict, clients, test_X, test_y,
                         num_sensors, local_epochs=1, lr=1e-2):
    """Run one FedAvg round. `num_sensors` is the single, shared input
    size for the global model AND every client this round — do not use a
    per-client sensor count here (see client_update / fed_avg docstrings)."""
    updates, sizes, client_maes = [], [], []
    for c in clients:
        sd, n, mae = client_update(global_state_dict, c, num_sensors, local_epochs, lr)
        updates.append(sd)
        sizes.append(n)
        client_maes.append(round(mae, 3))

    new_global = fed_avg(updates, sizes)
    fed_mae, fed_rmse = evaluate_global(new_global, test_X, test_y, num_sensors, std=clients[0]["std"])
    comm_bytes = count_params(TrafficGRU(num_sensors)) * 4 * 2 * len(clients)

    return dict(global_state=new_global, fed_mae=fed_mae, fed_rmse=fed_rmse,
                comm_bytes=comm_bytes, client_maes=client_maes)


def train_centralized_baseline(X_train, y_train, X_test, y_test, num_sensors, std,
                                rounds: int, lr=1e-2, batch_size=64):
    """Train once on pooled (non-federated) data for the comparison tab.

    `rounds` MUST equal the number of federated rounds this run will be
    compared against — app.py zips the federated and centralized series
    round-for-round (Tab 1's dashed lines, Tab 3's comparison, and the
    Evaluation Report's round-wise table). Returns one (mae, rmse) pair
    per round, training one additional local epoch each time so the two
    series line up exactly."""
    model = TrafficGRU(num_sensors).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    X, y = _to_tensor(X_train), _to_tensor(y_train)
    n = len(X)
    history = []
    for _ in range(rounds):
        perm = torch.randperm(n, device=DEVICE)
        model.train()
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(X[idx]), y[idx])
            loss.backward()
            opt.step()
        mae, rmse = evaluate_global(model.state_dict(), X_test, y_test, num_sensors, std)
        history.append((mae, rmse))
    return history