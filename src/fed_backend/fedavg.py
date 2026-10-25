"""
fedavg.py
---------
Core federated-learning logic: local client training, FedAvg
aggregation, a single-round driver (used by the Streamlit app in
place of simulate_round), and a centralized baseline trainer.
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
    Returns (updated_state_dict, num_samples, local_val_mae)."""
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
    """Weighted average of client state_dicts by number of local samples."""
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


import fedavg

def run_federated_round_fixed(global_state_dict, clients, test_X, test_y,
                               num_sensors, local_epochs=1, lr=1e-2):
    updates, sizes, client_maes = [], [], []
    for c in clients:
        sd, n, mae = fedavg.client_update(global_state_dict, c, num_sensors, local_epochs, lr)
        updates.append(sd)
        sizes.append(n)
        client_maes.append(round(mae, 3))

    new_global = fedavg.fed_avg(updates, sizes)
    fed_mae, fed_rmse = fedavg.evaluate_global(new_global, test_X, test_y, num_sensors, std=clients[0]["std"])
    comm_bytes = fedavg.count_params(fedavg.TrafficGRU(num_sensors)) * 4 * 2 * len(clients)

    return dict(global_state=new_global, fed_mae=fed_mae, fed_rmse=fed_rmse,
                comm=comm_bytes, client_maes=client_maes)

run_federated_round = run_federated_round_fixed


def train_centralized_baseline(X_train, y_train, X_test, y_test, num_sensors, std,
                                epochs=5, lr=1e-2, batch_size=64):
    """Train once on pooled (non-federated) data for the comparison tab.
    Returns per-epoch (mae, rmse) so you can line it up round-for-round,
    or just call once and reuse the final value across all rounds."""
    model = TrafficGRU(num_sensors).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    X, y = _to_tensor(X_train), _to_tensor(y_train)
    n = len(X)
    history = []
    for _ in range(epochs):
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