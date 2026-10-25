"""
model.py
--------
A small GRU regressor: takes SEQ_LEN steps of traffic readings across
a client's sensors and predicts the next step for all of them at once.
Deliberately tiny (hackathon-scale) so a federated round trains in
well under a second on CPU.
"""

import torch
import torch.nn as nn


class TrafficGRU(nn.Module):
    def __init__(self, num_sensors: int, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(input_size=num_sensors, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, num_sensors)

    def forward(self, x):
        # x: (batch, seq_len, num_sensors)
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.head(last)  # (batch, num_sensors)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())