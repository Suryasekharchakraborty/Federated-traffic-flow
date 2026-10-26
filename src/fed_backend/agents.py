"""
agents.py
---------
Agent-based orchestration layer for the federated learning round.
Wraps the existing fedavg.py functions (client_update, fed_avg,
evaluate_global) with client/orchestrator abstractions — the
underlying math is untouched, this adds a coordination layer so the
round can be described and logged as agent-driven, per the proposal's
"orchestrator agent coordinates client sub-agents" architecture.
"""

from __future__ import annotations

from fedavg import client_update, fed_avg, evaluate_global, count_params, TrafficGRU


class ClientAgent:
    """One federated client (simulated traffic intersection)."""

    def __init__(self, client_id: int, client_data: dict):
        self.client_id = client_id
        self.client_data = client_data
        self.name = f"client-{client_id}"

    def run_local_round(self, global_state, num_sensors, local_epochs, lr):
        print(f"[{self.name}] starting local training "
              f"(epochs={local_epochs}, lr={lr})")
        sd, n, mae = client_update(global_state, self.client_data, num_sensors,
                                    local_epochs, lr)
        print(f"[{self.name}] local MAE={mae:.4f}, samples={n}")
        return sd, n, mae


class OrchestratorAgent:
    """Coordinates 3–5 client sub-agents for one FedAvg round."""

    def __init__(self, clients: list[dict]):
        self.client_agents = [ClientAgent(i, c) for i, c in enumerate(clients)]
        self.name = "fedavg-orchestrator"

    def run_round(self, global_state, test_X, test_y, num_sensors,
                  local_epochs=1, lr=1e-2):
        print(f"[{self.name}] dispatching round to "
              f"{len(self.client_agents)} client agents")
        updates, sizes, client_maes = [], [], []

        for agent in self.client_agents:
            sd, n, mae = agent.run_local_round(global_state, num_sensors, local_epochs, lr)
            updates.append(sd)
            sizes.append(n)
            client_maes.append(round(mae, 3))

        print(f"[{self.name}] aggregating {len(updates)} client updates (FedAvg)")
        new_global = fed_avg(updates, sizes)

        fed_mae, fed_rmse = evaluate_global(
            new_global, test_X, test_y, num_sensors,
            std=self.client_agents[0].client_data["std"]
        )
        comm_bytes = count_params(TrafficGRU(num_sensors)) * 4 * 2 * len(self.client_agents)

        print(f"[{self.name}] round complete — Fed MAE={fed_mae:.4f}, "
              f"comm={comm_bytes // 1024} KB")

        return dict(global_state=new_global, fed_mae=fed_mae, fed_rmse=fed_rmse,
                    comm=comm_bytes, client_maes=client_maes)