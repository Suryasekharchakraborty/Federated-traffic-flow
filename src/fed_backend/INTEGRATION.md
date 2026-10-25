# Wiring fedavg.py into app.py

## 1. Folder layout
```
project/
├── app.py                 (your existing Streamlit file)
├── data/
│   └── metr-la.h5         (download this — see below)
└── fed_backend/
    ├── data_loader.py
    ├── model.py
    └── fedavg.py
```

## 2. Get the dataset
METR-LA as a single `.h5` is ~6 MB (not the 2-3 GB you were worried about —
that's the raw UTD-19 CSVs). Get it from either:
- DCRNN repo: https://github.com/liyaguang/DCRNN (data/metr-la.h5)
- Graph WaveNet repo mirror: https://github.com/nnzhan/Graph-WaveNet
Put the file at `data/metr-la.h5`.

## 3. Edits to app.py

**Top of file, with the other imports:**
```python
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent / "fed_backend"))
from data_loader import build_federated_dataset
from fedavg import run_federated_round, train_centralized_baseline
from model import TrafficGRU
```

**Replace `simulate_round()` entirely** — it's no longer needed. Instead,
right where `run_clicked` currently kicks off the loop, do a one-time
dataset build + centralized baseline, then call `run_federated_round`
per round instead of `simulate_round`:

```python
if run_clicked:
    ... # (keep all your existing state resets exactly as-is)

    clients, (test_X, test_y), meta = build_federated_dataset(
        h5_path="data/metr-la.h5",
        num_clients=sel_clients,
        mode=sel_mode,
        seed=sel_seed,
        max_sensors=60,          # raise later if you have time to spare
    )
    num_sensors = meta["num_sensors"]

    # centralized baseline, trained once up front (pooled client data)
    import numpy as np
    pooled_X = np.concatenate([c["X_train"] for c in clients], axis=-1)[:len(clients[0]["X_train"])]
    pooled_y = np.concatenate([c["y_train"] for c in clients], axis=-1)[:len(clients[0]["y_train"])]
    central_hist = train_centralized_baseline(
        pooled_X, pooled_y, test_X, test_y, num_sensors,
        std=clients[0]["std"], epochs=sel_rounds,
    )

    global_state = TrafficGRU(num_sensors).state_dict()

    for rnd in range(1, sel_rounds + 1):
        result = run_federated_round(global_state, clients, test_X, test_y,
                                      num_sensors, local_epochs=1, lr=1e-2)
        global_state = result["global_state"]

        st.session_state.fed_mae.append(result["fed_mae"])
        st.session_state.fed_rmse.append(result["fed_rmse"])
        st.session_state.central_mae.append(central_hist[rnd - 1][0])
        st.session_state.central_rmse.append(central_hist[rnd - 1][1])
        st.session_state.comm_bytes.append(result["comm"])
        st.session_state.client_status = result["client_maes"]
        st.session_state.client_hist.append(result["client_maes"])
        st.session_state.rounds_done = rnd

        add_log(f"Round {rnd:02d} · Fed MAE={result['fed_mae']} · RMSE={result['fed_rmse']}", "ok")
        render_metrics(); render_tabs(); render_side()
        time.sleep(delay)  # keep this if you still want a visible pacing effect
```

Everything else in `app.py` — the CSS, header, KPI cards, tabs, side
panel, reset logic — needs **zero changes**. The UI only ever reads
`st.session_state.fed_mae`, `.comm_bytes`, etc., and doesn't care
whether those numbers came from `np.random` or real PyTorch training.

## 4. Where to actually run this
- **Locally**: `pip install streamlit plotly numpy torch pandas tables` then
  `streamlit run app.py` from the `project/` folder. (`tables` is needed
  for pandas to read `.h5`.)
- **Google Antigravity**: good for writing/debugging `fedavg.py` and
  `data_loader.py` with agent help (e.g. tuning `local_epochs`, adding
  a real `FedProx`-style variant for the "advanced FL" bonus points),
  but you still execute with `streamlit run app.py` locally or deploy
  to **Streamlit Community Cloud** for the judge-facing demo link —
  Antigravity itself isn't a hosting target.
- Keep `max_sensors=60` and `local_epochs=1` for your live demo so each
  round finishes in roughly 1-3 seconds on a laptop CPU; bump both up
  only if you're generating final plots offline for your report.

## 5. Known rough edges (fine for a hackathon, flag in your report)
- `comm_bytes` now reflects the *client's* model size, which varies
  slightly if clients own different numbers of sensors (Non-IID sensor
  chunks aren't perfectly equal-sized) — close enough for the chart.
- The centralized baseline trains for `sel_rounds` epochs once, rather
  than literally matching FL's per-round schedule 1:1 — flagged in the
  comparison tab's math already, just be ready to explain it if asked.