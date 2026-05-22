"""
Bootstrap edge-stability validation for idopNetwork.

Tests whether reconstructed edges are stable under subject-level resampling (internal validity).

Note (do not confuse with 1000 linspace evaluation points during network build):
  - Linspace 1000 points: power-law fit on the current sample, then 1000 evenly spaced pseudo-time
    points for regression; one curve, not 1000 subjects. All n rows inform that curve.
  - Bootstrap: resample subjects (rows) with replacement -> new curves -> new networks. Tests
    robustness to sample composition.

Usage:
  python bootstrap_edge_stability.py

Outputs:
  - results/bootstrap_edge_frequency.csv: directed-edge occurrence frequency
  - results/bootstrap_stable_network.csv: adjacency keeping edges above stability threshold
  - results/bootstrap_summary.txt: summary stats (full-data edges, stable edges, Jaccard, etc.)
"""

import os
import numpy as np
import pandas as pd
import clinical_data
import idop_reconstruction

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_SCRIPT_DIR, "data", "RUC_bins")
DATA_PATH = os.path.join(_DATA_DIR, "Normal_weight.csv")
N_BOOTSTRAP = 50
SAMPLE_FRAC = 0.8
STABLE_THRESHOLD = 0.5
N_SAMPLES = 1000
N_FEATURES = None
SCALER_TYPE = "MinMaxScaler"
BASIS_MAX_ORDER = 3

def directed_edge_set_from_adjacency(W, exclude_diag=True, thresh=1e-10):
    """Return set of nonzero directed edges (i, j) from adjacency W (for Jaccard)."""
    n = W.shape[0]
    edges = set()
    for i in range(n):
        for j in range(n):
            if exclude_diag and i == j:
                continue
            if not np.isnan(W[i, j]) and abs(W[i, j]) > thresh:
                edges.add((i, j))
    return edges

def run_edge_bootstrap():
    os.makedirs("results", exist_ok=True)

    print("Loading data...")
    data = clinical_data.load_clinical_matrix(
        DATA_PATH,
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        scaler_type=SCALER_TYPE,
    )
    n, p = data.shape
    print(f"Data shape: {n} rows x {p} columns")

    print("\n========== Full-data network reconstruction ==========")
    _, _, W_full, _ = idop_reconstruction.reconstruct_idop_network(
        data, basis_max_order=BASIS_MAX_ORDER
    )
    edge_set_full = directed_edge_set_from_adjacency(W_full)
    n_edges_full = len(edge_set_full)
    print(f"Full-data nonzero edges: {n_edges_full}")

    print(f"\n========== Bootstrap (B={N_BOOTSTRAP}, frac={SAMPLE_FRAC}) ==========")
    n_draw = max(1, int(n * SAMPLE_FRAC))
    edge_counts = np.zeros((p, p))
    W_sum = np.zeros((p, p))
    W_sum_sq = np.zeros((p, p))

    for b in range(N_BOOTSTRAP):
        np.random.seed(b)
        idx = np.random.choice(n, size=n_draw, replace=True)
        data_b = data.iloc[idx].copy()
        data_b.index = range(len(data_b))
        try:
            _, _, W_b, _ = idop_reconstruction.reconstruct_idop_network(
                data_b, basis_max_order=BASIS_MAX_ORDER
            )
            edge_counts += (np.abs(W_b) > 1e-10).astype(float)
            W_sum += W_b
            W_sum_sq += W_b ** 2
        except Exception as e:
            print(f"  Bootstrap {b+1} failed: {e}")
            continue
        if (b + 1) % 10 == 0 or b == 0:
            print(f"  Completed {b+1}/{N_BOOTSTRAP}")

    freq = edge_counts / N_BOOTSTRAP
    mean_W = W_sum / N_BOOTSTRAP
    var_W = (W_sum_sq / N_BOOTSTRAP) - (mean_W ** 2)
    var_W = np.maximum(var_W, 0)

    stable_mask = (freq >= STABLE_THRESHOLD) & (np.arange(p)[None, :] != np.arange(p)[:, None])
    stable_edges = set()
    for i in range(p):
        for j in range(p):
            if i != j and stable_mask[i, j]:
                stable_edges.add((i, j))

    jaccard = len(edge_set_full & stable_edges) / len(edge_set_full | stable_edges) if (edge_set_full or stable_edges) else 0.0

    rows = []
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            rows.append({
                "source": i,
                "target": j,
                "frequency": round(float(freq[i, j]), 4),
                "mean_weight": round(float(mean_W[i, j]), 6),
                "std_weight": round(float(np.sqrt(var_W[i, j])), 6),
                "stable": freq[i, j] >= STABLE_THRESHOLD,
            })
    freq_df = pd.DataFrame(rows)
    freq_df.to_csv("results/bootstrap_edge_frequency.csv", index=False, encoding="utf-8-sig")
    print(f"\nEdge frequencies saved: results/bootstrap_edge_frequency.csv")

    W_stable = np.zeros_like(W_full)
    for i in range(p):
        for j in range(p):
            if (i, j) in stable_edges:
                W_stable[i, j] = mean_W[i, j]
    np.fill_diagonal(W_stable, np.diag(mean_W))
    pd.DataFrame(W_stable).to_csv("results/bootstrap_stable_network.csv", index=False, encoding="utf-8-sig")
    print(f"Stable network matrix saved: results/bootstrap_stable_network.csv (threshold={STABLE_THRESHOLD})")

    summary = [
        "========== Bootstrap stability summary ==========",
        f"Data: {DATA_PATH}",
        f"Bootstrap runs: {N_BOOTSTRAP}, sample size per run: {n_draw}",
        f"Full-data nonzero edges: {n_edges_full}",
        f"Stable edges (frequency>={STABLE_THRESHOLD}): {len(stable_edges)}",
        f"Jaccard similarity (full-data vs stable edges): {jaccard:.4f}",
        "==========================================",
    ]
    summary_text = "\n".join(summary)
    with open("results/bootstrap_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)
    print("\n" + summary_text)

    return W_full, freq, stable_edges, jaccard

if __name__ == "__main__":
    run_edge_bootstrap()
