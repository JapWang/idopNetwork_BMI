"""
Network construction entry point (no clustering, no full main pipeline).

1) Edge-list CSV with From, To, and Effect/Weight/w -> directed weighted adjacency (sum weights per (From, To)).
2) Adjacency CSV with row index as node names and matching columns -> square matrix.
3) Otherwise -> sample x variable wide table + idopNetwork reconstruction (same as main step 6).

Usage (from idopNetwork):
  python build_network_from_csv.py --csv data/M3_positive.csv
  python build_network_from_csv.py --csv data/RUC_bins/Normal_weight.csv --out results_net
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd

import clinical_data
import network_io
import idop_reconstruction

_HERE = Path(__file__).resolve().parent

RECONSTRUCTION_HYPERPARAMS = dict(
    basis_max_order=3,
    theta_min=0.2,
    lambda_cross=1e-2,
    lambda_balance=0.05,
    green_max_ratio=1.2,
    sparsify_cross=True,
    support_norm_ratio=None,
    asgl_lambda1_range=[1e-4],
    asgl_alpha_range=[0.5],
    asgl_use_bic=True,
)

RESET_INDEX_FOR_UNIQUE_ROWS = True
N_SAMPLES = 1000

def is_edge_list_csv(path: Path) -> bool:
    h = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
    cols = {str(c).strip().lower() for c in h.columns}
    return "from" in cols and "to" in cols

def variable_catalog_from_labels(verts: list[str]) -> pd.DataFrame:
    """Build variable catalog from node labels when edge/adjacency tables have no clinical headers."""
    return pd.DataFrame(
        {
            "index": range(len(verts)),
            "variable_name": verts,
            "abbreviation": verts,
        }
    )

def export_adjacency_and_edge_list(
    adjusted_matrix: np.ndarray,
    variable_mapping: pd.DataFrame,
    out_dir: Path,
    stem: str,
    *,
    plot: bool,
    self_effect_pure: np.ndarray | None,
) -> None:
    adjacency_df = pd.DataFrame(
        adjusted_matrix,
        index=variable_mapping["abbreviation"],
        columns=variable_mapping["abbreviation"],
    )
    adjacency_df.to_csv(out_dir / f"adjacency_matrix_{stem}.csv", encoding="utf-8-sig")

    edges_rows: list[dict] = []
    n = adjusted_matrix.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            w = adjusted_matrix[i, j]
            if np.isnan(w) or w == 0:
                continue
            edges_rows.append(
                {
                    "From": variable_mapping.loc[i, "abbreviation"],
                    "To": variable_mapping.loc[j, "abbreviation"],
                    "Effect": float(w),
                }
            )
    pd.DataFrame(edges_rows).to_csv(
        out_dir / f"edges_{stem}.csv", index=False, encoding="utf-8-sig"
    )

    color_src = (
        self_effect_pure
        if self_effect_pure is not None
        else np.diag(adjusted_matrix).astype(float)
    )
    if plot:
        idop_reconstruction.plot_directed_adjacency_graph(
            adjusted_matrix,
            save_path=str(out_dir / f"network_matrix_{stem}.png"),
            layout_mode="circle",
            self_effect_for_color=color_src,
        )

def build_from_edge_or_adjacency_table(
    csv_path: Path,
    out_dir: Path,
    *,
    plot: bool,
    mode: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    assert mode in ("edges", "adjacency")
    if mode == "edges":
        W, verts = network_io.load_edges_csv(csv_path)
        print(f"Edge-list network: {csv_path}  vertices={len(verts)}  nonzero edges={(W != 0).sum()}")
    else:
        W, verts = network_io.load_adjacency_csv(csv_path)
        print(f"Adjacency matrix loaded: {csv_path}  order={W.shape[0]}")

    variable_mapping = variable_catalog_from_labels(verts)
    stem = csv_path.stem
    variable_mapping.to_csv(out_dir / "variable_mapping.csv", index=False, encoding="utf-8-sig")
    self_pure = np.diag(W).astype(float).copy()
    export_adjacency_and_edge_list(W, variable_mapping, out_dir, stem, plot=plot, self_effect_pure=self_pure)
    print(f"Network build complete ({mode}), output: {out_dir}")
    return W, variable_mapping

def reconstruct_from_feature_table(
    csv_path: Path,
    out_dir: Path,
    *,
    plot: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    csv_path = csv_path.expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    out_dir = out_dir.expanduser().resolve()
    os.makedirs(out_dir, exist_ok=True)

    if is_edge_list_csv(csv_path):
        return build_from_edge_or_adjacency_table(csv_path, out_dir, plot=plot, mode="edges")

    try:
        W, verts = network_io.load_adjacency_csv(csv_path)
        return build_from_edge_or_adjacency_table(csv_path, out_dir, plot=plot, mode="adjacency")
    except ValueError:
        pass

    data = clinical_data.load_clinical_matrix(
        str(csv_path),
        n_samples=N_SAMPLES,
        n_features=None,
        scaler_type="MinMaxScaler",
        reset_index_for_unique_rows=RESET_INDEX_FOR_UNIQUE_ROWS,
    )
    variable_mapping = clinical_data.build_variable_catalog(data)

    print(f"idop input (feature wide table): {csv_path}  shape={data.shape}")
    models, effects, adjusted_matrix, intercepts = idop_reconstruction.reconstruct_idop_network(
        data,
        **RECONSTRUCTION_HYPERPARAMS,
    )

    variable_mapping.to_csv(out_dir / "variable_mapping.csv", index=False, encoding="utf-8-sig")

    n_time = effects[0].shape[0]
    intercepts_arr = np.asarray(intercepts).ravel()
    self_effect_pure = np.diag(adjusted_matrix) - n_time * intercepts_arr

    stem = csv_path.stem
    export_adjacency_and_edge_list(
        adjusted_matrix,
        variable_mapping,
        out_dir,
        stem,
        plot=plot,
        self_effect_pure=self_effect_pure,
    )

    print(f"Network build complete (idop), output: {out_dir}")
    return adjusted_matrix, variable_mapping

def cli_main() -> None:
    p = argparse.ArgumentParser(description="Build network: edge list / adjacency / feature table + idop")
    p.add_argument(
        "--csv",
        type=str,
        default=str(_HERE / "data" / "M3.csv"),
        help="CSV: From/To edge list, square adjacency, or sample x variable feature table",
    )
    p.add_argument(
        "--out",
        type=str,
        default="results_build_network_from_csv",
        help="Output directory (relative to cwd unless absolute)",
    )
    p.add_argument("--no-plot", action="store_true", help="Do not save network topology figure")
    args = p.parse_args()

    reconstruct_from_feature_table(
        Path(args.csv),
        Path(args.out),
        plot=not args.no_plot,
    )

if __name__ == "__main__":
    cli_main()
