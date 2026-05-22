"""Load directed network tables from CSV (edge list or square adjacency matrix)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

def _normalize_col(name: str) -> str:
    return str(name).strip().lower()

def _find_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    norm = {_normalize_col(c): c for c in columns}
    for cand in candidates:
        if cand in norm:
            return norm[cand]
    return None

def load_edges_csv(path: str | Path) -> tuple[np.ndarray, list[str]]:
    """
    Edge-list CSV with From/To and Effect, Weight, or w columns.
    Returns (adjacency matrix, vertex labels in row/column order).
    """
    path = Path(path)
    df = pd.read_csv(path, encoding="utf-8-sig")
    cols = list(df.columns)
    from_col = _find_column(cols, ("from", "source", "src"))
    to_col = _find_column(cols, ("to", "target", "dst"))
    w_col = _find_column(cols, ("effect", "weight", "w", "value"))
    if from_col is None or to_col is None:
        raise ValueError(f"Edge CSV must have From/To columns: {path}")
    if w_col is None:
        raise ValueError(f"Edge CSV must have Effect/Weight/w column: {path}")

    verts: list[str] = []
    index: dict[str, int] = {}

    def _idx(label) -> int:
        key = str(label).strip()
        if key not in index:
            index[key] = len(verts)
            verts.append(key)
        return index[key]

    n = 0
    for _, row in df.iterrows():
        i = _idx(row[from_col])
        j = _idx(row[to_col])
        n = max(n, i + 1, j + 1)
    W = np.zeros((n, n), dtype=float)
    for _, row in df.iterrows():
        i = _idx(row[from_col])
        j = _idx(row[to_col])
        W[i, j] += float(row[w_col])
    return W, verts

def load_adjacency_csv(path: str | Path) -> tuple[np.ndarray, list[str]]:
    """
    Square adjacency CSV: first column is row index (node names), remaining columns match.
    """
    path = Path(path)
    df = pd.read_csv(path, index_col=0, encoding="utf-8-sig")
    if df.shape[0] != df.shape[1]:
        raise ValueError(f"Adjacency must be square: {path} shape={df.shape}")
    row_labels = [str(x).strip() for x in df.index.tolist()]
    col_labels = [str(x).strip() for x in df.columns.tolist()]
    if row_labels != col_labels:
        raise ValueError(f"Row/column labels must match for adjacency: {path}")
    W = df.values.astype(float)
    return W, row_labels
