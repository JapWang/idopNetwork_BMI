"""
Compute in/out degree from an adjacency matrix and plot:
- Quadrant scatter plot
- Bidirectional vertical bars: x = variables; y >= 0 out-degree, y <= 0 in-degree (absolute scale)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

SAVE_DIR = "results_degree"
os.makedirs(SAVE_DIR, exist_ok=True)

def read_adj_matrix(file_path: str) -> pd.DataFrame:
    """Load adjacency matrix: first column is row index, header lists column names."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(file_path, index_col=0, encoding="utf-8-sig")
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path, index_col=0)
    else:
        raise ValueError("Only .csv / .xlsx / .xls formats are supported")

    return df

def calc_degree(adj_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Count in/out degree by nonzero directed edges (same as idopNetwork main; diagonal excluded).
    """
    A = adj_matrix.values.astype(float)
    n = A.shape[0]
    nodes = list(adj_matrix.index)
    mask = (
        (np.arange(n)[None, :] != np.arange(n)[:, None])
        & ~np.isnan(A)
        & (A != 0)
    )
    out_deg = mask.sum(axis=1).astype(int)
    in_deg = mask.sum(axis=0).astype(int)
    return pd.DataFrame({"in_degree": in_deg, "out_degree": out_deg}, index=nodes)

def plot_quadrant_scatter(
    df_degree,
    title="In/Out Degree Quadrant Plot",
    save_as="quadrant_scatter.png",
):
    plt.figure(figsize=(10, 8))

    x = df_degree["in_degree"]
    y = df_degree["out_degree"]
    labels = df_degree.index

    plt.scatter(
        x,
        y,
        s=200,
        c="#E63946",
        alpha=0.7,
        edgecolors="black",
        linewidth=1.5,
        zorder=3,
    )

    for xi, yi, lab in zip(x, y, labels):
        plt.text(xi + 0.1, yi + 0.1, lab, fontsize=11, fontweight="bold")

    x_mid = np.mean(x)
    y_mid = np.mean(y)
    plt.axvline(x=x_mid, color="gray", linestyle="--", alpha=0.7)
    plt.axhline(y=y_mid, color="gray", linestyle="--", alpha=0.7)

    plt.text(x.max() * 0.8, y.max() * 0.9, "High In\nHigh Out", fontsize=10, ha="center")
    plt.text(
        x.min() * 0.2,
        y.max() * 0.9,
        "Low In\nHigh Out\n(Source Hub)",
        fontsize=10,
        ha="center",
        color="#c92c6d",
        fontweight="bold",
    )
    plt.text(
        x.max() * 0.8,
        y.min() * 0.2,
        "High In\nLow Out\n(Sink Hub)",
        fontsize=10,
        ha="center",
        color="#0070c0",
        fontweight="bold",
    )
    plt.text(x.min() * 0.2, y.min() * 0.2, "Low In\nLow Out", fontsize=10, ha="center")

    plt.xlabel("In-degree", fontsize=13, fontweight="bold")
    plt.ylabel("Out-degree", fontsize=13, fontweight="bold")
    plt.title(title, fontsize=15, pad=15, fontweight="bold")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, save_as), dpi=300, bbox_inches="tight")
    plt.close()

def plot_vertical_grouped_degree(
    df_degree,
    title="In- / Out-Degree by Node",
    save_as="grouped_vertical_degree.png",
):
    """
    x = variables (compact spacing); y split at 0: out above, in below; asymmetric margins by each max degree.
    One bar per variable: out up, in down, shared x center.
    """
    df = df_degree.copy()
    df["total"] = df["in_degree"] + df["out_degree"]
    df = df.sort_values("total", ascending=True)

    labels = [str(i) for i in df.index]
    n = len(df)
    x_scale = 0.4
    x = np.arange(n, dtype=float) * x_scale
    in_vals = df["in_degree"].values.astype(float)
    out_vals = df["out_degree"].values.astype(float)

    w_bar = min(0.34, 0.17 + 0.21 / max(n, 5), x_scale * 0.86)

    fig_w = max(5.6, n * 0.22)
    fig_h = max(6.8, 4.6 + n * 0.04)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    bars_in = ax.bar(
        x,
        -in_vals,
        w_bar,
        align="center",
        label="In-degree (↓)",
        color="#0070C0",
        alpha=0.88,
        edgecolor="black",
        linewidth=0.55,
        zorder=2,
    )
    bars_out = ax.bar(
        x,
        out_vals,
        w_bar,
        align="center",
        label="Out-degree (↑)",
        color="#E63946",
        alpha=0.88,
        edgecolor="black",
        linewidth=0.55,
        zorder=3,
    )

    ax.axhline(0, color="black", linewidth=1.0)
    out_max = float(np.nanmax(out_vals)) if len(out_vals) else 0.0
    in_max = float(np.nanmax(in_vals)) if len(in_vals) else 0.0
    y_top = max(out_max * 1.14, 0.8)
    y_bot = -max(in_max * 1.14, 0.8)
    span = y_top - y_bot
    y_bot -= span * 0.05
    ax.set_ylim(y_bot, y_top)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{int(abs(v))}"))
    ax.set_ylabel("Degree (out ↑ / in ↓)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Variable", fontsize=12, fontweight="bold", labelpad=14)
    ax.set_title(title, fontsize=14, pad=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
    ax.tick_params(axis="x", which="major", pad=16, length=4)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    x_right = max(n - 1, 0) * x_scale
    margin = max(w_bar * 0.52, 0.10 * x_scale)
    ax.set_xlim(-margin, x_right + margin)
    ax.margins(x=0.005)

    if n <= 28:
        pad_top = y_top * 0.035
        pad_bot = abs(y_bot) * 0.035
        for rect in bars_out:
            h = rect.get_height()
            if h <= 0:
                continue
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                min(h + pad_top, y_top * 0.995),
                f"{int(h)}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
        for rect in bars_in:
            h = rect.get_height()
            if h >= 0:
                continue
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                max(h - pad_bot, y_bot * 0.995),
                f"{int(-h)}",
                ha="center",
                va="top",
                fontsize=7,
            )

    plt.tight_layout(pad=1.0, rect=(0.02, 0.14, 0.98, 0.96))
    plt.savefig(os.path.join(SAVE_DIR, save_as), dpi=300, bbox_inches="tight")
    plt.close()

def run_from_file(file_path: str, title_prefix=""):
    print("Loading adjacency matrix:", file_path)
    adj = read_adj_matrix(file_path)

    print("Computing in/out degree...")
    df_deg = calc_degree(adj)

    print("Plotting quadrant scatter...")
    plot_quadrant_scatter(df_deg, title=f"{title_prefix} Quadrant Scatter")

    print("Plotting bidirectional vertical bars (out up, in down)...")
    plot_vertical_grouped_degree(df_deg, title=f"{title_prefix} In/Out Degree")

    print(f"\nDone. Figures saved to: {os.path.abspath(SAVE_DIR)}")
    return df_deg

if __name__ == "__main__":
    run_from_file(
        file_path=r"D:\idopNetwork工作\体检数据项目\构网结果\Cytoscape数据\正常(稀疏度阈值1e-4)\adjacency_matrix_Normal.csv",
        title_prefix="Normal",
    )
