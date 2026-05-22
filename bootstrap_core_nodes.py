"""
Bootstrap core-node stability analysis.

For each group, repeat B times: sample ~80% of rows without replacement, sort row indices
to preserve ALI pseudo-time order, then rebuild idopNetwork. Count how often each node
ranks in the top top_k by weighted out-degree, in-degree, or total strength (node strength).
Nodes with frequency >= stable_threshold are stable core nodes.

Outputs: frequency tables, heatmaps, bar plots, rank distributions, etc.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from scipy.stats import rankdata

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

import idop_reconstruction

def weighted_degrees_from_adjacency(W):
    """
    Weighted out-degree, in-degree, and total degree (node strength) from adjacency W.
    Uses sum of absolute weights to avoid near-tied ranks when W!=0 for most pairs.
    """
    W = np.asarray(W, dtype=float)
    n = W.shape[0]

    mask = (np.arange(n)[None, :] != np.arange(n)[:, None]) & ~np.isnan(W)
    abs_W = np.abs(W) * mask

    out_deg = np.sum(abs_W, axis=1)
    in_deg = np.sum(abs_W, axis=0)
    total_deg = out_deg + in_deg

    return out_deg, in_deg, total_deg

def rank_descending(x):
    """Descending rank: larger values get smaller ranks, starting at 1; ties use minimum rank."""
    return rankdata(-np.asarray(x, dtype=float), method="min").astype(int)

def edge_presence_mask(W, thresh: float = 1e-10) -> np.ndarray:
    """Same as bootstrap_edge_stability: |W_ij|>thresh counts as edge present in this bootstrap."""
    W = np.asarray(W, dtype=float)
    return (np.abs(W) > thresh) & np.isfinite(W)

def plot_edge_bootstrap_heatmap(
    freq: np.ndarray,
    abbrevs: list,
    save_path: str,
    *,
    title: str,
    star_threshold_pct: float = 95.0,
    figsize: tuple[float, float] = (12, 10),
    dpi: int = 150,
) -> None:
    """
    Directed-edge bootstrap frequency heatmap (publication style): gray diagonal (self-loops excluded);
    mark high-stability cells with *.
    freq shape (p, p), values 0–1; rows = source, columns = target.
    """
    freq = np.asarray(freq, dtype=float)
    p = freq.shape[0]
    freq_pct = freq * 100.0
    freq_plot = freq_pct.astype(float).copy()
    np.fill_diagonal(freq_plot, np.nan)

    off_diag = ~np.eye(p, dtype=bool)
    annot_matrix = np.where(
        off_diag & (freq_pct >= star_threshold_pct),
        "*",
        "",
    )

    try:
        cmap_obj = sns.color_palette("rocket_r", as_cmap=True)
    except Exception:
        cmap_obj = plt.get_cmap("Reds")
    if hasattr(cmap_obj, "copy"):
        cmap_obj = cmap_obj.copy()
    cmap_obj.set_bad("#E8E8E8")

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        freq_plot,
        cmap=cmap_obj,
        vmin=0,
        vmax=100,
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.75, "label": "Bootstrap frequency, %"},
        xticklabels=abbrevs,
        yticklabels=abbrevs,
        annot=annot_matrix,
        fmt="",
        annot_kws={"size": 14, "color": "black", "weight": "bold"},
        ax=ax,
    )
    ax.set_ylabel("Source node", fontsize=12, fontweight="bold")
    ax.set_xlabel("Target node", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, pad=20)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def run_core_node_bootstrap(
    data_list,
    group_names,
    variable_mapping,
    results_dir="results_bootstrap",
    B=1000,
    top_k=5,
    stable_threshold=0.95,
    basis_max_order=3,
    verbose=True,
    output_tag=None,
):
    """
    Run B bootstrap replicates per group (~80% rows, sorted indices), rebuild networks, and
    count top-k frequency by weighted out/in/total degree. Nodes with frequency >= stable_threshold
    are stable core nodes.

    Parameters
    ----------
    data_list : list of pd.DataFrame
        One DataFrame per group; rows = samples, columns = variables.
    group_names : list of str
        Group labels (e.g. Obesity, Normal), aligned with data_list.
    variable_mapping : pd.DataFrame
        Columns: index, variable_name, abbreviation.
    results_dir : str
        Output directory for tables and figures.
    output_tag : str or None
        If set, append _{output_tag} before file extensions for fixed output names.
    B : int
        Number of bootstrap draws (~80% rows, indices sorted for pseudo-time order).
    top_k : int
        Core definition: rank <= top_k in that metric.
    stable_threshold : float
        Stable if in top top_k in >= this fraction of bootstrap networks.
    basis_max_order : int
        Legendre basis order (same as main).
    verbose : bool
        Print progress.

    Returns
    -------
    summary : dict
        frequency_table, stable_nodes, rank_store, etc.
    """
    os.makedirs(results_dir, exist_ok=True)

    def _tagged(filename):
        if not output_tag:
            return os.path.join(results_dir, filename)
        stem, ext = os.path.splitext(filename)
        return os.path.join(results_dir, f"{stem}_{output_tag}{ext}")

    n_groups = len(data_list)
    p = variable_mapping.shape[0]
    abbrevs = variable_mapping["abbreviation"].tolist()

    in_topk_out = np.zeros((n_groups, B, p), dtype=float)
    in_topk_in = np.zeros((n_groups, B, p), dtype=float)
    in_topk_total = np.zeros((n_groups, B, p), dtype=float)
    rank_out_store = np.full((n_groups, B, p), np.nan)
    rank_in_store = np.full((n_groups, B, p), np.nan)
    rank_total_store = np.full((n_groups, B, p), np.nan)
    edge_counts = np.zeros((n_groups, p, p), dtype=float)

    full_net_metrics = []
    for g in range(n_groups):
        data = data_list[g]
        n = data.shape[0]
        if verbose:
            _n_draw = max(1, min(n, int(n * 0.8)))
            print(
                f"\nGroup {group_names[g]}: n={n}, p={p}, Bootstrap {B} runs "
                f"(~{_n_draw} rows each, 80% without replacement + index sort)"
            )
        try:
            _, _, W_full, _ = idop_reconstruction.reconstruct_idop_network(
                data, basis_max_order=basis_max_order
            )
            o, i, t = weighted_degrees_from_adjacency(W_full)
            full_net_metrics.append({"out": o, "in": i, "total": t})
        except Exception as e:
            if verbose:
                print(f"  Full-data network failed: {e}")
            full_net_metrics.append({"out": np.zeros(p), "in": np.zeros(p), "total": np.zeros(p)})

    for g in range(n_groups):
        data = data_list[g]
        n = data.shape[0]
        for b in range(B):
            np.random.seed(g * (B + 1) + b)

            n_draw = max(1, min(n, int(n * 0.8)))
            idx = np.random.choice(n, size=n_draw, replace=False)
            idx = np.sort(idx)

            data_b = data.iloc[idx].copy()
            data_b.index = range(len(data_b))
            if verbose:
                print(f"  Group {group_names[g]}: bootstrap {b+1}/{B}")
            try:
                _, _, W_b, _ = idop_reconstruction.reconstruct_idop_network(
                    data_b, basis_max_order=basis_max_order
                )
                out_deg, in_deg, total_deg = weighted_degrees_from_adjacency(W_b)
                r_out = rank_descending(out_deg)
                r_in = rank_descending(in_deg)
                r_total = rank_descending(total_deg)
                in_topk_out[g, b, :] = (r_out <= top_k).astype(float)
                in_topk_in[g, b, :] = (r_in <= top_k).astype(float)
                in_topk_total[g, b, :] = (r_total <= top_k).astype(float)
                rank_out_store[g, b, :] = r_out
                rank_in_store[g, b, :] = r_in
                rank_total_store[g, b, :] = r_total
                edge_counts[g] += edge_presence_mask(W_b).astype(float)
            except Exception as e:
                if verbose and b == 0:
                    print(f"  Bootstrap failed (b=0): {e}")
                continue
        if verbose and (g + 1) % 1 == 0:
            print(f"  Finished group {group_names[g]}")

    n_valid = np.sum(~np.isnan(rank_out_store), axis=1)
    n_valid = np.maximum(n_valid, 1)
    freq_out = np.nansum(in_topk_out, axis=1) / n_valid
    freq_in = np.nansum(in_topk_in, axis=1) / n_valid
    freq_total = np.nansum(in_topk_total, axis=1) / n_valid

    stable_out = freq_out >= stable_threshold
    stable_in = freq_in >= stable_threshold
    stable_total = freq_total >= stable_threshold

    rows = []
    for g in range(n_groups):
        for i in range(p):
            rows.append({
                "Group": group_names[g],
                "Node_index": int(i),
                "Abbrev": abbrevs[i],
                f"Freq_in_top{top_k}_Out": round(float(freq_out[g, i]), 4),
                f"Freq_in_top{top_k}_In": round(float(freq_in[g, i]), 4),
                f"Freq_in_top{top_k}_Total": round(float(freq_total[g, i]), 4),
                "Pct_Out": round(float(freq_out[g, i]) * 100, 2),
                "Pct_In": round(float(freq_in[g, i]) * 100, 2),
                "Pct_Total": round(float(freq_total[g, i]) * 100, 2),
                "Stable_core_Out": stable_out[g, i],
                "Stable_core_In": stable_in[g, i],
                "Stable_core_Total": stable_total[g, i],
            })
    frequency_table = pd.DataFrame(rows)
    _freq_csv = _tagged("bootstrap_core_node_frequency.csv")
    frequency_table.to_csv(
        _freq_csv,
        index=False,
        encoding="utf-8-sig",
    )
    if verbose:
        print(f"\nFrequency table saved: {_freq_csv}")

    stable_rows = []
    for g in range(n_groups):
        for metric, stable_mat, name in [
            (stable_out, stable_out, "Out"),
            (stable_in, stable_in, "In"),
            (stable_total, stable_total, "Total"),
        ]:
            nodes = [abbrevs[i] for i in range(p) if stable_mat[g, i]]
            stable_rows.append({
                "Group": group_names[g],
                "Metric": name,
                "Stable_core_nodes": ", ".join(nodes) if nodes else "-",
                "Count": len(nodes),
            })
    stable_summary_df = pd.DataFrame(stable_rows)
    stable_summary_df.to_csv(
        _tagged("bootstrap_stable_core_nodes_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    heatmap_data = np.zeros((p, n_groups * 3))
    col_labels = []
    for g in range(n_groups):
        heatmap_data[:, g * 3 + 0] = freq_out[g, :] * 100
        heatmap_data[:, g * 3 + 1] = freq_in[g, :] * 100
        heatmap_data[:, g * 3 + 2] = freq_total[g, :] * 100
        col_labels.extend([f"{group_names[g]}\nOut", f"{group_names[g]}\nIn", f"{group_names[g]}\nTotal"])
    fig, ax = plt.subplots(figsize=(4 * n_groups, max(6, p * 0.35)))
    im = ax.imshow(heatmap_data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
    ax.set_xticks(np.arange(n_groups * 3))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(np.arange(p))
    ax.set_yticklabels(abbrevs, fontsize=9)
    plt.colorbar(im, ax=ax, label=f"% of bootstrap runs in top {top_k}")
    ax.set_title(f"Core node stability: % of runs in top-{top_k} by weighted degree (B={B})")
    plt.tight_layout()
    _heat_png = _tagged("bootstrap_core_node_stability_heatmap.png")
    fig.savefig(
        _heat_png,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)
    if verbose:
        print(f"Heatmap saved: {_heat_png}")

    for g in range(n_groups):
        fig, ax = plt.subplots(figsize=(max(8, p * 0.5), 5))
        x = np.arange(p)
        w = 0.25
        ax.bar(x - w, freq_out[g, :] * 100, width=w, label="Out strength", color="#2E86C1")
        ax.bar(x, freq_in[g, :] * 100, width=w, label="In strength", color="#28A745")
        ax.bar(x + w, freq_total[g, :] * 100, width=w, label="Total strength", color="#D35400")
        _thr_pct = stable_threshold * 100.0
        ax.axhline(
            y=_thr_pct,
            color="gray",
            linestyle="--",
            linewidth=1,
            label=f"{_thr_pct:.0f}% threshold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(abbrevs, rotation=45, ha="right")
        ax.set_ylabel(f"% of bootstrap runs in top {top_k}")
        ax.set_title(f"Core node stability — {group_names[g]} (B={B}, top {top_k})")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_ylim(0, 105)
        plt.tight_layout()
        fig.savefig(
            os.path.join(results_dir, f"bootstrap_core_node_bar_{group_names[g]}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)
    if verbose:
        print(f"Bar plots saved: {results_dir}/bootstrap_core_node_bar_*.png")

    fig, axes = plt.subplots(n_groups, 3, figsize=(12, 4 * n_groups))
    if n_groups == 1:
        axes = axes.reshape(1, -1)
    for g in range(n_groups):
        o, i, t = full_net_metrics[g]["out"], full_net_metrics[g]["in"], full_net_metrics[g]["total"]
        top1_out = int(np.argmax(o))
        top1_in = int(np.argmax(i))
        top1_total = int(np.argmax(t))
        for m, (top1_idx, metric_name) in enumerate([
            (top1_out, "Out strength"),
            (top1_in, "In strength"),
            (top1_total, "Total strength"),
        ]):
            ax = axes[g, m]
            ranks = rank_out_store[g, :, top1_idx] if m == 0 else (rank_in_store[g, :, top1_idx] if m == 1 else rank_total_store[g, :, top1_idx])
            ranks = ranks[~np.isnan(ranks)].astype(int)
            if len(ranks) > 0:
                ax.hist(ranks, bins=np.arange(0.5, p + 2, 1), color=["#2E86C1", "#28A745", "#D35400"][m], edgecolor="white")
                ax.axvline(x=1, color="red", linestyle="--", linewidth=1.5, label="Rank 1")
                ax.set_xlabel("Rank in bootstrap run")
            ax.set_ylabel("Count")
            ax.set_title(f"{group_names[g]} — Top1 {metric_name}\n({abbrevs[top1_idx]})")
            ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    _rank_png = _tagged("bootstrap_rank_distribution_top1.png")
    fig.savefig(
        _rank_png,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)
    if verbose:
        print(f"Rank distribution plot saved: {_rank_png}")

    fig, ax = plt.subplots(figsize=(10, 1 + n_groups * 1.2))
    ax.axis("off")
    cell_text = []
    for g in range(n_groups):
        for metric_name, stable_mat in [("Out", stable_out), ("In", stable_in), ("Total", stable_total)]:
            nodes = [abbrevs[i] for i in range(p) if stable_mat[g, i]]
            cell_text.append([group_names[g], metric_name, ", ".join(nodes) if nodes else "—", len(nodes)])
    _pct = int(round(stable_threshold * 100))
    table = ax.table(
        colLabels=[
            "Group",
            "Metric",
            f"Stable core nodes (≥{_pct}% in top {top_k})",
            "Count",
        ],
        cellText=cell_text,
        loc="center",
        cellLoc="left",
        colColours=["#E8E8E8"] * 4,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2)
    ax.set_title(
        f"Bootstrap core node stability summary (B={B}, top-{top_k}, ≥{_pct}% criterion)",
        fontsize=12,
    )
    plt.tight_layout()
    _sum_png = _tagged("bootstrap_stable_core_summary_figure.png")
    fig.savefig(
        _sum_png,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    jaccard_results = []
    for g in range(n_groups):
        o, i, t = full_net_metrics[g]["out"], full_net_metrics[g]["in"], full_net_metrics[g]["total"]
        topk_out_full = set(np.argsort(-o)[:top_k])
        topk_in_full = set(np.argsort(-i)[:top_k])
        topk_total_full = set(np.argsort(-t)[:top_k])
        for metric_name, topk_full in [
            ("Out", topk_out_full),
            ("In", topk_in_full),
            ("Total", topk_total_full),
        ]:
            jaccards = []
            for b in range(B):
                if metric_name == "Out":
                    r = rank_out_store[g, b, :]
                elif metric_name == "In":
                    r = rank_in_store[g, b, :]
                else:
                    r = rank_total_store[g, b, :]
                if np.all(np.isnan(r)):
                    continue
                topk_b = set(np.where(r <= top_k)[0])
                if len(topk_full | topk_b) == 0:
                    jaccards.append(1.0)
                else:
                    jaccards.append(len(topk_full & topk_b) / len(topk_full | topk_b))
            jaccard_results.append({
                "Group": group_names[g],
                "Metric": metric_name,
                "Mean_Jaccard": np.mean(jaccards) if jaccards else np.nan,
                "Std_Jaccard": np.std(jaccards) if jaccards else np.nan,
            })
    jaccard_df = pd.DataFrame(jaccard_results)
    _jac_stem = f"bootstrap_top{top_k}_jaccard_consistency"
    _jac_csv = _tagged(f"{_jac_stem}.csv")
    jaccard_df.to_csv(_jac_csv, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(jaccard_df))
    bars = ax.bar(x, jaccard_df["Mean_Jaccard"] * 100, yerr=jaccard_df["Std_Jaccard"] * 100, capsize=3, color="#3498DB", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.Group}\n{r.Metric}" for _, r in jaccard_df.iterrows()], fontsize=9)
    ax.set_ylabel(f"Mean Jaccard (top-{top_k} vs full-data top-{top_k}), %")
    ax.set_title(
        f"Consistency of top-{top_k} node set across Bootstrap (higher = more reproducible)"
    )
    ax.set_ylim(0, 105)
    ax.axhline(y=stable_threshold * 100, color="gray", linestyle="--", linewidth=1)
    plt.tight_layout()
    _jac_png = _tagged(f"{_jac_stem}.png")
    fig.savefig(_jac_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print(f"Top-{top_k} Jaccard consistency plot saved: {_jac_png}")

    n_succ = np.array(
        [int(np.sum(np.isfinite(rank_out_store[g, :, 0]))) for g in range(n_groups)],
        dtype=int,
    )
    freq_edge = np.zeros((n_groups, p, p), dtype=float)
    for g in range(n_groups):
        denom = max(int(n_succ[g]), 1)
        freq_edge[g] = edge_counts[g] / float(denom)
    edge_heatmap_paths = []
    _star_pct = float(stable_threshold) * 100.0
    for g in range(n_groups):
        _edge_png = _tagged(f"bootstrap_directed_edge_stability_matrix_{group_names[g]}.png")
        plot_edge_bootstrap_heatmap(
            freq_edge[g],
            abbrevs,
            _edge_png,
            title=f"Directed edge stability matrix — {group_names[g]}, B={B}",
            star_threshold_pct=_star_pct,
        )
        edge_heatmap_paths.append(_edge_png)
        if verbose:
            print(f"Directed edge stability heatmap saved: {_edge_png}")

    summary = {
        "frequency_table": frequency_table,
        "stable_summary_df": stable_summary_df,
        "jaccard_df": jaccard_df,
        "stable_out": stable_out,
        "stable_in": stable_in,
        "stable_total": stable_total,
        "freq_out": freq_out,
        "freq_in": freq_in,
        "freq_total": freq_total,
        "rank_out_store": rank_out_store,
        "rank_in_store": rank_in_store,
        "rank_total_store": rank_total_store,
        "full_net_metrics": full_net_metrics,
        "freq_edge": freq_edge,
        "edge_stability_heatmap_paths": edge_heatmap_paths,
        "n_bootstrap_success_per_group": n_succ,
    }
    return summary
