"""
In/out degree and radial network plots: plot_degrees_from_adjacency draws horizontal bars;
plot_radial_out_degree_network BFS radial layout centered on max out-degree node;
plot_radial_in_degree_network radial layout centered on max in-degree node.

Requires networkx, matplotlib, pandas, numpy.
Default output: results/ or caller-provided output_dir.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent

def _default_output_dir(is_normal: bool | None) -> str:
    """Same as main: prefer results/; keep is_normal branch for backward compatibility."""
    if is_normal is False:
        return str(_SCRIPT_DIR / "results")
    return str(_SCRIPT_DIR / "results")

def plot_degrees_from_adjacency(
    df: pd.DataFrame,
    title: str = "In- and Out-Degree Distribution",
    save_name: str = "in_out_degree_comparison.png",
    sort: bool = True,
    is_normal: bool | None = None,
    output_dir: str | None = None,
):
    """
    Publication-quality In- / Out-degree visualization
    (diverging horizontal bar chart with annotations)
    """
    required_cols = {"in_degree", "out_degree"}
    if not required_cols.issubset(df.columns):
        raise ValueError("DataFrame must contain 'in_degree' and 'out_degree'")

    df_plot = df.copy()

    if "Module" not in df_plot.columns:
        df_plot = df_plot.reset_index()
        df_plot.rename(columns={df_plot.columns[0]: "Module"}, inplace=True)

    df_plot["in_degree"] = df_plot["in_degree"].astype(int)
    df_plot["out_degree"] = df_plot["out_degree"].astype(int)

    if sort:
        df_plot["total_degree"] = df_plot["in_degree"] + df_plot["out_degree"]
        df_plot = df_plot.sort_values("total_degree", ascending=True)

    modules = df_plot["Module"].values
    out_degree = -df_plot["out_degree"].values
    in_degree = df_plot["in_degree"].values
    y = np.arange(len(modules))

    out_color = "#4C72B0"
    in_color = "#C44E52"

    fig, ax = plt.subplots(figsize=(10, max(6, len(modules) * 0.35)))

    bars_out = ax.barh(
        y,
        out_degree,
        color=out_color,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.6,
        label="Out-degree",
    )

    bars_in = ax.barh(
        y,
        in_degree,
        color=in_color,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.6,
        label="In-degree",
    )

    ax.axvline(0, color="black", linewidth=1)

    ax.set_yticks(y)
    ax.set_yticklabels(modules, fontsize=11)
    ax.set_xlabel("Node degree", fontsize=12)
    ax.set_ylabel("Modules", fontsize=12)
    ax.set_title(title, fontsize=14, pad=12)

    max_degree = max(df_plot["in_degree"].max(), df_plot["out_degree"].max())
    ax.set_xlim(-max_degree - 2, max_degree + 2)

    offset = max_degree * 0.04 + 0.2

    for bar in bars_out:
        width = bar.get_width()
        if width != 0:
            ax.text(
                width - offset,
                bar.get_y() + bar.get_height() / 2,
                f"{abs(int(width))}",
                va="center",
                ha="right",
                fontsize=9,
            )

    for bar in bars_in:
        width = bar.get_width()
        if width != 0:
            ax.text(
                width + offset,
                bar.get_y() + bar.get_height() / 2,
                f"{int(width)}",
                va="center",
                ha="left",
                fontsize=9,
            )

    ax.grid(axis="x", linestyle="--", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=10)

    plt.tight_layout()

    result_dir = output_dir if output_dir is not None else _default_output_dir(is_normal)
    os.makedirs(result_dir, exist_ok=True)
    save_path = os.path.join(result_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def _nonzero_weight(w) -> bool:
    try:
        if w is None or (isinstance(w, float) and np.isnan(w)):
            return False
    except TypeError:
        pass
    return float(w) != 0.0

def plot_radial_out_degree_network(
    adjusted_matrix: pd.DataFrame,
    figsize=(13, 13),
    layer_gap=4.0,
    seed=42,
    save_name="radial_outdegree_network.png",
    output_dir: str | None = None,
    is_normal: bool | None = None,
):
    """
    Radial + BFS directed network centered on max out-degree node.
    adjusted_matrix: row/column node names, edge weights; diagonal may hold self-effects.
    """
    np.random.seed(seed)

    G = nx.DiGraph()
    nodes = list(adjusted_matrix.columns)
    G.add_nodes_from(nodes)

    for i, src in enumerate(adjusted_matrix.index):
        for j, dst in enumerate(adjusted_matrix.columns):
            if i == j:
                continue
            w = adjusted_matrix.iloc[i, j]
            if not _nonzero_weight(w):
                continue
            w = float(w)
            G.add_edge(src, dst, raw_weight=w, sign="pos" if w > 0 else "neg")

    out_degree = dict(G.out_degree())
    center_node = max(out_degree, key=out_degree.get)

    diag = np.diag(adjusted_matrix.values.astype(float))
    abs_diag = np.abs(diag)
    max_diag = abs_diag.max() if abs_diag.max() > 0 else 1

    node_sizes_dict = {}
    node_colors = []

    for n in G.nodes():
        idx = nodes.index(n)
        size = 1800 + 4200 * abs_diag[idx] / max_diag
        if n == center_node:
            size *= 1.25
        node_sizes_dict[n] = size
        node_colors.append("#FFECB3" if diag[idx] >= 0 else "#B3D7FF")

    fig_temp, ax_temp = plt.subplots(figsize=figsize, facecolor="white")
    estimated_range = len(nodes) * layer_gap * 1.5
    ax_temp.set_xlim(-estimated_range, estimated_range)
    ax_temp.set_ylim(-estimated_range, estimated_range)
    data_width = estimated_range * 2
    data_height = estimated_range * 2
    inches_per_data_x = figsize[0] / data_width
    inches_per_data_y = figsize[1] / data_height

    node_radii = {}
    for n, size in node_sizes_dict.items():
        radius_points = np.sqrt(size / np.pi)
        radius_inches = radius_points / 72.0
        radius_data = radius_inches / min(inches_per_data_x, inches_per_data_y)
        node_radii[n] = radius_data

    plt.close(fig_temp)

    layers = {0: [center_node]}
    visited = {center_node}
    frontier = [center_node]
    depth = 1

    while frontier:
        nxt = []
        for u in frontier:
            for v in G.successors(u):
                if v not in visited:
                    visited.add(v)
                    nxt.append(v)
        if not nxt:
            break
        layers[depth] = nxt
        frontier = nxt
        depth += 1

    remaining = list(set(G.nodes()) - visited)
    if remaining:
        layers[depth] = remaining

    pos = {}
    max_layer = max(layers.keys())
    pos[center_node] = (0.0, 0.0)
    center_radius = node_radii[center_node]
    placed_nodes = [(0.0, 0.0, center_radius)]

    def check_overlap(x, y, r, placed_nodes, min_gap=0.3):
        for px, py, pr in placed_nodes:
            dist = np.hypot(x - px, y - py)
            if dist < (r + pr) * (1 + min_gap):
                return True
        return False

    def find_valid_position(target_r, layer_r, angle, placed_nodes, max_attempts=50):
        for attempt in range(max_attempts):
            angle_offset = np.random.uniform(-0.3, 0.3)
            test_angle = angle + angle_offset
            radius_factor = 1.0 + (attempt * 0.05)
            test_r = layer_r * radius_factor
            x = test_r * np.cos(test_angle)
            y = test_r * np.sin(test_angle)
            if not check_overlap(x, y, target_r, placed_nodes):
                return (x, y)
        fallback_r = layer_r * 1.5
        x = fallback_r * np.cos(angle)
        y = fallback_r * np.sin(angle)
        return (x, y)

    def find_valid_position_large_search(target_r, layer_r, angle, placed_nodes, max_attempts=100):
        for attempt in range(max_attempts):
            angle_offset = np.random.uniform(-0.6, 0.6)
            test_angle = angle + angle_offset
            radius_factor = 1.0 + (attempt * 0.08)
            test_r = layer_r * radius_factor
            x = test_r * np.cos(test_angle)
            y = test_r * np.sin(test_angle)
            if not check_overlap(x, y, target_r, placed_nodes, min_gap=0.4):
                return (x, y)
        fallback_r = layer_r * 1.8
        x = fallback_r * np.cos(angle)
        y = fallback_r * np.sin(angle)
        return (x, y)

    layer_radii = {}
    prev_layer_max_r = center_radius

    for layer in range(1, max_layer + 1):
        if layer not in layers:
            continue
        layer_nodes = layers[layer]
        n = len(layer_nodes)
        if n == 0:
            continue
        max_node_r = max(node_radii[n] for n in layer_nodes)
        if layer == 1:
            min_circumference = sum(node_radii[n] * 2 * 2.2 for n in layer_nodes)
            min_radius = min_circumference / (2 * np.pi) if n > 0 else max_node_r * 3
            min_layer_r = prev_layer_max_r + max_node_r * 3.5 + layer_gap * 1.8
        elif layer == 2:
            min_circumference = sum(node_radii[n] * 2 * 1.5 for n in layer_nodes)
            min_radius = min_circumference / (2 * np.pi) if n > 0 else max_node_r * 2
            min_layer_r = prev_layer_max_r + max_node_r * 2.0 + layer_gap * 0.6
        else:
            min_circumference = sum(node_radii[n] * 2 * 1.5 for n in layer_nodes)
            min_radius = min_circumference / (2 * np.pi) if n > 0 else max_node_r * 2
            min_layer_r = prev_layer_max_r + max_node_r * 2.5 + layer_gap
        layer_r = max(min_radius, min_layer_r)
        layer_radii[layer] = layer_r
        prev_layer_max_r = layer_r + max_node_r

    for layer in range(1, max_layer + 1):
        if layer not in layers:
            continue
        layer_nodes = layers[layer]
        n = len(layer_nodes)
        if n == 0:
            continue
        layer_r = layer_radii[layer]
        layer_nodes_sorted = sorted(layer_nodes, key=lambda x: node_radii[x], reverse=True)
        if layer == 1:
            base_angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            angle_spread = 2 * np.pi / n
            base_angles = [
                a + angle_spread * 0.2 * np.sin(i * np.pi / n) for i, a in enumerate(base_angles)
            ]
        else:
            base_angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        for idx, node in enumerate(layer_nodes_sorted):
            node_r = node_radii[node]
            base_angle = base_angles[idx]
            if layer == 1:
                x, y = find_valid_position_large_search(node_r, layer_r, base_angle, placed_nodes)
            else:
                x, y = find_valid_position(node_r, layer_r, base_angle, placed_nodes)
            pos[node] = (x, y)
            placed_nodes.append((x, y, node_r))

    node_sizes = [node_sizes_dict[n] for n in G.nodes()]

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    all_coords = np.array(list(pos.values()))
    if len(all_coords) > 0:
        x_range = all_coords[:, 0].max() - all_coords[:, 0].min()
        y_range = all_coords[:, 1].max() - all_coords[:, 1].min()
        data_range = max(x_range, y_range) * 1.3
        center_x, center_y = all_coords.mean(axis=0)
        ax.set_xlim(center_x - data_range / 2, center_x + data_range / 2)
        ax.set_ylim(center_y - data_range / 2, center_y + data_range / 2)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        data_width = xlim[1] - xlim[0]
        data_height = ylim[1] - ylim[0]
        inches_per_data_x = figsize[0] / data_width
        inches_per_data_y = figsize[1] / data_height
        for n in G.nodes():
            size = node_sizes_dict[n]
            radius_points = np.sqrt(size / np.pi)
            radius_inches = radius_points / 72.0
            radius_data = radius_inches / min(inches_per_data_x, inches_per_data_y)
            node_radii[n] = radius_data

    all_w = np.array([abs(G[u][v]["raw_weight"]) for u, v in G.edges()])
    max_w = all_w.max() if len(all_w) else 1

    def edge_width(w):
        return 1.8 + 7.0 * abs(w) / max_w

    pos_edges = [(u, v) for u, v in G.edges() if G[u][v]["raw_weight"] > 0]
    neg_edges = [(u, v) for u, v in G.edges() if G[u][v]["raw_weight"] < 0]

    def draw_edge_with_arrow(u, v, color):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        dx, dy = x1 - x0, y1 - y0
        d = np.hypot(dx, dy)
        if d == 0:
            return
        ux, uy = dx / d, dy / d
        r0 = node_radii[u]
        r1 = node_radii[v]
        start_offset = r0 * 1.05
        start = (x0 + ux * start_offset, y0 + uy * start_offset)
        if d < (r0 + r1) * 2.5:
            end_offset = r1 * 1.2
        else:
            end_offset = r1 * 1.12
        end = (x1 - ux * end_offset, y1 - uy * end_offset)
        edge_length = np.hypot(end[0] - start[0], end[1] - start[1])
        min_length = max(r0, r1) * 0.1
        if edge_length < min_length:
            return
        edge_length_normalized = min(edge_length / (max(r0, r1) * 3), 1.0)
        min_arrow_size, max_arrow_size = 20, 28
        arrow_size = min_arrow_size + (max_arrow_size - min_arrow_size) * max(
            edge_length_normalized, 0.3
        )
        arrow = mpatches.FancyArrowPatch(
            start,
            end,
            arrowstyle="->",
            connectionstyle="arc3,rad=0.15",
            lw=edge_width(G[u][v]["raw_weight"]),
            color=color,
            alpha=0.9,
            mutation_scale=arrow_size,
            mutation_aspect=1.2,
            zorder=1,
            shrinkA=0,
            shrinkB=0,
        )
        ax.add_patch(arrow)

    for u, v in pos_edges:
        draw_edge_with_arrow(u, v, "#FF6B9D")
    for u, v in neg_edges:
        draw_edge_with_arrow(u, v, "#00B4D8")

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="#212121",
        linewidths=1.8,
        ax=ax,
    )

    coords = np.array(list(pos.values()))
    center = coords.mean(axis=0)
    radius = np.max(np.linalg.norm(coords - center, axis=1)) * 1.12
    circle = plt.Circle(
        center,
        radius,
        color="#B0BEC5",
        fill=False,
        linewidth=2.4,
        alpha=0.35,
        zorder=0,
    )
    ax.add_patch(circle)

    nx.draw_networkx_labels(
        G, pos, font_size=13, font_weight="bold", font_color="#333", ax=ax
    )

    ax.set_title(
        f"Radial causal network (center by out-degree: {center_node})",
        fontsize=18,
        pad=20,
    )
    ax.axis("off")
    ax.set_aspect("equal")

    result_dir = output_dir if output_dir is not None else _default_output_dir(is_normal)
    os.makedirs(result_dir, exist_ok=True)
    path = os.path.join(result_dir, save_name)
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    return path

def plot_radial_in_degree_network(
    adjusted_matrix: pd.DataFrame,
    figsize=(13, 13),
    layer_gap=4.5,
    seed=42,
    save_name="radial_indegree_network.png",
    is_normal: bool | None = None,
    output_dir: str | None = None,
):
    """
    Radial + BFS directed network centered on max in-degree node; BFS along predecessors.
    """
    np.random.seed(seed)

    G = nx.DiGraph()
    nodes = list(adjusted_matrix.columns)
    G.add_nodes_from(nodes)

    for i, src in enumerate(adjusted_matrix.index):
        for j, dst in enumerate(adjusted_matrix.columns):
            if i == j:
                continue
            w = adjusted_matrix.iloc[i, j]
            if not _nonzero_weight(w):
                continue
            w = float(w)
            G.add_edge(src, dst, raw_weight=w, sign="pos" if w > 0 else "neg")

    in_degree = dict(G.in_degree())
    center_node = max(in_degree, key=in_degree.get)

    diag = np.diag(adjusted_matrix.values.astype(float))
    abs_diag = np.abs(diag)
    max_diag = abs_diag.max() if abs_diag.max() > 0 else 1

    node_sizes_dict = {}
    node_colors = []

    for n in G.nodes():
        idx = nodes.index(n)
        size = 1800 + 4200 * abs_diag[idx] / max_diag
        if n == center_node:
            size *= 1.25
        node_sizes_dict[n] = size
        node_colors.append("#FFECB3" if diag[idx] >= 0 else "#B3D7FF")

    fig_temp, ax_temp = plt.subplots(figsize=figsize, facecolor="white")
    estimated_range = len(nodes) * layer_gap * 1.5
    ax_temp.set_xlim(-estimated_range, estimated_range)
    ax_temp.set_ylim(-estimated_range, estimated_range)
    data_width = estimated_range * 2
    data_height = estimated_range * 2
    inches_per_data_x = figsize[0] / data_width
    inches_per_data_y = figsize[1] / data_height

    node_radii = {}
    for n, size in node_sizes_dict.items():
        radius_points = np.sqrt(size / np.pi)
        radius_inches = radius_points / 72.0
        radius_data = radius_inches / min(inches_per_data_x, inches_per_data_y)
        node_radii[n] = radius_data

    plt.close(fig_temp)

    layers = {0: [center_node]}
    visited = {center_node}
    frontier = [center_node]
    depth = 1

    while frontier:
        nxt = []
        for u in frontier:
            for v in G.predecessors(u):
                if v not in visited:
                    visited.add(v)
                    nxt.append(v)
        if not nxt:
            break
        layers[depth] = nxt
        frontier = nxt
        depth += 1

    remaining = list(set(G.nodes()) - visited)
    if remaining:
        layers[depth] = remaining

    pos = {}
    max_layer = max(layers.keys())
    pos[center_node] = (0.0, 0.0)
    center_radius = node_radii[center_node]
    placed_nodes = [(0.0, 0.0, center_radius)]

    def check_overlap(x, y, r, placed_nodes, min_gap=0.3):
        for px, py, pr in placed_nodes:
            dist = np.hypot(x - px, y - py)
            if dist < (r + pr) * (1 + min_gap):
                return True
        return False

    def find_valid_position(target_r, layer_r, angle, placed_nodes, max_attempts=50):
        for attempt in range(max_attempts):
            angle_offset = np.random.uniform(-0.3, 0.3)
            test_angle = angle + angle_offset
            radius_factor = 1.0 + (attempt * 0.05)
            test_r = layer_r * radius_factor
            x = test_r * np.cos(test_angle)
            y = test_r * np.sin(test_angle)
            if not check_overlap(x, y, target_r, placed_nodes):
                return (x, y)
        fallback_r = layer_r * 1.5
        x = fallback_r * np.cos(angle)
        y = fallback_r * np.sin(angle)
        return (x, y)

    def find_valid_position_large_search(target_r, layer_r, angle, placed_nodes, max_attempts=100):
        for attempt in range(max_attempts):
            angle_offset = np.random.uniform(-0.6, 0.6)
            test_angle = angle + angle_offset
            radius_factor = 1.0 + (attempt * 0.08)
            test_r = layer_r * radius_factor
            x = test_r * np.cos(test_angle)
            y = test_r * np.sin(test_angle)
            if not check_overlap(x, y, target_r, placed_nodes, min_gap=0.4):
                return (x, y)
        fallback_r = layer_r * 1.8
        x = fallback_r * np.cos(angle)
        y = fallback_r * np.sin(angle)
        return (x, y)

    layer_radii = {}
    prev_layer_max_r = center_radius

    for layer in range(1, max_layer + 1):
        if layer not in layers:
            continue
        layer_nodes = layers[layer]
        n = len(layer_nodes)
        if n == 0:
            continue
        max_node_r = max(node_radii[n] for n in layer_nodes)
        if layer == 1:
            min_circumference = sum(node_radii[n] * 2 * 2.2 for n in layer_nodes)
            min_radius = min_circumference / (2 * np.pi) if n > 0 else max_node_r * 3
            min_layer_r = prev_layer_max_r + max_node_r * 3.5 + layer_gap * 1.8
        elif layer == 2:
            min_circumference = sum(node_radii[n] * 2 * 1.5 for n in layer_nodes)
            min_radius = min_circumference / (2 * np.pi) if n > 0 else max_node_r * 2
            min_layer_r = prev_layer_max_r + max_node_r * 2.0 + layer_gap * 0.6
        else:
            min_circumference = sum(node_radii[n] * 2 * 1.5 for n in layer_nodes)
            min_radius = min_circumference / (2 * np.pi) if n > 0 else max_node_r * 2
            min_layer_r = prev_layer_max_r + max_node_r * 2.5 + layer_gap
        layer_r = max(min_radius, min_layer_r)
        layer_radii[layer] = layer_r
        prev_layer_max_r = layer_r + max_node_r

    for layer in range(1, max_layer + 1):
        if layer not in layers:
            continue
        layer_nodes = layers[layer]
        n = len(layer_nodes)
        if n == 0:
            continue
        layer_r = layer_radii[layer]
        layer_nodes_sorted = sorted(layer_nodes, key=lambda x: node_radii[x], reverse=True)
        if layer == 1:
            base_angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            angle_spread = 2 * np.pi / n
            base_angles = [
                a + angle_spread * 0.2 * np.sin(i * np.pi / n) for i, a in enumerate(base_angles)
            ]
        else:
            base_angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        for idx, node in enumerate(layer_nodes_sorted):
            node_r = node_radii[node]
            base_angle = base_angles[idx]
            if layer == 1:
                x, y = find_valid_position_large_search(node_r, layer_r, base_angle, placed_nodes)
            else:
                x, y = find_valid_position(node_r, layer_r, base_angle, placed_nodes)
            pos[node] = (x, y)
            placed_nodes.append((x, y, node_r))

    node_sizes = [node_sizes_dict[n] for n in G.nodes()]

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    all_coords = np.array(list(pos.values()))
    if len(all_coords) > 0:
        x_range = all_coords[:, 0].max() - all_coords[:, 0].min()
        y_range = all_coords[:, 1].max() - all_coords[:, 1].min()
        data_range = max(x_range, y_range) * 1.3
        center_x, center_y = all_coords.mean(axis=0)
        ax.set_xlim(center_x - data_range / 2, center_x + data_range / 2)
        ax.set_ylim(center_y - data_range / 2, center_y + data_range / 2)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        data_width = xlim[1] - xlim[0]
        data_height = ylim[1] - ylim[0]
        inches_per_data_x = figsize[0] / data_width
        inches_per_data_y = figsize[1] / data_height
        for n in G.nodes():
            size = node_sizes_dict[n]
            radius_points = np.sqrt(size / np.pi)
            radius_inches = radius_points / 72.0
            radius_data = radius_inches / min(inches_per_data_x, inches_per_data_y)
            node_radii[n] = radius_data

    all_w = np.array([abs(G[u][v]["raw_weight"]) for u, v in G.edges()])
    max_w = all_w.max() if len(all_w) else 1

    def edge_width(w):
        return 1.8 + 7.0 * abs(w) / max_w

    pos_edges = [(u, v) for u, v in G.edges() if G[u][v]["raw_weight"] > 0]
    neg_edges = [(u, v) for u, v in G.edges() if G[u][v]["raw_weight"] < 0]

    def draw_edge_with_arrow(u, v, color):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        dx, dy = x1 - x0, y1 - y0
        d = np.hypot(dx, dy)
        if d == 0:
            return
        ux, uy = dx / d, dy / d
        r0 = node_radii[u]
        r1 = node_radii[v]
        start_offset = r0 * 1.05
        start = (x0 + ux * start_offset, y0 + uy * start_offset)
        if d < (r0 + r1) * 2.5:
            end_offset = r1 * 1.2
        else:
            end_offset = r1 * 1.12
        end = (x1 - ux * end_offset, y1 - uy * end_offset)
        edge_length = np.hypot(end[0] - start[0], end[1] - start[1])
        min_length = max(r0, r1) * 0.1
        if edge_length < min_length:
            return
        edge_length_normalized = min(edge_length / (max(r0, r1) * 3), 1.0)
        min_arrow_size, max_arrow_size = 20, 28
        arrow_size = min_arrow_size + (max_arrow_size - min_arrow_size) * max(
            edge_length_normalized, 0.3
        )
        arrow = mpatches.FancyArrowPatch(
            start,
            end,
            arrowstyle="->",
            connectionstyle="arc3,rad=0.15",
            lw=edge_width(G[u][v]["raw_weight"]),
            color=color,
            alpha=0.9,
            mutation_scale=arrow_size,
            mutation_aspect=1.2,
            zorder=1,
            shrinkA=0,
            shrinkB=0,
        )
        ax.add_patch(arrow)

    for u, v in pos_edges:
        draw_edge_with_arrow(u, v, "#FF6B9D")
    for u, v in neg_edges:
        draw_edge_with_arrow(u, v, "#00B4D8")

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="#212121",
        linewidths=1.8,
        ax=ax,
    )

    coords = np.array(list(pos.values()))
    center = coords.mean(axis=0)
    radius = np.max(np.linalg.norm(coords - center, axis=1)) * 1.12
    circle = plt.Circle(
        center,
        radius,
        color="#B0BEC5",
        fill=False,
        linewidth=2.4,
        alpha=0.35,
        zorder=0,
    )
    ax.add_patch(circle)

    nx.draw_networkx_labels(
        G, pos, font_size=13, font_weight="bold", font_color="#333", ax=ax
    )

    ax.set_title(
        f"Radial causal network (center by in-degree: {center_node})",
        fontsize=18,
        pad=20,
    )
    ax.axis("off")
    ax.set_aspect("equal")

    result_dir = output_dir if output_dir is not None else _default_output_dir(is_normal)
    os.makedirs(result_dir, exist_ok=True)
    path = os.path.join(result_dir, save_name)
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    return path
