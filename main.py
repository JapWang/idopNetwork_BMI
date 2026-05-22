"""
iDOP Network entry point.
Pipeline: data loading → quasi-dynamic transform → curve fitting → network reconstruction.
"""

import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import os
import shutil
import numpy as np
import pandas as pd
import clinical_data
import pseudotime_curves
import idop_reconstruction
import bootstrap_core_nodes

RESET_INDEX_FOR_UNIQUE_ROWS = False
data_1_path = r"data\RUC_bins\Underweight.csv"
data_2_path = r"data\RUC_bins\Normal_weight.csv"
data_3_path = r"data\RUC_bins\Overweight.csv"
data_4_path = r"data\RUC_bins\Obesity.csv"

data_1 = clinical_data.load_clinical_matrix(
    data_1_path, n_samples=1000, n_features=None, scaler_type="MinMaxScaler",
    reset_index_for_unique_rows=RESET_INDEX_FOR_UNIQUE_ROWS,
)
data_2 = clinical_data.load_clinical_matrix(
    data_2_path, n_samples=1000, n_features=None, scaler_type="MinMaxScaler",
    reset_index_for_unique_rows=RESET_INDEX_FOR_UNIQUE_ROWS,
)
data_3 = clinical_data.load_clinical_matrix(
    data_3_path, n_samples=1000, n_features=None, scaler_type="MinMaxScaler",
    reset_index_for_unique_rows=RESET_INDEX_FOR_UNIQUE_ROWS,
)
data_4 = clinical_data.load_clinical_matrix(
    data_4_path, n_samples=1000, n_features=None, scaler_type="MinMaxScaler",
    reset_index_for_unique_rows=RESET_INDEX_FOR_UNIQUE_ROWS,
)

variable_mapping = clinical_data.build_variable_catalog(data_1)
print("=" * 60)
print("Variable mapping (index → name → abbreviation)")
print(variable_mapping.to_string(index=False))
print(
    f"Total variables: {len(variable_mapping)}; "
    f"data shape: {data_1.shape[0]} rows × {data_1.shape[1]} columns"
)

delete_by_index = []
delete_by_name = []

[data_1, data_2, data_3, data_4], variable_mapping, _ = clinical_data.drop_variables_by_index_or_name(
    [data_1, data_2, data_3, data_4],
    variable_mapping,
    delete_by_index=delete_by_index,
    delete_by_name=delete_by_name,
)

if os.path.isdir("results"):
    shutil.rmtree("results")
os.makedirs("results", exist_ok=True)

variable_mapping.to_csv("results/variable_mapping.csv", index=False, encoding="utf-8-sig")
print("Variable mapping saved to results/variable_mapping.csv\n")
column_labels = dict(zip(variable_mapping["variable_name"], variable_mapping["abbreviation"]))

data_quasi_dynamic_1 = pseudotime_curves.build_quasi_dynamic_frame(data_1)
data_quasi_dynamic_2 = pseudotime_curves.build_quasi_dynamic_frame(data_2)
data_quasi_dynamic_3 = pseudotime_curves.build_quasi_dynamic_frame(data_3)
data_quasi_dynamic_4 = pseudotime_curves.build_quasi_dynamic_frame(data_4)

SAVE_CURVE_FIT_SEPARATELY_BY_GROUP = False

if SAVE_CURVE_FIT_SEPARATELY_BY_GROUP:
    curve_fit_sources = [
        ("Underweight", data_quasi_dynamic_1, 1),
        ("Normal", data_quasi_dynamic_2, 2),
        ("Overweight", data_quasi_dynamic_3, 0),
        ("Obesity", data_quasi_dynamic_4, 4),
    ]
    for _label, _qd, _palette_idx in curve_fit_sources:
        pseudotime_curves.plot_quasi_dynamic_curve_fits(
            [_qd],
            save_path=f"results/data_curve_fit_{_label}.png",
            group_labels=[_label],
            group_palette_indices=[_palette_idx],
            first_n=None,
            column_labels=column_labels,
        )
else:
    pseudotime_curves.plot_quasi_dynamic_curve_fits(
        [data_quasi_dynamic_1, data_quasi_dynamic_2, data_quasi_dynamic_3, data_quasi_dynamic_4],
        save_path="results/data_curve_fit.png",
        group_labels=["Underweight", "Normal", "Overweight", "Obesity"],
        group_palette_indices=[1, 2, 0, 4],
        first_n=None,
        column_labels=column_labels,
    )

selected_cohort_index = 3  # 0 Underweight, 1 Normal, 2 Overweight, 3 Obesity

BMI_COHORT_SOURCES = [
    ("Underweight", data_1, data_1_path),
    ("Normal", data_2, data_2_path),
    ("Overweight", data_3, data_3_path),
    ("Obesity", data_4, data_4_path),
]

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

group_label, data_for_network, data_path_for_network = BMI_COHORT_SOURCES[selected_cohort_index]
_bmi_label = group_label
print(f"\nidopNetwork reconstruction will use group: {_bmi_label}")

models, effects, adjusted_matrix, intercepts = idop_reconstruction.reconstruct_idop_network(
    data_for_network,
    **RECONSTRUCTION_HYPERPARAMS,
)

n_time = effects[0].shape[0]
intercepts_arr = np.asarray(intercepts).ravel()
pure_self_effects = np.diag(adjusted_matrix) - n_time * intercepts_arr

_idx_dup_first = data_for_network.index.duplicated(keep="first")
_n_eff_rows = len(data_for_network)
_n_eff_dup = int(_idx_dup_first.sum())
_n_eff_dedup = _n_eff_rows - _n_eff_dup
print(
    "\n[Effect decomposition / build_quasi_dynamic_frame] data_for_network: "
    f"rows={_n_eff_rows}; dropped (duplicate index)={_n_eff_dup}; "
    f"kept after dedup={_n_eff_dedup} (keep=first, same as pseudotime_curves.build_quasi_dynamic_frame). "
    f"unique index count={data_for_network.index.nunique()}."
)
idop_reconstruction.plot_effect_decomposition_curves(data_for_network, effects, 
save_path="results/effect_decomposition.png",
var_names=['SBP','DBP','WBC','Hb','MCV','PLT','MPV','RDW-SD','Lym#','Neu#','RBC','Mo','Eo#','Ba#','Alt','Tbil','dbil','Sg','ph','cr','uric'])

center_node = None

idop_reconstruction.plot_directed_adjacency_graph(
    adjusted_matrix,
    save_path="results/network_matrix.png",
    layout_mode="circle",
    self_effect_for_color=pure_self_effects,
)

idop_reconstruction.plot_directed_adjacency_graph(
    adjusted_matrix,
    save_path="results/network_matrix_single.png",
    center_node=center_node,
    center_mode="total",
    self_effect_for_color=pure_self_effects,
)

idop_reconstruction.plot_directed_adjacency_graph(adjusted_matrix, save_path="results/network_matrix_outdeg.png", center_mode="out", self_effect_for_color=pure_self_effects)
idop_reconstruction.plot_directed_adjacency_graph(adjusted_matrix, save_path="results/network_matrix_indeg.png", center_mode="in", self_effect_for_color=pure_self_effects)

topology_df = idop_reconstruction.summarize_directed_topology(adjusted_matrix, bmi_stratum=_bmi_label)
topology_df.to_csv("results/network_topology_summary.csv", index=False, encoding="utf-8-sig")

mask = (np.arange(adjusted_matrix.shape[0])[None, :] != np.arange(adjusted_matrix.shape[0])[:, None]) & ~np.isnan(adjusted_matrix) & (adjusted_matrix != 0)
out_deg = mask.sum(axis=1)
in_deg = mask.sum(axis=0)
degree_df = pd.DataFrame({
    "index": variable_mapping["index"],
    "variable_name": variable_mapping["variable_name"],
    "abbreviation": variable_mapping["abbreviation"],
    "out_degree": out_deg.astype(int),
    "in_degree": in_deg.astype(int),
})
degree_df["total_degree"] = degree_df["out_degree"] + degree_df["in_degree"]
row_max_out = degree_df.loc[degree_df["out_degree"].idxmax()]
row_max_in = degree_df.loc[degree_df["in_degree"].idxmax()]
row_max_total = degree_df.loc[degree_df["total_degree"].idxmax()]

summary_text = (
    f"Max out-degree: index={int(row_max_out['index'])}, name={row_max_out['variable_name']}, "
    f"abbr={row_max_out['abbreviation']}, out-degree={int(row_max_out['out_degree'])}\n"
    f"Max in-degree: index={int(row_max_in['index'])}, name={row_max_in['variable_name']}, "
    f"abbr={row_max_in['abbreviation']}, in-degree={int(row_max_in['in_degree'])}\n"
    f"Max total degree: index={int(row_max_total['index'])}, name={row_max_total['variable_name']}, "
    f"abbr={row_max_total['abbreviation']}, total degree={int(row_max_total['total_degree'])}"
)

degree_df.to_csv("results/network_degrees.csv", index=False, encoding="utf-8-sig")
with open("results/network_degrees.csv", "a", encoding="utf-8-sig") as f:
    f.write("\n")
    f.write(summary_text)

cyto_dir = "results_Cytospace"
if os.path.isdir(cyto_dir):
    shutil.rmtree(cyto_dir)
os.makedirs(cyto_dir, exist_ok=True)

adjacency_df = pd.DataFrame(
    adjusted_matrix,
    index=variable_mapping["abbreviation"],
    columns=variable_mapping["abbreviation"],
)
adjacency_path = os.path.join(cyto_dir, f"adjacency_matrix_{_bmi_label}.csv")
adjacency_df.to_csv(adjacency_path, encoding="utf-8-sig")
print(f"Adjacency matrix saved: {adjacency_path}")

try:
    import degree_and_radial_plots

    _deg_plot = degree_df.rename(
        columns={"abbreviation": "Module", "in_degree": "in_degree", "out_degree": "out_degree"}
    )
    degree_and_radial_plots.plot_degrees_from_adjacency(
        _deg_plot,
        title=f"In- and Out-Degree ({_bmi_label})",
        save_name=f"in_out_degree_comparison_{_bmi_label}.png",
        output_dir="results",
    )
    degree_and_radial_plots.plot_radial_out_degree_network(
        adjacency_df,
        save_name=f"radial_outdegree_network_{_bmi_label}.png",
        output_dir="results",
    )
    degree_and_radial_plots.plot_radial_in_degree_network(
        adjacency_df,
        save_name=f"radial_indegree_network_{_bmi_label}.png",
        output_dir="results",
    )
    print(
        "Degree and radial network figures written to results/: "
        f"in_out_degree_comparison_{_bmi_label}.png, "
        f"radial_outdegree_network_{_bmi_label}.png, "
        f"radial_indegree_network_{_bmi_label}.png"
    )
except ImportError as _nde:
    print(f"Skipped degree_and_radial_plots (missing dependency: {_nde}). Run: pip install networkx")

edges = []
n_nodes = adjusted_matrix.shape[0]
for i in range(n_nodes):
    for j in range(n_nodes):
        if i == j:
            continue
        w = adjusted_matrix[i, j]
        if np.isnan(w) or w == 0:
            continue
        sign = "positive" if w > 0 else "negative"
        edges.append({
            "source_index": int(i),
            "target_index": int(j),
            "source_name_cn": variable_mapping.loc[i, "variable_name"],
            "target_name_cn": variable_mapping.loc[j, "variable_name"],
            "source_name_abbr": variable_mapping.loc[i, "abbreviation"],
            "target_name_abbr": variable_mapping.loc[j, "abbreviation"],
            "weight": float(w),
            "sign": sign,
        })

edges_df = pd.DataFrame(edges)
edges_path = os.path.join(cyto_dir, f"cytoscape_edges_{_bmi_label}.csv")
edges_df.to_csv(edges_path, index=False, encoding="utf-8-sig")
print(f"Cytoscape edge table saved: {edges_path}")

nodes = []
for i in range(n_nodes):
    self_eff = float(pure_self_effects[i])
    node_type = "self_activation" if self_eff > 0 else "self_inhibition"
    nodes.append({
        "id_index": int(i),
        "name_cn": variable_mapping.loc[i, "variable_name"],
        "name_abbr": variable_mapping.loc[i, "abbreviation"],
        "self_effect": self_eff,
        "self_type": node_type,
    })

nodes_df = pd.DataFrame(nodes)
nodes_path = os.path.join(cyto_dir, f"cytoscape_nodes_{_bmi_label}.csv")
nodes_df.to_csv(nodes_path, index=False, encoding="utf-8-sig")
print(f"Cytoscape node table saved: {nodes_path}")

RUN_CORE_NODE_BOOTSTRAP = False
BOOTSTRAP_ALL_COHORTS = True
BOOTSTRAP_RESULTS_DIR = "results_bootstrap"
BOOTSTRAP_TOP_K = 5
_BOOTSTRAP_SOURCE_INDEX_ORDER = [0, 1, 2, 3]
BOOTSTRAP_B = 100

if RUN_CORE_NODE_BOOTSTRAP:
    if os.path.isdir(BOOTSTRAP_RESULTS_DIR):
        shutil.rmtree(BOOTSTRAP_RESULTS_DIR)
    os.makedirs(BOOTSTRAP_RESULTS_DIR, exist_ok=True)

    def _print_bootstrap_outputs_for_group(results_subdir: str, bmi_en: str) -> None:
        """List relative output paths after one group finishes."""
        _bt = ""
        print(f"\nResults directory: {results_subdir}/")
        print(f"  - bootstrap_core_node_frequency{_bt}.csv")
        print(f"  - bootstrap_stable_core_nodes_summary{_bt}.csv")
        print(f"  - bootstrap_core_node_stability_heatmap{_bt}.png")
        print(f"  - bootstrap_core_node_bar_{bmi_en}.png")
        print(f"  - bootstrap_rank_distribution_top1{_bt}.png")
        print(f"  - bootstrap_stable_core_summary_figure{_bt}.png")
        print(f"  - bootstrap_top{BOOTSTRAP_TOP_K}_jaccard_consistency{_bt}.csv/.png")
        print(f"  - bootstrap_directed_edge_stability_matrix_{bmi_en}.png")

    if BOOTSTRAP_ALL_COHORTS:
        print("\n" + "=" * 70)
        print(
            f"Bootstrap core-node stability — four groups in sequence "
            f"(B={BOOTSTRAP_B}, top {BOOTSTRAP_TOP_K}, stability threshold >= 95%)"
        )
        print(
            f"Order: Underweight → Normal → Overweight → Obesity; "
            f"root {BOOTSTRAP_RESULTS_DIR}/<group>/"
        )
        print("=" * 70)
        for _ord_i, _src_idx in enumerate(_BOOTSTRAP_SOURCE_INDEX_ORDER, start=1):
            _g_label, _g_data, _g_path = BMI_COHORT_SOURCES[_src_idx]
            _sub = os.path.join(BOOTSTRAP_RESULTS_DIR, _g_label)
            if os.path.isdir(_sub):
                shutil.rmtree(_sub)
            os.makedirs(_sub, exist_ok=True)
            print("\n" + "-" * 70)
            print(
                f"[{_ord_i}/4] Bootstrap: {_g_label} → {_sub}/ "
                f"(available after this group completes)"
            )
            print("-" * 70)
            bootstrap_core_nodes.run_core_node_bootstrap(
                data_list=[_g_data],
                group_names=[_g_label],
                variable_mapping=variable_mapping,
                results_dir=_sub,
                output_tag=None,
                B=BOOTSTRAP_B,
                top_k=BOOTSTRAP_TOP_K,
                stable_threshold=0.95,
                basis_max_order=RECONSTRUCTION_HYPERPARAMS["basis_max_order"],
                verbose=True,
            )
            print(
                f"\n>>> Group {_ord_i}/4 done: {_g_label}. See {os.path.normpath(_sub)}"
            )
            _print_bootstrap_outputs_for_group(_sub, _g_label)
        print("\n" + "=" * 70)
        print(f"All four Bootstrap runs finished. Root: {BOOTSTRAP_RESULTS_DIR}/")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print(
            f"Bootstrap core-node stability — current group: {group_label} "
            f"(B={BOOTSTRAP_B}, top {BOOTSTRAP_TOP_K}, stability threshold >= 95%)"
        )
        print("=" * 70)
        bootstrap_core_nodes.run_core_node_bootstrap(
            data_list=[data_for_network],
            group_names=[_bmi_label],
            variable_mapping=variable_mapping,
            results_dir=BOOTSTRAP_RESULTS_DIR,
            output_tag=_bmi_label,
            B=BOOTSTRAP_B,
            top_k=BOOTSTRAP_TOP_K,
            stable_threshold=0.95,
            basis_max_order=RECONSTRUCTION_HYPERPARAMS["basis_max_order"],
            verbose=True,
        )
        _bt = f"_{_bmi_label}"
        print(f"\nBootstrap finished ({group_label}). Outputs under {BOOTSTRAP_RESULTS_DIR}/:")
        print(f"  - bootstrap_core_node_frequency{_bt}.csv")
        print(f"  - bootstrap_stable_core_nodes_summary{_bt}.csv")
        print(f"  - bootstrap_core_node_stability_heatmap{_bt}.png")
        print(f"  - bootstrap_core_node_bar_{_bmi_label}.png")
        print(f"  - bootstrap_rank_distribution_top1{_bt}.png")
        print(f"  - bootstrap_stable_core_summary_figure{_bt}.png")
        print(f"  - bootstrap_top{BOOTSTRAP_TOP_K}_jaccard_consistency{_bt}.csv/.png")
        print(f"  - bootstrap_directed_edge_stability_matrix_{_bmi_label}{_bt}.png")