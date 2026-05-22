# idopNetwork (informative, dynamic, omnidirectional, and personalized Network)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official Python implementation of **idopNetwork** — a quantitative framework to reconstruct **dynamic, directed, weighted** interaction networks of clinical laboratory indicators across BMI strata (Underweight, Normal, Overweight, Obesity).

## 1. Project overview

Cross-sectional clinical tables are turned into quasi-dynamic trajectories (pseudotime sorting + curve fitting), then sparse regression with terminal-divergence constraints yields a directed adjacency matrix. Topology metrics, degree plots, Cytoscape exports, and optional bootstrap validation follow.

**End-to-end pipeline (`main.py`):**

```text
CSV load (4 BMI cohorts)
  → variable catalog & optional column drop
  → quasi-dynamic frames (pseudotime sorting)
  → multi-cohort curve-fit figure
  → idopNetwork reconstruction (one selected cohort)
  → effect decomposition & adjacency plots
  → topology summary & degree tables
  → Cytoscape CSV export
  → optional radial/degree figures (networkx)
  → optional bootstrap core-node stability (results_bootstrap/)
```

---

## 2. Environment setup

**Requirements:** Python ≥ 3.8

**Install core dependencies:**

```bash
pip install numpy pandas scikit-learn matplotlib scipy igraph asgl seaborn
```

| Package | Role |
| --- | --- |
| `numpy`, `pandas`, `scipy` | Data & numerics |
| `scikit-learn` | Scaling in `clinical_data.load_clinical_matrix` |
| `matplotlib` | All figures |
| `igraph` | Layout in `plot_directed_adjacency_graph` |
| `asgl` | ASGL sparse regression in `reconstruct_idop_network` |
| `seaborn` | Bootstrap heatmaps in `bootstrap_core_nodes` |

**Optional:**

```bash
pip install networkx   # radial in/out-degree networks in degree_and_radial_plots
pip install streamlit  # interactive UI: streamlit_app.py
```

> If `asgl` is not on PyPI in your environment, install it from your project’s documented source before running reconstruction.

---

## 3. Data layout

Default paths in `main.py` (RUC BMI bins):

```text
data/
└── RUC_bins/
    ├── Underweight.csv
    ├── Normal_weight.csv
    ├── Overweight.csv
    └── Obesity.csv
```

Additional public cohort files (not used by default in `main.py`):

```text
data/NHANES_bins/
    NHANES_underweight.csv, NHANES_normal.csv, ...
```

**CSV format:** rows = samples; columns = clinical features (Chinese names mapped to abbreviations in `clinical_data.CLINICAL_NAME_TO_ABBREV`). Optional first column as index via `load_clinical_matrix(..., index_col=0)`.

---

## 4. Quick start

### Full pipeline (paper-style workflow)

```bash
python main.py
```

On each run, `main.py` **deletes and recreates** `results/` and `results_Cytospace/` before writing new outputs.

### Network from a single CSV (no 4-cohort loop)

```bash
python build_network_from_csv.py --csv data/RUC_bins/Obesity.csv --out results_build_network_from_csv
```

Auto-detects input type:

1. Edge list (`From`, `To`, `Effect`/`Weight`/`w`)
2. Square adjacency matrix (row index = node names)
3. Otherwise: sample × feature table → `reconstruct_idop_network`

### Edge-level bootstrap (standalone)

```bash
python bootstrap_edge_stability.py
```

Writes under `results/`: `bootstrap_edge_frequency.csv`, `bootstrap_stable_network.csv`, `bootstrap_summary.txt`.

### Interactive UI

```bash
streamlit run streamlit_app.py
```

---

## 5. `main.py` workflow (step by step)

| Step | What happens | Key API |
| --- | --- | --- |
| 1 | Load four BMI CSVs (`n_samples=1000`, `MinMaxScaler`) | `clinical_data.load_clinical_matrix` |
| 2 | Build & print variable catalog; optional drops | `build_variable_catalog`, `drop_variables_by_index_or_name` |
| 3 | Save `results/variable_mapping.csv` | — |
| 4 | Quasi-dynamic sort per cohort | `pseudotime_curves.build_quasi_dynamic_frame` |
| 5 | Four-group curve fits | `plot_quasi_dynamic_curve_fits` → `results/data_curve_fit.png` |
| 6 | Reconstruct network for **one** cohort (`selected_cohort_index`) | `idop_reconstruction.reconstruct_idop_network` |
| 7 | Effect curves for selected variables | `plot_effect_decomposition_curves` → `results/effect_decomposition.png` |
| 8 | Directed network figures | `plot_directed_adjacency_graph` → `network_matrix*.png` |
| 9 | Topology & degrees | `summarize_directed_topology`, degree table → `network_topology_summary.csv`, `network_degrees.csv` |
| 10 | Cytoscape bundle | `results_Cytospace/adjacency_matrix_{BMI}.csv`, `cytoscape_edges_*.csv`, `cytoscape_nodes_*.csv` |
| 11 | Degree / radial plots (if `networkx` installed) | `degree_and_radial_plots.plot_degrees_from_adjacency`, `plot_radial_*` |
| 12 | Optional bootstrap (off by default) | `bootstrap_core_nodes.run_core_node_bootstrap` → `results_bootstrap/` |

---

## 6. Output files

### `results/` (main run)

| File | Description |
| --- | --- |
| `variable_mapping.csv` | `index`, `variable_name`, `abbreviation` |
| `data_curve_fit.png` | Power-law (or configured) fits, four BMI groups |
| `data_curve_fit_{Group}.png` | Per-group fits if `SAVE_CURVE_FIT_SEPARATELY_BY_GROUP = True` |
| `effect_decomposition.png` | Decomposed effect trajectories (`plot_effect_decomposition_curves`) |
| `network_matrix.png` | Circle layout, full network |
| `network_matrix_single.png` | Hub layout (`center_mode="total"`) |
| `network_matrix_outdeg.png` / `network_matrix_indeg.png` | Out- / in-degree centered layouts |
| `network_topology_summary.csv` | From `summarize_directed_topology` |
| `network_degrees.csv` | Per-node in/out/total degree + max-degree summary lines |
| `in_out_degree_comparison_{BMI}.png` | Bar comparison (needs `networkx`) |
| `radial_outdegree_network_{BMI}.png` / `radial_indegree_network_{BMI}.png` | Radial layouts |

### `results_Cytospace/`

| File | Description |
| --- | --- |
| `adjacency_matrix_{BMI}.csv` | Weighted adjacency (abbreviations as labels) |
| `cytoscape_edges_{BMI}.csv` | Directed edges with sign and Chinese/abbr names |
| `cytoscape_nodes_{BMI}.csv` | Self-effect, `self_activation` / `self_inhibition` |

### `results_bootstrap/` (when `RUN_CORE_NODE_BOOTSTRAP = True`)

Per cohort subdirectory (`Underweight`, `Normal`, …), e.g.:

- `bootstrap_core_node_frequency.csv`
- `bootstrap_stable_core_nodes_summary.csv`
- `bootstrap_core_node_stability_heatmap.png`
- `bootstrap_core_node_bar_{BMI}.png`
- `bootstrap_rank_distribution_top1.png`
- `bootstrap_stable_core_summary_figure.png`
- `bootstrap_top{k}_jaccard_consistency.csv` / `.png`
- `bootstrap_directed_edge_stability_matrix_{BMI}.png`

### `results_build_network_from_csv/` (CLI default `--out`)

`variable_mapping.csv`, `adjacency_matrix_{stem}.csv`, `edges_{stem}.csv`, `network_matrix_{stem}.png`

---

## 7. Customization (`main.py`)

**A. BMI cohort for reconstruction**

```python
selected_cohort_index = 3  # 0 Underweight, 1 Normal, 2 Overweight, 3 Obesity
```

**B. Drop variables**

```python
delete_by_index = []   # e.g. [0, 5]
delete_by_name = []    # exact variable_name from catalog
```

**C. Reconstruction hyperparameters**

```python
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
```

Passed to `reconstruct_idop_network`. Additional kwargs supported in code: `curve_mode` (`"linspace"` | `"by_index"` | `"raw"`), `curve_num_points`.

**D. Separate curve-fit PNGs per group**

```python
SAVE_CURVE_FIT_SEPARATELY_BY_GROUP = True
```

**E. Bootstrap core nodes**

```python
RUN_CORE_NODE_BOOTSTRAP = False   # set True to run
BOOTSTRAP_ALL_COHORTS = True      # all four groups vs current cohort only
BOOTSTRAP_B = 100
BOOTSTRAP_TOP_K = 5
BOOTSTRAP_RESULTS_DIR = "results_bootstrap"
```

**F. Row index / deduplication**

```python
RESET_INDEX_FOR_UNIQUE_ROWS = False
```

Set `True` when the first CSV column is sample ID (use `index_col=0` in `load_clinical_matrix`) to avoid heavy row loss in `build_quasi_dynamic_frame`.

---

## 8. Python module map

| Role | Module | Main public API |
| --- | --- | --- |
| Data loading & catalog | `clinical_data.py` | `load_clinical_matrix`, `build_variable_catalog`, `drop_variables_by_index_or_name`, `scale_numeric_features` |
| Pseudotime & curves | `pseudotime_curves.py` | `build_quasi_dynamic_frame`, `plot_quasi_dynamic_curve_fits`, `fit_power_law_params`, `sample_fitted_curve` |
| Network reconstruction | `idop_reconstruction.py` | `reconstruct_idop_network`, `plot_directed_adjacency_graph`, `plot_effect_decomposition_curves`, `summarize_directed_topology` |
| CSV → network CLI | `build_network_from_csv.py` | `reconstruct_from_feature_table`, `build_from_edge_or_adjacency_table`; `python build_network_from_csv.py` |
| Network I/O | `network_io.py` | `load_edges_csv`, `load_adjacency_csv` |
| Bootstrap (core nodes) | `bootstrap_core_nodes.py` | `run_core_node_bootstrap`, `weighted_degrees_from_adjacency` |
| Bootstrap (edges) | `bootstrap_edge_stability.py` | `run_edge_bootstrap` |
| Degree & radial plots | `degree_and_radial_plots.py` | `plot_degrees_from_adjacency`, `plot_radial_out_degree_network`, `plot_radial_in_degree_network` |
| Adjacency-only degree CLI | `plot_degrees_from_adjacency.py` | `run_from_file` (standalone script) |
| Subset curve panels | `subset_curve_plots.py` | `plot_cohort_subset_curves`, `resolve_variable_tokens` |
| BMI schematic figure | `plot_bmi_core_node_schematic.py` | Standalone matplotlib script |
| Web UI | `streamlit_app.py` | `streamlit run streamlit_app.py` |

**Entry points**

| Command | Purpose |
| --- | --- |
| `python main.py` | Full 4-cohort pipeline + exports |
| `python build_network_from_csv.py --csv … --out …` | Single-file network build |
| `python bootstrap_edge_stability.py` | Edge stability bootstrap |
| `streamlit run streamlit_app.py` | Interactive analysis |

---

## 9. Citation

If you use this code, please cite:

```bibtex
@article{idop2026,
  title={Topological Deadlock in Obesity: idopNetwork Reveals Systemic Regulatory Collapse via Path Homology},
  author={Jiapeng Wang, Yu Wang},
  journal={Journal of Translational Medicine},
  year={2026}
}
```

**Contact:** 19553751383@163.com  
**Affiliation:** Inner Mongolia Normal University
