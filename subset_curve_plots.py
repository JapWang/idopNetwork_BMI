import os
import re
import shutil

import clinical_data
import pseudotime_curves

def resolve_variable_tokens(variable_mapping, tokens):
    """
    Resolve user tokens (abbreviation, case-insensitive, or exact variable_name) to column names.
    Order follows tokens; duplicates keep first occurrence only.
    """
    if not tokens:
        return []
    abbrev_to_name = {}
    for _, row in variable_mapping.iterrows():
        ab = str(row["abbreviation"]).strip()
        abbrev_to_name[ab.lower()] = row["variable_name"]
    name_set = set(variable_mapping["variable_name"])
    resolved = []
    seen = set()
    for raw in tokens:
        t = str(raw).strip()
        if not t:
            continue
        if t in name_set:
            col = t
        else:
            col = abbrev_to_name.get(t.lower())
            if col is None:
                raise ValueError(
                    "Cannot resolve token {!r}; use abbreviation or variable_name from variable_mapping.".format(
                        raw
                    )
                )
        if col not in seen:
            seen.add(col)
            resolved.append(col)
    return resolved

def abbrev_suffix_for_filename(column_names, variable_mapping):
    """Build output filename suffix from abbreviations joined by underscores."""
    name_to_abbr = dict(zip(variable_mapping["variable_name"], variable_mapping["abbreviation"]))
    parts = []
    for c in column_names:
        ab = str(name_to_abbr.get(c, c))
        safe = re.sub(r"[^\w\-.]+", "_", ab, flags=re.ASCII)
        safe = safe.strip("_") or "var"
        parts.append(safe)
    return "_".join(parts)

def plot_cohort_subset_curves(
    tokens,
    curve_fit_sources,
    variable_mapping,
    column_labels,
    out_dir="result_power",
    sample_method="linspace",
    dpi=300,
    panel_aspect="golden",
    row_height_inches=2.35,
):
    """
    One figure per BMI stratum with only tokens variables (scatter + power-law fit).
    Single-column subplot stack; panel_aspect: square / 6:4 / golden.
    row_height_inches: row height in inches when stacked vertically.
    tokens: abbreviations or variable names, e.g. ["dbp", "sbp"].
    curve_fit_sources: [ (label, quasi_dynamic_df, palette_idx), ... ]
    """
    column_names = resolve_variable_tokens(variable_mapping, tokens)
    if not column_names:
        return
    suffix = abbrev_suffix_for_filename(column_names, variable_mapping)
    for label, qd, palette_idx in curve_fit_sources:
        pseudotime_curves.plot_quasi_dynamic_curve_fits(
            [qd],
            save_path=os.path.join(out_dir, "data_curve_fit_{}_compare_{}.png".format(label, suffix)),
            group_labels=[label],
            group_palette_indices=[palette_idx],
            column_labels=column_labels,
            only_columns=column_names,
            stack_vertically=True,
            panel_aspect=panel_aspect,
            row_height_inches=row_height_inches,
            sample_method=sample_method,
            first_n=None,
            dpi=dpi,
        )

RESET_INDEX_FOR_UNIQUE_ROWS = True
data_1_path = r"D:\idopNetwork工作\idopNetwork_v3(3.9)\idopNetwork_v3\data\肥胖(合并).csv"
data_2_path = r"D:\idopNetwork工作\idopNetwork_v3(3.9)\idopNetwork_v3\data\超重.csv"
data_3_path = r"D:\idopNetwork工作\idopNetwork_v3(3.9)\idopNetwork_v3\data\体重过轻.csv"
data_4_path = r"D:\idopNetwork工作\idopNetwork_v3(3.9)\idopNetwork_v3\data\正常.csv"

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

delete_by_index = []
delete_by_name = [
    "年龄",
    "红细胞压积",
    "平均血红蛋白量",
    "平均血红蛋白浓度",
    "红细胞分布宽度Cv",
    "中性细胞比率",
    "淋巴细胞比率",
    "单核细胞百分比",
    "嗜酸性细胞 Eo",
    "嗜碱性细胞 Ba",
    "血小板压积",
    "大型血小板比率",
    "血小板分布宽度",
]

[data_1, data_2, data_3, data_4], variable_mapping, _ = clinical_data.drop_variables_by_index_or_name(
    [data_1, data_2, data_3, data_4],
    variable_mapping,
    delete_by_index=delete_by_index,
    delete_by_name=delete_by_name,
)

if os.path.isdir("result_power"):
    shutil.rmtree("result_power")
os.makedirs("result_power", exist_ok=True)

variable_mapping.to_csv("result_power/variable_mapping.csv", index=False, encoding="utf-8-sig")
column_labels = dict(zip(variable_mapping["variable_name"], variable_mapping["abbreviation"]))

data_quasi_dynamic_1 = pseudotime_curves.build_quasi_dynamic_frame(data_1)
data_quasi_dynamic_2 = pseudotime_curves.build_quasi_dynamic_frame(data_2)
data_quasi_dynamic_3 = pseudotime_curves.build_quasi_dynamic_frame(data_3)
data_quasi_dynamic_4 = pseudotime_curves.build_quasi_dynamic_frame(data_4)

data_curve_sample_1 = pseudotime_curves.sample_fitted_curve(data_quasi_dynamic_1, sample_method="linspace", num_points=1000)
data_curve_sample_2 = pseudotime_curves.sample_fitted_curve(data_quasi_dynamic_2, sample_method="linspace", num_points=1000)
data_curve_sample_3 = pseudotime_curves.sample_fitted_curve(data_quasi_dynamic_3, sample_method="linspace", num_points=1000)
data_curve_sample_4 = pseudotime_curves.sample_fitted_curve(data_quasi_dynamic_4, sample_method="linspace", num_points=1000)

SAVE_CURVE_FIT_SEPARATELY_BY_GROUP = False

curve_fit_sources = [
    ("Obesity", data_quasi_dynamic_1, 4),
    ("Overweight", data_quasi_dynamic_2, 0),
    ("Underweight", data_quasi_dynamic_3, 1),
    ("Normal", data_quasi_dynamic_4, 2),
]

if SAVE_CURVE_FIT_SEPARATELY_BY_GROUP:
    for _label, _qd, _palette_idx in curve_fit_sources:
        pseudotime_curves.plot_quasi_dynamic_curve_fits(
            [_qd],
            save_path=f"result_power/data_curve_fit_{_label}.png",
            group_labels=[_label],
            group_palette_indices=[_palette_idx],
            first_n=None,
            column_labels=column_labels,
            sample_method="linspace",
        )
else:
    pseudotime_curves.plot_quasi_dynamic_curve_fits(
        [data_quasi_dynamic_1, data_quasi_dynamic_2, data_quasi_dynamic_3, data_quasi_dynamic_4],
        save_path="result_power/data_curve_fit.png",
        group_labels=["Obesity", "Overweight", "Underweight", "Normal"],
        group_palette_indices=[4, 0, 1, 2],
        first_n=None,
        column_labels=column_labels,
        sample_method="linspace",
    )

COMPARE_VARIABLE_TOKENS = ["SBP","DBP","WBC","RDW-SD","Neu#","Mo","Alt","cr","uric"]
COMPARE_PANEL_ASPECT = "golden"
COMPARE_ROW_HEIGHT_INCHES = 2.35

if COMPARE_VARIABLE_TOKENS:
    plot_cohort_subset_curves(
        COMPARE_VARIABLE_TOKENS,
        curve_fit_sources,
        variable_mapping,
        column_labels,
        out_dir="result_power",
        sample_method="linspace",
        panel_aspect=COMPARE_PANEL_ASPECT,
        row_height_inches=COMPARE_ROW_HEIGHT_INCHES,
    )