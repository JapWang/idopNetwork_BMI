import os
import re
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

CLINICAL_NAME_TO_ABBREV = {
    "收缩压": "SBP",
    "舒张压": "DBP",
    "白细胞": "WBC",
    "淋巴细胞比率": "Lym%",
    "中性细胞比率": "Neu%",
    "血红蛋白": "Hb",
    "红细胞压积": "Hct",
    "红细胞平均体积": "MCV",
    "平均血红蛋白量": "MCH",
    "平均血红蛋白浓度": "MCHC",
    "血小板": "PLT",
    "平均血小板体积": "MPV",
    "大型血小板比率": "P-LCR",
    "血小板分布宽度": "PDW",
    "红细胞分布宽度Sd": "RDW-SD",
    "红细胞分布宽度Cv": "RDW-CV",
    "淋巴细胞数": "Lym#",
    "中性粒细胞数": "Neu#",
    "红细胞": "RBC",
    "单核细胞（Mo）": "Mo",
    "单核细胞(Mo)": "Mo",
    "嗜酸性细胞 Eo": "Eo",
    "嗜碱性细胞 Ba": "Ba",
    "嗜酸性细胞# Eo#": "Eo#",
    "嗜碱性细胞#  Ba#": "Ba#",
    "嗜碱性细胞# Ba#": "Ba#",
    "单核细胞百分比": "Mo%",
    "血小板压积": "PCT",
    "谷丙转氨酶（Alt）": "Alt",
    "谷丙转氨酶(Alt)": "Alt",
    "总胆红素（Tbil)": "Tbil",
    "总胆红素(Tbil)": "Tbil",
    "直接胆红素(dbil)": "dbil",
    "直接胆红素（dbil）": "dbil",
    "比重（Sg）": "Sg",
    "比重(Sg)": "Sg",
    "Ph值（ph）": "pH",
    "Ph值(ph)": "pH",
    "肌酐(cr)": "Cr",
    "肌酐（cr）": "Cr",
    "尿酸(uric)": "UA",
    "尿酸（uric）": "UA",
}

def load_clinical_matrix(
    file_path: str,
    n_samples: int = 100,
    n_features: Optional[int] = 100,
    scaler_type: str = "MinMaxScaler",
    reset_index_for_unique_rows: bool = False,
    index_col: Optional[int] = None,
) -> pd.DataFrame:
    """Load CSV; all columns are features unless index_col is set.

    reset_index_for_unique_rows:
        If True, drop the former index column and use 0..n-1 row labels (avoids
        massive row loss in build_quasi_dynamic_frame when index has many duplicates).
        Only use when the CSV first column is a sample ID (read with index_col=0).
    n_features: number of feature columns; None keeps all columns.
    """
    data = pd.read_csv(file_path, index_col=index_col)
    data = passthrough_columns(data)
    if reset_index_for_unique_rows:
        data = data.reset_index(drop=True)
    data = data.iloc[:n_samples, :n_features] if n_features is not None else data.iloc[:n_samples, :]
    data = scale_numeric_features(data, scaler_type)
    return data

def passthrough_columns(data: pd.DataFrame) -> pd.DataFrame:
    return data

def _abbreviation_for_column(col_name: str, index: int) -> str:
    """Resolve abbreviation: parenthetical text, lookup table, or V{i}."""
    m = re.search(r"[（(]([^）)]+)[）)]", str(col_name))
    if m:
        return m.group(1).strip()
    if col_name in CLINICAL_NAME_TO_ABBREV:
        return CLINICAL_NAME_TO_ABBREV[col_name]
    key = str(col_name).strip()
    if key in CLINICAL_NAME_TO_ABBREV:
        return CLINICAL_NAME_TO_ABBREV[key]
    return "V{}".format(index)

def build_variable_catalog(data: pd.DataFrame) -> pd.DataFrame:
    """Map column index, variable name (CSV header), and English abbreviation."""
    cols = data.columns.tolist()
    mapping = pd.DataFrame({
        "index": range(len(cols)),
        "variable_name": cols,
        "abbreviation": [_abbreviation_for_column(c, i) for i, c in enumerate(cols)],
    })
    return mapping

def drop_variables_by_index_or_name(
    data_list,
    variable_mapping: pd.DataFrame,
    delete_by_index=None,
    delete_by_name=None,
):
    """Drop variables by index and/or variable_name across all groups in data_list."""
    delete_by_index = list(delete_by_index or [])
    delete_by_name = list(delete_by_name or [])

    if not delete_by_index and not delete_by_name:
        return data_list, variable_mapping, []

    print("\nDeleting variables per configuration...")
    if delete_by_name:
        name_to_index = dict(zip(variable_mapping["variable_name"], variable_mapping["index"]))
        for name in delete_by_name:
            if name in name_to_index:
                delete_by_index.append(int(name_to_index[name]))
            else:
                print(f"Warning: variable name {name!r} not in mapping; skipped.")

    delete_by_index = sorted(set(int(i) for i in delete_by_index))

    first_df = data_list[0]
    cols_to_drop = [first_df.columns[i] for i in delete_by_index if 0 <= i < first_df.shape[1]]

    if not cols_to_drop:
        print("No columns matched index/name list; nothing removed.")
        return data_list, variable_mapping, []

    print("Columns to drop:", cols_to_drop)

    new_data_list = [df.drop(columns=cols_to_drop) for df in data_list]
    new_variable_mapping = build_variable_catalog(new_data_list[0])
    print("\nUpdated variable mapping:")
    print(new_variable_mapping.to_string(index=False))
    print(
        f"Remaining variables: {len(new_variable_mapping)}; "
        f"shape: {new_data_list[0].shape[0]} rows × {new_data_list[0].shape[1]} columns"
    )

    return new_data_list, new_variable_mapping, cols_to_drop

def scale_numeric_features(data: pd.DataFrame, scaler_type: str) -> pd.DataFrame:
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    X = data[numeric_cols]
    if scaler_type is None or str(scaler_type).strip().lower() == "none":
        return X.copy()
    if str(scaler_type).strip() == "LeadingDigit":
        out = X.copy()
        for col in numeric_cols:
            arr = out[col].values.astype(float)
            M = np.nanmax(np.abs(arr))
            if M == 0 or not np.isfinite(M):
                continue
            exp = np.floor(np.log10(M))
            scale = np.power(10.0, exp)
            out[col] = out[col] / scale
        return out
    if str(scaler_type).strip() == "LeadingDigit_byrow":
        arr = X.values.astype(float)
        out_arr = arr.copy()
        for i in range(arr.shape[0]):
            row = arr[i, :]
            M = np.nanmax(np.abs(row))
            if M == 0 or not np.isfinite(M):
                continue
            exp = np.floor(np.log10(M))
            scale = np.power(10.0, exp)
            out_arr[i, :] = row / scale
        return pd.DataFrame(out_arr, index=data.index, columns=numeric_cols)
    if str(scaler_type).strip() == "LeadingDigitMinShift":
        X_ld = scale_numeric_features(X.copy(), "LeadingDigit")
        arr = X_ld.values
        global_min = np.nanmin(arr)
        shift = np.abs(global_min) if np.isfinite(global_min) else 0.0
        out = X_ld + shift
        return out
    scaler_map = {
        'MinMaxScaler': MinMaxScaler(),
        'StandardScaler': StandardScaler(),
        'MaxAbsScaler': MaxAbsScaler(),
    }
    scaler = scaler_map[scaler_type]
    scaled_values = scaler.fit_transform(X)
    return pd.DataFrame(scaled_values, index=data.index, columns=numeric_cols)
