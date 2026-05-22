import os
import warnings
from typing import Optional, Sequence
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator, LinearLocator, FuncFormatter
from scipy.interpolate import UnivariateSpline

# Matplotlib font settings (legacy CJK fonts; minus sign fix)
rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

def _log_axis_formatter_ascii(value, pos):
    """Log-axis ticks as ASCII 10^n to avoid missing-glyph boxes."""
    if value <= 0 or not np.isfinite(value):
        return ""
    exp = int(round(np.log10(value)))
    return "10^{}".format(exp)

def _log_axis_formatter_mathtext(value, pos):
    """Log-axis mathtext superscripts; dejavusans avoids minus-sign boxes."""
    if value <= 0 or not np.isfinite(value):
        return ""
    exp = int(round(np.log10(value)))
    return r"$10^{%d}$" % exp

def smooth_columns_spline(data: pd.DataFrame, s: float = 0.05) -> pd.DataFrame:
    """
    Spline smoothing for denoising.

    Parameters:
    -----------
    data : DataFrame
        Input data (index = time)
    s : float
        Smoothing factor; smaller = closer to data

    Returns:
    --------
    smoothed_data : DataFrame
        Smoothed data
    """
    times = data.index.values.astype(float)
    smoothed = pd.DataFrame(index=data.index, columns=data.columns)

    for col in data.columns:
        y = data[col].values.astype(float)
        spl = UnivariateSpline(times, y, s=s)
        smoothed[col] = spl(times)

    return smoothed

def build_quasi_dynamic_frame(data: pd.DataFrame) -> pd.DataFrame:
    data = data[~data.index.duplicated(keep="first")]
    row_sum_sorted = data.sum(axis=1).sort_values()
    qd_df = data.loc[row_sum_sorted.index].copy()
    qd_df.index = pd.Index(row_sum_sorted.values)
    return qd_df

def eval_power_law(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * np.power(x, b)

def fit_power_law_params(data: pd.DataFrame) -> pd.DataFrame:
    x = data.index.values.astype(float)
    results = {}
    for col in data.columns:
        y = data[col].values.astype(float)
        mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
        x_log, y_log = np.log(x[mask]), np.log(y[mask])
        slope, intercept, *_ = stats.linregress(x_log, y_log)
        a, b = np.exp(intercept), slope
        results[col] = [a, b]
    return pd.DataFrame(results, index=["a", "b"]).T

def sample_power_law_curve(data: pd.DataFrame, sample_method: str = "linspace", num_points: int = 1000) -> pd.DataFrame:
    x = np.asarray(data.index, dtype=float) if sample_method == "by_index" else np.linspace(float(np.min(data.index)), float(np.max(data.index)), num_points)
    results = {}
    for col in data.columns:
        a, b = fit_power_law_params(data).loc[col, ["a", "b"]]
        results[col] = eval_power_law(x, a, b)
    return pd.DataFrame(results, index=x)

def eval_logistic_curve(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return c / (1 + np.exp(-b * (x - a)))

def fit_logistic_params(data: pd.DataFrame) -> pd.DataFrame:
    x = data.index.values.astype(float)
    results = {}
    for col in data.columns:
        y = data[col].values.astype(float)
        mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
        x_log, y_log = np.log(x[mask]), np.log(y[mask])
        slope, intercept, *_ = stats.linregress(x_log, y_log)
        a, b, c = intercept, slope, 1
        results[col] = [a, b, c]
    return pd.DataFrame(results, index=["a", "b", "c"]).T

def sample_logistic_curve(data: pd.DataFrame, sample_method: str = "linspace", num_points: int = 1000) -> pd.DataFrame:
    x = np.asarray(data.index, dtype=float) if sample_method == "by_index" else np.linspace(float(np.min(data.index)), float(np.max(data.index)), num_points)
    results = {}
    for col in data.columns:
        a, b, c = fit_logistic_params(data).loc[col, ["a", "b", "c"]]
        results[col] = eval_logistic_curve(x, a, b, c)
    return pd.DataFrame(results, index=x)

def fit_curve_params(data: pd.DataFrame, curve_function: str = "eval_power_law") -> pd.DataFrame:
    if curve_function == "eval_power_law":
        curve_params = fit_power_law_params(data)
    elif curve_function == "eval_logistic_curve":
        curve_params = fit_logistic_params(data)
    return curve_params

def sample_fitted_curve(data: pd.DataFrame, curve_function: str = "eval_power_law", sample_method: str = "by_index", num_points: int = 1000) -> pd.DataFrame:
    if curve_function == "eval_power_law":
        curve_sample = sample_power_law_curve(data, sample_method, num_points)
    elif curve_function == "eval_logistic_curve":
        curve_sample = sample_logistic_curve(data, sample_method, num_points)
    return curve_sample

_GOLDEN_RATIO = (1.0 + np.sqrt(5.0)) / 2.0

def plot_quasi_dynamic_curve_fits(
    data: pd.DataFrame | list[pd.DataFrame],
    curve_function: str = "eval_power_law",
    sample_method: str = "by_index",
    num_points: int = 1000,
    save_path: str | None = None,
    dpi: int = 300,
    group_labels: list[str] | None = None,
    group_palette_indices: list[int] | None = None,
    first_n: int = 9,
    n_cols: int = 3,
    use_log_y: bool = True,
    column_labels: Optional[dict] = None,
    only_columns: Optional[Sequence[str]] = None,
    stack_vertically: bool = False,
    panel_aspect: Optional[str] = None,
    row_height_inches: float = 2.85,
    grid_panel_aspect: Optional[str] = "golden",
    grid_row_height_inches: float = 2.35,
) -> plt.Figure:
        """
        Quasi-dynamic scatter + curve-fit plots.

        Default log-y; pass use_log_y=False for linear y.

        data: one DataFrame or list of DataFrames; multiple groups overlay per subplot.
        group_labels optional. first_n=None plots all variables.
        only_columns: subset of shared columns; takes precedence over first_n.
        group_palette_indices: palette index per group for consistent colors.
        column_labels: variable_name -> display label for subplot titles.
        stack_vertically: single column of subplots (ignores n_cols).
        panel_aspect with stack_vertically: "square", "6:4", or "golden".
        row_height_inches: row height when stacked vertically.
        grid_panel_aspect: aspect per subplot in grid mode.
        grid_row_height_inches: row height in grid mode.
        """
        if isinstance(data, pd.DataFrame):
            data = [data]
        data = [pd.DataFrame(d) if not isinstance(d, pd.DataFrame) else d for d in data]
        obs_list, fit_list = [], []
        for d in data:
            build_quasi_dynamic_frame_df = build_quasi_dynamic_frame(d)
            obs_list.append(build_quasi_dynamic_frame_df)
            fit_list.append(sample_fitted_curve(data=build_quasi_dynamic_frame_df,
                                             curve_function=curve_function,
                                             sample_method=sample_method,
                                             num_points=num_points))
        common_cols = list(obs_list[0].columns)
        for obs in obs_list[1:]:
            common_cols = [c for c in common_cols if c in obs.columns]
        if only_columns is not None:
            common_set = set(common_cols)
            ordered = [c for c in only_columns if c in common_set]
            missing = [c for c in only_columns if c not in common_set]
            if missing:
                warnings.warn(
                    "only_columns not in shared columns, skipped: {}".format(missing),
                    stacklevel=2,
                )
            if not ordered:
                raise ValueError("only_columns has no shared columns across groups; cannot plot.")
            common_cols = ordered
        elif first_n is not None:
            common_cols = common_cols[:first_n]
        n_groups = len(obs_list)
        if group_palette_indices is not None:
            if len(group_palette_indices) != n_groups:
                raise ValueError(
                    "group_palette_indices length must match number of groups."
                )
        palette = [
            ("#FAD7A0", "#D35400"), # orange
            ("#AED6F1", "#2E86C1"), # blue
            ("#A9DFBF", "#239B56"), # green
            ("#F5B7B1", "#C0392B"), # red
            ("#D7BDE2", "#8E44AD"), # purple
            ("#A3E4D7", "#16A085"), # teal
            ("#F8C471", "#E74C3C"), # pink
            ("#D5DBDB", "#566573"), # gray
            ("#FCF3CF", "#F1C40F"), # yellow
            ("#E6B0AA", "#7B241C"), # brown
        ]
        wh_panel = 6.0 / 4.0
        wh_grid: Optional[float] = None
        if stack_vertically:
            ncols = 1
            nrows = len(common_cols)
            pa = panel_aspect or "square"
            if pa not in ("square", "6:4", "golden"):
                raise ValueError('panel_aspect must be "square", "6:4", or "golden".')
            if pa == "square":
                wh_panel = 1.0
            elif pa == "6:4":
                wh_panel = 6.0 / 4.0
            else:
                wh_panel = _GOLDEN_RATIO
            rh = float(row_height_inches)
            fig_w = rh * wh_panel
            fig_h = rh * nrows + 0.48
            figsize = (fig_w, fig_h)
            subplots_kw = dict(hspace=0.035)
        else:
            ncols = n_cols
            nrows = -(-len(common_cols) // ncols)
            gpa = grid_panel_aspect
            if gpa is None:
                figsize = (4 * 2.5, 3 * 2)
            else:
                if gpa not in ("square", "6:4", "golden"):
                    raise ValueError(
                        'grid_panel_aspect must be None, "square", "6:4", or "golden".'
                    )
                if gpa == "square":
                    wh_grid = 1.0
                elif gpa == "6:4":
                    wh_grid = 6.0 / 4.0
                else:
                    wh_grid = _GOLDEN_RATIO
                gh = float(grid_row_height_inches)
                gw = gh * wh_grid
                figsize = (ncols * gw, nrows * gh + 1.05)
            subplots_kw = dict(hspace=0)
        if use_log_y:
            _saved_fontset = rcParams.get("mathtext.fontset", None)
            rcParams["mathtext.fontset"] = "dejavusans"
        fig, axes = plt.subplots(
            nrows, ncols, figsize=figsize, sharex=True, sharey=True
        )
        axes = np.atleast_1d(axes).ravel()
        for i, col in enumerate(common_cols):
            ax = axes[i]
            if stack_vertically:
                ax.set_box_aspect(1.0 / wh_panel)
            elif wh_grid is not None:
                ax.set_box_aspect(1.0 / wh_grid)
            for g in range(n_groups):
                obs, fit = obs_list[g], fit_list[g]
                if col not in obs.columns:
                    continue
                pidx = (
                    group_palette_indices[g]
                    if group_palette_indices is not None
                    else g
                )
                sc, lc = palette[pidx % len(palette)]
                if use_log_y:
                    ax.semilogy(
                        obs.index,
                        obs[col],
                        "o",
                        markerfacecolor="none",
                        markeredgecolor=sc,
                        markersize=5,
                        alpha=0.8,
                        zorder=1,
                    )
                    ax.semilogy(fit.index, fit[col], color=lc, linewidth=2, zorder=2)
                else:
                    ax.plot(
                        obs.index,
                        obs[col],
                        "o",
                        markerfacecolor="none",
                        markeredgecolor=sc,
                        markersize=5,
                        alpha=0.8,
                        zorder=1,
                    )
                    ax.plot(fit.index, fit[col], color=lc, linewidth=2, zorder=2)
            if use_log_y:
                ax.yaxis.set_major_formatter(FuncFormatter(_log_axis_formatter_mathtext))
            ax.tick_params(axis="y", labelsize=8)
            title_text = (column_labels.get(col, col) if column_labels else col)
            ax.text(0.03, 0.97, title_text, transform=ax.transAxes, fontsize=10, va='top', ha='left', fontweight='bold',
                    bbox=dict(facecolor='none', alpha=0.0, edgecolor='none', boxstyle='round,pad=0.2'))
            ax.margins(x=0.2, y=0.2)
        for ax in axes[len(common_cols) :]:
            fig.delaxes(ax)
        labels = group_labels if group_labels is not None else [f"Group {g}" for g in range(n_groups)]
        handles = [
            plt.Line2D(
                [],
                [],
                marker='o',
                ls='',
                color=palette[
                    (
                        group_palette_indices[g]
                        if group_palette_indices is not None
                        else g
                    ) % len(palette)
                ][0],
                markersize=6,
                label=labels[g],
            )
            for g in range(n_groups)
        ]
        y_label = "Value (log scale)" if use_log_y else "Value"
        if stack_vertically:
            _sv_left, _sv_right = 0.20, 0.97
            _sv_top, _sv_bottom = 0.916, 0.09
            fig.subplots_adjust(
                wspace=0,
                hspace=subplots_kw.get("hspace", 0),
                bottom=_sv_bottom,
                top=_sv_top,
                left=_sv_left,
                right=_sv_right,
            )
            fig.text(0.038, 0.5, y_label, va="center", rotation="vertical", fontsize=14)
            _legend_x = (_sv_left + _sv_right) / 2.0
            fig.legend(
                handles=handles,
                loc="lower center",
                ncol=n_groups,
                fontsize=12,
                bbox_to_anchor=(_legend_x, _sv_top + 0.0002),
                frameon=False,
            )
            fig.text(0.5, 0.03, "Quasi Dynamic Index (EI)", ha="center", fontsize=13)
        else:
            fig.subplots_adjust(
                wspace=0,
                hspace=0,
                bottom=0.10,
                top=0.94,
                left=0.12,
                right=0.98,
            )
            fig.text(0.06, 0.5, y_label, va="center", rotation="vertical", fontsize=14)
            fig.text(0.5, 0.04, "Quasi Dynamic Index (EI)", ha="center", fontsize=13)
            axes[0].legend(
                handles=handles,
                loc="lower right",
                fontsize=11,
                frameon=False,
                ncol=1,
            )
        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        return fig

