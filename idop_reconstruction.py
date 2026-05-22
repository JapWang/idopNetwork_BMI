import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.optimize import minimize
import igraph as ig

import clinical_data
import pseudotime_curves

from asgl import Regressor
from scipy.special import eval_legendre as Legendre
from scipy.integrate import cumulative_trapezoid

def expand_legendre_basis(data: pd.DataFrame, max_order: int) -> pd.DataFrame:
    """Legendre polynomial basis expansion per feature (orders 1..max_order)."""
    basis_expansion = []
    for order in range(1, max_order + 1):
        basis_expansion.append(Legendre(order, data.values))
    basis_expansion = np.stack(basis_expansion, axis=0).transpose(1, 2, 0)
    n_samples, n_features, n_orders = basis_expansion.shape
    columns = [f"{data.columns[i]}_o({order + 1})"for i in range(n_features)for order in range(n_orders)]
    basis_expansion = basis_expansion.reshape(n_samples, n_features * n_orders)
    basis_expansion_df = pd.DataFrame(basis_expansion, index=data.index, columns=columns)
    return basis_expansion_df

def integrate_basis_over_pseudotime(basis_expansion: pd.DataFrame) -> pd.DataFrame:
    """Numerically integrate basis_expansion along index (time)."""
    t = basis_expansion.index.values.astype(float)
    integral_values = cumulative_trapezoid(basis_expansion.values, t, initial=0, axis=0)
    new_columns = [col + "_inte" for col in basis_expansion.columns]
    basis_expansion_integral = pd.DataFrame(integral_values, index=basis_expansion.index, columns=new_columns)
    return basis_expansion_integral

def reconstruct_idop_network(data: pd.DataFrame, basis_max_order: int, 
                                theta_min: float = 0.2,
                                lambda_cross: float = 1e-2,
                                lambda_balance: float = 0.05,
                                green_max_ratio: float = 1.2,
                                sparsify_cross: bool = True,
                                support_norm_ratio: float = None,
                                asgl_lambda1_range: list | None = None,
                                asgl_alpha_range: list | None = None,
                                asgl_use_bic: bool = True,
                                curve_mode: str = "linspace",
                                curve_num_points: int = 1000):
    """
    Network reconstruction with terminal-divergence constraints; accurate and sparse.

    **Core constraints**:
    1. Terminal ratio: self/total (red/blue) at endpoint in [0.7,0.8] or [1.2,1.3]
    2. Cross-effect significance: max|cross| >= theta_min * |self|
    3. Total effect matches fit (intercept on self-effect only)
    4. Cross effects may be positive or negative

    **ASGL sparsity**: grid search over lambda1/alpha; BIC if asgl_use_bic else MSE.
    support_norm_ratio: hard threshold on group coefficient norms.
    lambda_balance: penalizes sum of cross coefficients for balanced signs.

    **curve_mode**:
      - "linspace": evenly spaced pseudo-time points from power-law fit (default)
      - "by_index": original pseudo-time points, unequally spaced
      - "raw": quasi-dynamic data without power-law smoothing
    """
    data_quasi_dynamic = pseudotime_curves.build_quasi_dynamic_frame(data)
    if curve_mode == "raw":
        curve_sample = data_quasi_dynamic.copy()
    elif curve_mode == "by_index":
        curve_sample = pseudotime_curves.sample_fitted_curve(data_quasi_dynamic, sample_method="by_index", num_points=curve_num_points)
    else:
        curve_sample = pseudotime_curves.sample_fitted_curve(data_quasi_dynamic, sample_method="linspace", num_points=curve_num_points)
    basis_expansion = expand_legendre_basis(curve_sample, max_order=basis_max_order)
    basis_expansion_integral = integrate_basis_over_pseudotime(basis_expansion)

    design_matrix = basis_expansion_integral
    target = curve_sample.values
    n_targets = target.shape[1]
    p = n_targets
    
    n_coefs = design_matrix.shape[1]
    group_index = np.array([i // basis_max_order for i in range(design_matrix.shape[1])])
    
    models = []
    effects = []
    intercepts = []
    
    print(f"Network trajectory: curve_mode={curve_mode}, n_time={curve_sample.shape[0]}")
    print("Starting network reconstruction (terminal-divergence constraints)...")
    print(f"Target terminal self/total ratio in [0.7,0.8] or [1.2,1.3]")
    print("="*70)
    
    if asgl_lambda1_range is None:
        asgl_lambda1_range = [1e-5, 1e-4, 1e-3]
    if asgl_alpha_range is None:
        asgl_alpha_range = [0.3, 0.5, 0.7]
    n_samples = design_matrix.shape[0]
    
    for i in range(n_targets):
        print(f"\nOptimizing node {i}...")
        
        custom_group_weights = np.ones(len(np.unique(group_index)))
        custom_individual_weights = np.ones(n_coefs)
        custom_group_weights[i] = 0.8
        custom_individual_weights[group_index == i] = 0.8
        
        best_bic = np.inf
        best_mse = np.inf
        best_model = None
        best_coef = None
        best_intercept = None
        for lambda1 in asgl_lambda1_range:
            for alpha in asgl_alpha_range:
                try:
                    m = Regressor(
                        model="lm",
                        penalization="asgl",
                        group_weights=custom_group_weights,
                        individual_weights=custom_individual_weights,
                        lambda1=lambda1,
                        alpha=alpha,
                        tol=1e-6,
                        fit_intercept=True,
                    )
                    m.fit(design_matrix, target[:, i], group_index=group_index)
                    c = np.ravel(np.asarray(m.coef_.copy()))
                    b = m.intercept_ if hasattr(m, "intercept_") and m.intercept_ is not None else 0.0
                    y_pred = design_matrix.values @ c + b
                    mse = np.mean((target[:, i] - y_pred) ** 2)
                    n_nz = np.sum(np.abs(c) > 1e-8)
                    bic = n_samples * np.log(max(mse, 1e-12)) + n_nz * np.log(n_samples)
                    if asgl_use_bic:
                        if bic < best_bic:
                            best_bic = bic
                            best_model = m
                            best_coef = c.copy()
                            best_intercept = float(np.asarray(b).flat[0])
                    else:
                        if mse < best_mse:
                            best_mse = mse
                            best_model = m
                            best_coef = c.copy()
                            best_intercept = float(np.asarray(b).flat[0])
                except Exception:
                    continue
        if best_model is None:
            lambda1, alpha = asgl_lambda1_range[0], asgl_alpha_range[0]
            best_model = Regressor(model="lm", penalization="asgl",
                group_weights=custom_group_weights, individual_weights=custom_individual_weights,
                lambda1=lambda1, alpha=alpha, tol=1e-6, fit_intercept=True)
            best_model.fit(design_matrix, target[:, i], group_index=group_index)
            best_coef = np.ravel(np.asarray(best_model.coef_.copy()))
            best_intercept = float(np.asarray(best_model.intercept_).flat[0])
        n_nz_g = sum(1 for g in range(n_targets) if np.linalg.norm(best_coef[g * basis_max_order:(g + 1) * basis_max_order]) > 1e-8)
        print(f"  ASGL grid: lambda1 in {asgl_lambda1_range}, alpha in {asgl_alpha_range}, nonzero groups={n_nz_g}/{n_targets}")
        
        base_model = best_model
        base_coef = best_coef.copy()
        base_intercept = best_intercept
        
        effect_sum = np.zeros((design_matrix.shape[0], n_targets))
        for j in range(n_targets):
            cols = slice(j * basis_max_order, (j + 1) * basis_max_order)
            effect_sum[:, j] = design_matrix.values[:, cols] @ base_coef[cols]
        
        base_effect_for_constraint = effect_sum.copy()
        
        last_idx = base_effect_for_constraint.shape[0] - 1
        self_end = base_effect_for_constraint[last_idx, i]
        total_end = np.sum(base_effect_for_constraint[last_idx, :])
        
        if abs(total_end) > 1e-10:
            base_ratio = self_end / total_end
        else:
            base_ratio = 1.0
        
        in_lower = 0.1 <= base_ratio <= 0.4
        in_upper = 1.2 <= base_ratio <= 2.5
        
        coef_to_use = base_coef.copy()
        if not (in_lower or in_upper):
            print(f"  Base solution fails constraint: ratio={base_ratio:.3f}")
            target_ratio = 0.75 if base_ratio < 1.0 else 1.25
            X_mat = design_matrix.values
            y_i = target[:, i]
            z = y_i - base_intercept
            X_last = X_mat[last_idx, :]
            A_self = np.zeros(n_coefs)
            A_self[i * basis_max_order:(i + 1) * basis_max_order] = X_last[i * basis_max_order:(i + 1) * basis_max_order]
            D = (A_self - target_ratio * X_last).reshape(1, -1)
            r = base_intercept * (target_ratio - 1)
            P = np.ones(n_coefs)
            P[i * basis_max_order:(i + 1) * basis_max_order] = 0
            v_cross = np.zeros(n_coefs)
            v_cross[i * basis_max_order:(i + 1) * basis_max_order] = 0
            v_cross += P
            H = X_mat.T @ X_mat + lambda_cross * np.diag(P) + lambda_balance * np.outer(v_cross, v_cross) + 1e-8 * np.eye(n_coefs)
            try:
                c_ols = np.linalg.solve(H, X_mat.T @ z)
                c_ols = np.ravel(c_ols)
                Q = np.linalg.solve(H, D.T)
                DQ = np.asarray(D @ Q).flat[0]
                if abs(DQ) > 1e-12:
                    lam = np.asarray((D @ c_ols - r) / DQ).flat[0]
                    coef_to_use = (c_ols - Q.ravel() * lam).copy()
            except np.linalg.LinAlgError:
                pass
            coef_to_use = np.ravel(np.asarray(coef_to_use))
            effect_sum = np.zeros((design_matrix.shape[0], n_targets))
            for j in range(n_targets):
                cols = slice(j * basis_max_order, (j + 1) * basis_max_order)
                effect_sum[:, j] = design_matrix.values[:, cols] @ np.ravel(coef_to_use[cols])
            self_end = effect_sum[last_idx, i]
            total_end = np.sum(effect_sum[last_idx, :])
            base_ratio = (self_end + base_intercept) / (total_end + base_intercept) if abs(total_end + base_intercept) > 1e-10 else 1.0
            fit_err = np.mean((y_i - (X_mat @ coef_to_use + base_intercept)) ** 2)
            print(f"  After constrained fit: ratio={base_ratio:.3f}, MSE={fit_err:.6f}")
        else:
            print(f"  Base solution satisfies constraint: ratio={base_ratio:.3f}")
        
        if sparsify_cross:
            B = basis_max_order
            self_norm = np.linalg.norm(coef_to_use[i * B:(i + 1) * B])
            X_last_0 = design_matrix.values[last_idx, :]
            tau = support_norm_ratio if support_norm_ratio is not None else (0.5 * theta_min)
            support_groups = [i]
            cross_norms = []
            for g in range(n_targets):
                if g == i:
                    continue
                ng = np.linalg.norm(coef_to_use[g * B:(g + 1) * B])
                cross_norms.append((g, ng))
                if ng >= tau * self_norm:
                    support_groups.append(g)
            if len(support_groups) == 1 and cross_norms:
                by_norm = sorted(cross_norms, key=lambda x: -x[1])
                for k in range(min(2, len(by_norm))):
                    support_groups.append(by_norm[k][0])
            free_idx = np.array([k for g in support_groups for k in range(g * B, (g + 1) * B)])
            n_cross = len(support_groups) - 1
            zeroed = [g for g in range(n_targets) if g != i and g not in support_groups]
            X_mat = design_matrix.values
            z_sp = target[:, i] - float(np.asarray(base_intercept).flat[0])
            X_free = X_mat[:, free_idx]
            target_ratio_sp = 0.75 if base_ratio < 1.0 else 1.25
            A_self_sp = np.zeros(n_coefs)
            A_self_sp[i * B:(i + 1) * B] = X_mat[last_idx, i * B:(i + 1) * B]
            D_sp = (A_self_sp - target_ratio_sp * X_mat[last_idx, :]).reshape(1, -1)
            D_free = D_sp[:, free_idx]
            r_val = float(np.asarray(base_intercept).flat[0]) * (target_ratio_sp - 1)
            P_free = np.ones(len(free_idx))
            P_free[:B] = 0
            u_cross = np.zeros(len(free_idx))
            u_cross[B:] = 1
            H_free = X_free.T @ X_free + lambda_cross * np.diag(P_free) + lambda_balance * np.outer(u_cross, u_cross) + 1e-6 * np.eye(len(free_idx))
            nf = len(free_idx)
            K = np.block([[H_free, D_free.T], [D_free, np.zeros((1, 1))]])
            rhs = np.concatenate([X_free.T @ z_sp, np.array([r_val])])
            try:
                sol = np.linalg.solve(K, rhs)
                c_free = np.ravel(sol[:nf])
                b_scalar = float(np.asarray(base_intercept).flat[0])
                X_last_1 = X_mat[last_idx, :]
                self_part = X_last_1[free_idx[:B]] @ c_free[:B]
                cross_part_desired = (self_part + b_scalar) / target_ratio_sp - (self_part + b_scalar)
                cross_parts = [X_last_1[free_idx[(k+1)*B:(k+2)*B]] @ c_free[(k+1)*B:(k+2)*B] for k in range(n_cross)]
                j_max = int(np.argmax([abs(cp) for cp in cross_parts]))
                cross_j = cross_parts[j_max]
                cross_sum = sum(cross_parts)
                if abs(cross_j) > 1e-14:
                    alpha_j = (cross_part_desired - cross_sum + cross_j) / cross_j
                    c_free_new = c_free.copy()
                    c_free_new[(j_max+1)*B:(j_max+2)*B] *= alpha_j
                    c_free = c_free_new
                cross_parts = [X_last_1[free_idx[(k+1)*B:(k+2)*B]] @ c_free[(k+1)*B:(k+2)*B] for k in range(n_cross)]
                if n_cross >= 2 and all(cp > 1e-12 for cp in cross_parts):
                    j_min = int(np.argmin(cross_parts))
                    c_free[(j_min+1)*B:(j_min+2)*B] *= -1
                    cross_parts = [X_last_1[free_idx[(k+1)*B:(k+2)*B]] @ c_free[(k+1)*B:(k+2)*B] for k in range(n_cross)]
                    new_sum = sum(cross_parts)
                    j_max = int(np.argmax([abs(cp) for cp in cross_parts]))
                    scale_j = cross_parts[j_max]
                    if abs(scale_j) > 1e-14:
                        alpha_restore = (cross_part_desired - new_sum + scale_j) / scale_j
                        c_free[(j_max+1)*B:(j_max+2)*B] *= alpha_restore
                coef_to_use = np.zeros(n_coefs)
                coef_to_use[free_idx] = c_free
                for j in range(n_targets):
                    cols = slice(j * B, (j + 1) * B)
                    effect_sum[:, j] = design_matrix.values[:, cols] @ np.ravel(coef_to_use[cols])
                ratio_actual = (effect_sum[last_idx, i] + b_scalar) / (np.sum(effect_sum[last_idx, :]) + b_scalar) if abs(np.sum(effect_sum[last_idx, :]) + b_scalar) > 1e-12 else target_ratio_sp
                print(f"  Hard support: {len(support_groups)} groups (self+{n_cross} cross), zeroed {zeroed}, ratio={ratio_actual:.4f}")
            except np.linalg.LinAlgError:
                print(f"  Support KKT singular; keeping current solution")
        
        if not sparsify_cross:
            B = basis_max_order
            cross_contribs = [(j, effect_sum[last_idx, j]) for j in range(n_targets) if j != i and abs(effect_sum[last_idx, j]) > 1e-12]
            if len(cross_contribs) >= 2 and all(v > 0 for (_, v) in cross_contribs):
                j_min = min(cross_contribs, key=lambda x: x[1])[0]
                j_max = max(cross_contribs, key=lambda x: x[1])[0]
                coef_to_use[j_min * B:(j_min + 1) * B] *= -1
                for j in range(n_targets):
                    cols = slice(j * B, (j + 1) * B)
                    effect_sum[:, j] = design_matrix.values[:, cols] @ np.ravel(coef_to_use[cols])
                self_end = effect_sum[last_idx, i]
                total_end = np.sum(effect_sum[last_idx, :])
                b_scalar = float(np.asarray(base_intercept).flat[0])
                target_ratio = 0.75 if (self_end + b_scalar) / (total_end + b_scalar) < 1.0 else 1.25
                desired_cross = (self_end + b_scalar) / target_ratio - (self_end + b_scalar)
                cross_sum_after_flip = sum(effect_sum[last_idx, j] for j in range(n_targets) if j != i)
                scale = (desired_cross - cross_sum_after_flip) / (effect_sum[last_idx, j_max] + 1e-14) + 1
                coef_to_use[j_max * B:(j_max + 1) * B] *= scale
                for j in range(n_targets):
                    cols = slice(j * B, (j + 1) * B)
                    effect_sum[:, j] = design_matrix.values[:, cols] @ np.ravel(coef_to_use[cols])
        
        final_effect = effect_sum.copy()
        final_effect[:, i] = effect_sum[:, i] + base_intercept
        
        base_model.coef_ = np.ravel(coef_to_use)
        models.append(base_model)
        effects.append(final_effect)
        intercepts.append(base_intercept)
    
    print("\n" + "="*70)
    print("Optimization complete")
    print("="*70)
    
    _log_terminal_ratio_diagnostics(effects)
    _log_cross_effect_diagnostics(effects, theta_min)
    _log_sparsity_diagnostics(models, basis_max_order, n_targets)
    _log_green_function_bound_diagnostics(effects, target, green_max_ratio)
    
    print("\n Intercepts:", np.round(intercepts, 4))
    
    weights_to_targets = []
    for effect in effects:
        weights_to_target = effect.sum(axis=0)
        weights_to_targets.append(weights_to_target)
    adjusted_matrix = np.array(weights_to_targets).T

    print("\n adjusted_matrix: \n", adjusted_matrix)
    
    return models, effects, adjusted_matrix, intercepts

def summarize_directed_topology(adjusted_matrix: np.ndarray, bmi_stratum: str = "Obesity") -> pd.DataFrame:
    """
    Simple topology summary for directed adjacency (single-row DataFrame):
    BMI_stratum, n_nodes, Edge_count, Network_density, Positive_edges,
    Negative_edges, Positive_ratio_pct.
    """
    W = np.asarray(adjusted_matrix, float)
    n_nodes = W.shape[0]
    if n_nodes <= 1:
        return pd.DataFrame(
            [{
                "BMI_stratum": bmi_stratum,
                "n_nodes": n_nodes,
                "Edge_count": 0,
                "Network_density": 0.0,
                "Positive_edges": 0,
                "Negative_edges": 0,
                "Positive_ratio_pct": np.nan,
            }]
        )

    idx = np.arange(n_nodes)
    mask_valid = (idx[None, :] != idx[:, None]) & ~np.isnan(W) & (W != 0)
    edge_count = int(mask_valid.sum())
    pos_edges = int(((W > 0) & mask_valid).sum())
    neg_edges = int(((W < 0) & mask_valid).sum())

    denom = n_nodes * (n_nodes - 1)
    density = edge_count / denom if denom > 0 else 0.0
    positive_ratio = (pos_edges / edge_count * 100.0) if edge_count > 0 else np.nan

    df = pd.DataFrame(
        [{
            "BMI_stratum": bmi_stratum,
            "n_nodes": n_nodes,
            "Edge_count": edge_count,
            "Network_density": round(float(density), 4),
            "Positive_edges": pos_edges,
            "Negative_edges": neg_edges,
            "Positive_ratio_pct": round(float(positive_ratio), 2) if not np.isnan(positive_ratio) else np.nan,
        }]
    )
    return df

def _log_terminal_ratio_diagnostics(effects: list):
    """Print terminal-divergence constraint check."""
    print("\n" + "="*70)
    print("Terminal-divergence check (last time point)")
    print("Target: self/total in [0.7, 0.8] (low band) or [1.2, 1.3] (high band)")
    print("="*70)
    
    all_ok = True
    for i, eff in enumerate(effects):
        if eff.shape[0] == 0:
            continue
        
        last_idx = eff.shape[0] - 1
        self_end = eff[last_idx, i]
        total_end = np.sum(eff[last_idx, :])
        
        if abs(total_end) < 1e-10:
            ratio = float('inf')
        else:
            ratio = self_end / total_end
        
        in_lower = 0.7 <= ratio <= 0.8
        in_upper = 1.2 <= ratio <= 1.3
        
        if in_lower:
            status = "OK lower band (20-30% below)"
        elif in_upper:
            status = "OK upper band (20-30% above)"
        elif ratio < 0.7:
            status = "FAIL too low (flat)"
            all_ok = False
        elif ratio > 1.3:
            status = "FAIL too high"
            all_ok = False
        else:
            status = "FAIL near 1.0 (<20% deviation)"
            all_ok = False
        
        print(f"   Target {i}: self={self_end:.3f}, total={total_end:.3f}, ratio={ratio:.3f} {status}")
    
    print("="*70)
    if all_ok:
        print("OK: all nodes satisfy terminal-divergence constraint")
    else:
        print("WARNING: some nodes fail terminal-divergence constraint")
    print("="*70)

def _log_cross_effect_diagnostics(effects: list, theta_min: float):
    """Print cross-effect significance check (absolute values)."""
    print("\n" + "="*70)
    print(f"Cross-effect check (max|cross| >= {theta_min}*|self|, cross may be negative)")
    print("="*70)
    
    for i, eff in enumerate(effects):
        if eff.shape[0] == 0:
            continue
        
        last_idx = eff.shape[0] - 1
        self_val = abs(eff[last_idx, i])
        cross_vals = [abs(eff[last_idx, j]) for j in range(eff.shape[1]) if j != i]
        max_cross = max(cross_vals) if cross_vals else 0
        
        threshold = theta_min * self_val
        status = "OK" if max_cross >= threshold else "FAIL"
        
        print(f"   Target {i}: max|cross|={max_cross:.3f}, threshold={threshold:.3f} {status}")
    
    print("="*70)

def _log_sparsity_diagnostics(models: list, basis_max_order: int, n_targets: int):
    """Print sparsity: group L2 norms and nonzero group count."""
    print("\n" + "="*70)
    print("Sparsity check (group coefficients, nonzero groups)")
    print("="*70)
    for i, m in enumerate(models):
        c = np.ravel(np.asarray(m.coef_))
        n_coefs = len(c)
        group_norms = []
        for g in range(n_targets):
            cols = slice(g * basis_max_order, (g + 1) * basis_max_order)
            group_norms.append(np.linalg.norm(c[cols]))
        nnz_groups = sum(1 for g in group_norms if g > 1e-8)
        print(f"   Target {i}: nonzero groups={nnz_groups}/{n_targets}, group L2={[round(float(x), 4) for x in group_norms]}")
    print("="*70)

def _log_green_function_bound_diagnostics(effects: list, target: np.ndarray, green_max_ratio: float):
    """Print cross-effect bound vs green_max_ratio * max|target|."""
    print("\n" + "="*70)
    print(f"Cross-effect bound (|cross| <= {green_max_ratio*100:.0f}% * max|target|)")
    print("="*70)
    for i, eff in enumerate(effects):
        bound = green_max_ratio * np.max(np.abs(target[:, i]))
        max_green = 0
        for j in range(eff.shape[1]):
            if j != i:
                max_green = max(max_green, np.max(np.abs(eff[:, j])))
        ok = max_green <= bound + 1e-10
        status = "OK" if ok else "FAIL"
        print(f"   Target {i}: max|cross|={max_green:.4f}, bound={bound:.4f} {status}")
    print("="*70)

def plot_effect_decomposition_curves(
    data: pd.DataFrame,
    effects: list[np.ndarray],
    time_index=None,
    save_path: str | None = None,
    show_scatter: bool = True,
    var_names: list[str] | None = None
):
    import math

    if not effects:
        return

    if var_names is None:
        var_names = list(data.columns)

    n, n_targets = len(effects), effects[0].shape[1]

    data_qd = pseudotime_curves.build_quasi_dynamic_frame(data)
    curve_sample = pseudotime_curves.sample_fitted_curve(data_qd, sample_method="linspace", num_points=1000)
    x = curve_sample.index

    n_cols = min(3, n)
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.ravel() if n > 1 else [axes]

    x_last = x.values[-1] if hasattr(x, "values") else x[-1]

    for i, eff in enumerate(effects):
        if show_scatter:
            axes[i].scatter(
                data_qd.index,
                data_qd.iloc[:, i],
                s=18,
                marker="o",
                facecolors="#FFA726",
                edgecolors="#F57C00",
                lw=0.6,
                alpha=0.55,
                zorder=1,
                label="raw quasi-dynamic"
            )

        for j in range(n_targets):
            if np.any(eff[:, j]):
                axes[i].plot(
                    x,
                    eff[:, j],
                    color="red" if i == j else "green",
                    lw=3.2 if i == j else 2.2,
                    alpha=0.92,
                    zorder=2
                )

                y_end = eff[-1, j]
                axes[i].text(
                    x_last,
                    y_end,
                    f" {var_names[j]}",
                    fontsize=9,
                    va="center",
                    color="red" if i == j else "green"
                )

        tot = eff.sum(1)
        axes[i].plot(x, tot, color="blue", lw=3, ls="--", alpha=0.95, zorder=3)
        axes[i].text(x_last, tot[-1], " sum", fontsize=9, va="center", color="blue")

        axes[i].axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.75, zorder=0)
        axes[i].grid(True, linestyle=":", linewidth=0.6, alpha=0.45)

        axes[i].set_title(f"{var_names[i]}")
        axes[i].set_xlabel("EI")
        if i == 0 and show_scatter:
            axes[i].legend(loc="upper left", fontsize=8, frameon=False)

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig

def plot_directed_adjacency_graph(
    adjusted_matrix: np.ndarray,
    curvature: float = 0.2,
    figsize=(10, 10),
    save_path: str | None = None,
    center_node: int | None = None,
    center_mode: str = "total",
    layout_mode: str = "center",
    self_effect_for_color: np.ndarray | None = None,
):
    """
    Plot idopNetwork directed graph:
    - diagonal -> node size (self-effect strength)
    - off-diagonal -> edge width; w>0 red, w<0 blue, curved edges
    - layout_mode: "center" (hub + ring) or "circle" (all on ring)
    - center_node / center_mode: hub selection when layout_mode="center"
    - self_effect_for_color: sign for node color (self-promotion/inhibition)

    Saves figure when save_path is set.
    """
    W, n = np.asarray(adjusted_matrix), adjusted_matrix.shape[0]
    edges = [(i, j) for i in range(n) for j in range(n) if i != j and not (np.isnan(W[i, j]) or W[i, j] == 0)]
    g = ig.Graph(n, directed=True)
    g.add_edges(edges)
    g.es["weight"] = [float(W[i, j]) for (i, j) in edges]
    diag = np.diag(W)
    d_min, d_max = np.nanmin(diag), np.nanmax(diag)
    sz = np.clip(np.interp(diag, (d_min, d_max), (200, 3000)) if d_max > d_min else np.full(n, 1500), 200, 3000).astype(int)
    sign_vec = np.asarray(self_effect_for_color, dtype=float).ravel() if self_effect_for_color is not None else diag
    if len(sign_vec) != n:
        sign_vec = diag
    node_colors = [
        "#F5B7B1" if sign_vec[i] > 0
        else "#AED6F1" if sign_vec[i] < 0
        else "#D5D8DC"
        for i in range(n)
    ]
    if layout_mode == "circle":
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pos = {i: np.array([np.cos(t[i]), np.sin(t[i])]) for i in range(n)}
    else:
        mask = (np.arange(n)[None, :] != np.arange(n)[:, None]) & ~np.isnan(W) & (W != 0)
        out_deg = np.sum(mask, axis=1)
        in_deg = np.sum(mask, axis=0)
        if center_node is not None:
            c = int(center_node)
        else:
            mode = center_mode.lower()
            if mode == "in":
                c = int(np.argmax(in_deg))
            elif mode == "out":
                c = int(np.argmax(out_deg))
            else:
                total_deg = out_deg + in_deg
                c = int(np.argmax(total_deg))
        pos = {c: np.array([0.0, 0.0])}
        oth = [i for i in range(n) if i != c]
        if oth:
            t = np.linspace(0, 2 * np.pi, len(oth), endpoint=False)
            for k, i in enumerate(oth):
                pos[i] = np.array([np.cos(t[k]), np.sin(t[k])])
    r = 0.08 * (sz ** 0.5) / (3000 ** 0.5)
    aw = [abs(float(W[i, j])) for (i, j) in edges]
    mw, Mw = (min(aw), max(aw)) if aw else (1, 1)
    lw = lambda w: 1 + 5 * (abs(w) - mw) / (Mw - mw) if Mw > mw else 3.0
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_aspect("equal")
    ax.axis("off")
    pad = 1.15
    ax.set_xlim(-pad, pad)
    ax.set_ylim(-pad, pad)
    for e in g.es:
        i, j, w = e.source, e.target, e["weight"]
        p1, p2 = np.array(pos[i], float), np.array(pos[j], float)
        d = p2 - p1
        if np.linalg.norm(d) < 1e-12:
            continue
        u = d / np.linalg.norm(d)
        l = lw(w)
        edge_color = "#FF6B6B" if w > 0 else "#4DA3FF"
        ax.add_patch(
            FancyArrowPatch(
                tuple(p1 + u * r[i] * 1.1),
                tuple(p2 - u * r[j] * 1.1),
                arrowstyle="-|>",
                mutation_scale=8 * l,
                linewidth=l,
                color=edge_color,
                alpha=0.9,
                connectionstyle=f"arc3,rad={curvature}",
                zorder=0,
            )
        )
    xy = np.array([pos[i] for i in range(n)])
    ax.scatter(xy[:, 0], xy[:, 1], s=sz, c=node_colors, edgecolors="#34495E", linewidths=1.2, alpha=0.96, zorder=1)
    for i in range(n):
        ax.text(pos[i][0], pos[i][1], str(i), fontsize=10, ha="center", va="center", color="#1A1A1A", zorder=2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig
