"""
iDOP Analysis App - Streamlit Application
Two modules: curve fitting and network reconstruction
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import os
import tempfile

import clinical_data
import pseudotime_curves
import idop_reconstruction

st.set_page_config(
    page_title="iDOP Analysis Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        font-size: 16px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

if 'fitted_data' not in st.session_state:
    st.session_state.fitted_data = None
if 'network_results' not in st.session_state:
    st.session_state.network_results = None

def save_fig_to_buffer(fig, format='png', dpi=150):
    """Save matplotlib figure to in-memory buffer."""
    buf = BytesIO()
    fig.savefig(buf, format=format, dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    return buf

def load_uploaded_files(uploaded_files, n_samples=1000, n_features=100, scaler_type="MinMaxScaler"):
    """Load uploaded CSV files."""
    data_list = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            data = clinical_data.load_clinical_matrix(tmp_path, n_samples=n_samples, n_features=n_features, scaler_type=scaler_type)
            data_list.append(data)
        finally:
            os.unlink(tmp_path)
    
    return data_list

def load_dataframe(df, scaler_type="MinMaxScaler"):
    """Load DataFrame and apply scaling."""
    data = clinical_data.passthrough_columns(df)
    data = clinical_data.scale_numeric_features(data, scaler_type)
    return data

st.markdown('<p class="main-header">🧬 iDOP Analysis Platform</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Dynamic omics fitting · network reconstruction</p>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Global settings")
    
    st.subheader("Preprocessing")
    scaler_type = st.selectbox(
        "Scaling method",
        ["None", "MinMaxScaler", "StandardScaler", "MaxAbsScaler", "LeadingDigit", "LeadingDigit_byrow", "LeadingDigitMinShift"],
        index=1,
        help="None: no scaling; MinMax/Standard/MaxAbs: sklearn; LeadingDigit*: leading-digit transforms"
    )
    
    n_samples = st.number_input("Max samples", min_value=10, max_value=5000, value=1000, step=100)
    n_features = st.number_input("Max features", min_value=1, max_value=500, value=100, step=10)
    
    if st.session_state.get("upload_shape"):
        nr, nc = st.session_state["upload_shape"]
        st.success(f"Uploaded: **{nr}** samples · **{nc}** features")
    
    st.divider()
    st.info("📌 Upload data in each tab before running analysis")
    st.info("📌 Format: CSV with index column (time/sample) in first column")

tab1, tab2 = st.tabs([
    "📈 Curve fitting",
    "🌐 Network reconstruction (idopNetwork)",
])

with tab1:
    st.header("📈 Curve fitting")
    st.markdown("Power-law or logistic curve fitting with multi-group comparison")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Fitting options")
        
        col_upload, col_btn = st.columns([4, 1])
        with col_upload:
            uploaded_files = st.file_uploader(
                "Upload data (CSV)",
                type=['csv'],
                accept_multiple_files=True,
                key="curve_fit_files"
            )
        with col_btn:
            check_shape_fit = st.button("Check shape", key="check_shape_fit", use_container_width=True)
        if check_shape_fit and uploaded_files:
            for f in uploaded_files:
                f.seek(0)
                df = pd.read_csv(f)
                f.seek(0)
                st.caption(f"**{f.name}**: {df.shape[0]} rows × {df.shape[1]} columns")
        if uploaded_files:
            try:
                f = uploaded_files[0]
                f.seek(0)
                df_preview = pd.read_csv(f)
                f.seek(0)
                st.session_state["upload_shape"] = df_preview.shape
            except Exception:
                pass
        else:
            if "upload_shape" in st.session_state:
                del st.session_state["upload_shape"]
        
        curve_function = st.selectbox(
            "Curve type",
            ["eval_power_law", "eval_logistic_curve"],
            format_func=lambda x: "Power law (y = a·xᵇ)" if x == "eval_power_law" else "Logistic"
        )
        
        sample_method = st.selectbox(
            "Sampling method",
            ["linspace", "by_index"],
            format_func=lambda x: "Linspace" if x == "linspace" else "By index"
        )
        
        num_points = st.number_input("Sample points", min_value=100, max_value=5000, value=1000, step=100)
        
        use_log_y = st.checkbox("Log-scale Y axis", value=True)
        
        first_n = st.number_input("Show first N features", min_value=1, max_value=50, value=9, step=1)
        
        n_cols = st.number_input("Subplots per row", min_value=1, max_value=5, value=3, step=1)
        
        default_labels = ", ".join(os.path.splitext(f.name)[0] for f in uploaded_files) if uploaded_files else "Group 1, Group 2, Group 3"
        fit_key = "fit_labels_" + ",".join(sorted(f.name for f in uploaded_files)) if uploaded_files else "fit_labels"
        group_labels = st.text_input(
            "Group labels (comma-separated)",
            value=default_labels,
            key=fit_key,
            help="Default: file stems; e.g. I, II, III or Control, Treatment"
        )
        
        run_fitting = st.button("🚀 Run fitting", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("📊 Results")
        
        if run_fitting:
            if not uploaded_files:
                st.warning("⚠️ Upload at least one data file")
            else:
                with st.spinner("Processing data..."):
                    try:
                        labels = [l.strip() for l in group_labels.split(",")]
                        if len(labels) < len(uploaded_files):
                            labels = [f"Group {i+1}" for i in range(len(uploaded_files))]
                        labels = labels[:len(uploaded_files)]
                        
                        data_list = load_uploaded_files(uploaded_files, n_samples, n_features, scaler_type)
                        
                        qd_list = [pseudotime_curves.build_quasi_dynamic_frame(d) for d in data_list]
                        
                        st.session_state.fitted_data = {
                            'qd_list': qd_list,
                            'original_list': data_list,
                            'labels': labels
                        }
                        
                        fig = pseudotime_curves.plot_quasi_dynamic_curve_fits(
                            qd_list,
                            curve_function=curve_function,
                            sample_method=sample_method,
                            num_points=num_points,
                            group_labels=labels,
                            first_n=first_n,
                            n_cols=n_cols,
                            use_log_y=use_log_y
                        )
                        
                        st.pyplot(fig)
                        
                        buf = save_fig_to_buffer(fig, dpi=300)
                        st.download_button(
                            label="💾 Download figure (PNG)",
                            data=buf,
                            file_name="curve_fitting_result.png",
                            mime="image/png"
                        )
                        
                        with st.expander("📋 Fit parameters"):
                            for i, (qd, label) in enumerate(zip(qd_list, labels)):
                                st.write(f"**{label}**")
                                params = pseudotime_curves.fit_curve_params(qd, curve_function)
                                st.dataframe(params, use_container_width=True)
                        
                        plt.close(fig)
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.exception(e)
        else:
            st.info("👈 Upload data and settings on the left, then click Run fitting")

with tab2:
    st.header("🌐 idopNetwork reconstruction")
    st.markdown("ASGL sparse regression with terminal-divergence constraints")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Network options")
        
        col_upload, col_btn = st.columns([4, 1])
        with col_upload:
            uploaded_file = st.file_uploader(
                "Upload data (CSV)",
                type=["csv"],
                accept_multiple_files=False,
                key="network_file",
                help="Single CSV with time-series / cross-sectional features"
            )
        with col_btn:
            check_shape_net = st.button("Check shape", key="check_shape_net", use_container_width=True)
        if check_shape_net and uploaded_file is not None:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
            uploaded_file.seek(0)
            st.caption(f"**{uploaded_file.name}**: {df.shape[0]} rows × {df.shape[1]} columns")
        if uploaded_file is not None:
            try:
                uploaded_file.seek(0)
                df_preview = pd.read_csv(uploaded_file)
                uploaded_file.seek(0)
                st.session_state["upload_shape"] = df_preview.shape
            except Exception:
                pass
        else:
            if "upload_shape" in st.session_state:
                del st.session_state["upload_shape"]
        
        st.divider()
        st.subheader("Basis functions")
        
        basis_max_order = st.number_input("Legendre polynomial order", min_value=1, max_value=10, value=3, step=1)
        
        st.divider()
        st.subheader("ASGL sparsity")
        
        theta_min = st.slider("Cross-effect threshold (theta_min)", min_value=0.05, max_value=0.5, value=0.2, step=0.05,
                             help="max|cross| >= theta_min * |self|")
        
        lambda_cross = st.number_input("Cross penalty (lambda_cross)", min_value=1e-4, max_value=1.0, value=1e-2, step=1e-3, format="%.4f")
        
        col_lambda1, col_alpha = st.columns(2)
        with col_lambda1:
            lambda1_str = st.text_input("lambda1 range", value="1e-5, 1e-4, 1e-3",
                                       help="Comma-separated; larger = sparser")
        with col_alpha:
            alpha_str = st.text_input("alpha range", value="0.3, 0.5, 0.7",
                                     help="Comma-separated; within-group sparsity")
        
        support_norm_ratio = st.slider("Support norm ratio", min_value=0.1, max_value=1.0, value=0.1, step=0.1)
        
        st.divider()
        st.subheader("Visualization")
        
        viz_network = st.checkbox("Show network graph", value=True)
        viz_effects = st.checkbox("Show effect decomposition", value=True)
        show_scatter = st.checkbox("Show scatter on effect plots", value=False)
        curvature = st.slider("Edge curvature", min_value=0.0, max_value=0.5, value=0.2, step=0.05)
        
        run_network = st.button("🚀 Reconstruct network", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("📊 Network results")
        
        if run_network:
            if uploaded_file is None:
                st.warning("⚠️ Upload a data file")
            else:
                with st.spinner("Reconstructing network (may take several minutes)..."):
                    try:
                        asgl_lambda1_range = [float(x.strip()) for x in lambda1_str.split(",")]
                        asgl_alpha_range = [float(x.strip()) for x in alpha_str.split(",")]
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        try:
                            data = clinical_data.load_clinical_matrix(tmp_path, n_samples, n_features, scaler_type)
                        finally:
                            os.unlink(tmp_path)
                        
                        n_targets = data.shape[1]
                        
                        models, effects, adjusted_matrix = idop_reconstruction.reconstruct_idop_network(
                            data,
                            basis_max_order=basis_max_order,
                            theta_min=theta_min,
                            lambda_cross=lambda_cross,
                            asgl_lambda1_range=asgl_lambda1_range,
                            asgl_alpha_range=asgl_alpha_range,
                            asgl_use_bic=True,
                            sparsify_cross=True,
                            support_norm_ratio=support_norm_ratio
                        )
                        
                        st.session_state.network_results = {
                            'models': models,
                            'effects': effects,
                            'matrix': adjusted_matrix,
                            'data': data
                        }
                        
                        col_net1, col_net2 = st.columns(2)
                        with col_net1:
                            st.metric("Nodes", n_targets)
                        with col_net2:
                            nonzero_edges = np.sum(np.abs(adjusted_matrix) > 1e-8) - np.sum(np.abs(np.diag(adjusted_matrix)) > 1e-8)
                            st.metric("Nonzero edges", int(nonzero_edges))
                        
                        with st.expander("📋 Adjacency matrix"):
                            mat_df = pd.DataFrame(
                                adjusted_matrix,
                                index=[f"Node {i}" for i in range(n_targets)],
                                columns=[f"Node {i}" for i in range(n_targets)]
                            )
                            st.dataframe(mat_df.style.background_gradient(cmap='RdBu_r', axis=None), use_container_width=True)
                            
                            csv = mat_df.to_csv().encode('utf-8')
                            st.download_button(
                                label="📥 Download adjacency matrix (CSV)",
                                data=csv,
                                file_name="network_adjacency_matrix.csv",
                                mime="text/csv"
                            )
                        
                        if viz_effects:
                            st.markdown("### Effect decomposition")
                            fig_effects = idop_reconstruction.plot_effect_decomposition_curves(data, effects, show_scatter=show_scatter)
                            if fig_effects is not None:
                                st.pyplot(fig_effects)
                                buf = save_fig_to_buffer(fig_effects, dpi=300)
                                st.download_button(
                                    label="💾 Download effect plot (PNG)",
                                    data=buf,
                                    file_name="effect_decomposition.png",
                                    mime="image/png"
                                )
                                plt.close(fig_effects)
                        
                        if viz_network:
                            st.markdown("### Network topology")
                            fig_network = idop_reconstruction.plot_directed_adjacency_graph(adjusted_matrix, curvature=curvature)
                            if fig_network is not None:
                                st.pyplot(fig_network)
                                buf = save_fig_to_buffer(fig_network, dpi=300)
                                st.download_button(
                                    label="💾 Download network figure (PNG)",
                                    data=buf,
                                    file_name="network_topology.png",
                                    mime="image/png"
                                )
                                plt.close(fig_network)
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.exception(e)
        else:
            st.info("👈 Upload data and settings on the left, then click Reconstruct network")

st.divider()
st.markdown("""
<div style="text-align: center; color: #888;">
    <p>iDOP Analysis Platform | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
