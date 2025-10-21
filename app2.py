import streamlit as st
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
from PIL import Image
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Solar Cell AI Lab", layout="wide", initial_sidebar_state="expanded")

# Configure matplotlib for dark theme
plt.rcParams.update({
    'figure.facecolor': '#1e293b',
    'axes.facecolor': '#1e293b',
    'axes.edgecolor': '#475569',
    'axes.labelcolor': '#f1f5f9',
    'text.color': '#f1f5f9',
    'xtick.color': '#f1f5f9',
    'ytick.color': '#f1f5f9',
    'grid.color': '#475569',
    'grid.alpha': 0.1,
    'legend.facecolor': '#1e293b',
    'legend.edgecolor': '#475569',
})

# -----------------------------
# Modern Dark-Light Hybrid Theme with Glassmorphism
# -----------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;600;700&display=swap');

    :root {
        --primary: #6366f1;
        --primary-dark: #4f46e5;
        --secondary: #8b5cf6;
        --accent: #06b6d4;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --bg-main: #0f172a;
        --bg-card: rgba(30, 41, 59, 0.7);
        --bg-hover: rgba(51, 65, 85, 0.8);
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --border: rgba(148, 163, 184, 0.1);
        --shadow: 0 20px 60px -15px rgba(0, 0, 0, 0.3);
        --glow: 0 0 30px rgba(99, 102, 241, 0.3);
    }

    * {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
        color: var(--text-primary) !important;
    }

    /* Animated gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .block-container {
        padding: 2rem 2rem 3rem 2rem;
        max-width: 1400px;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: var(--bg-card);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 28px;
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }

    .glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--primary), var(--secondary), var(--accent));
        opacity: 0;
        transition: opacity 0.3s;
    }

    .glass-card:hover::before {
        opacity: 1;
    }

    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 25px 70px -20px rgba(99, 102, 241, 0.4);
        border-color: rgba(99, 102, 241, 0.3);
    }

    /* Hero Header */
    .hero-header {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 48px;
        margin-bottom: 32px;
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
        position: relative;
        overflow: hidden;
    }

    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.3) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }

    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 48px;
        background: linear-gradient(135deg, #fff 0%, #a5b4fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 12px;
        letter-spacing: -0.02em;
        position: relative;
    }

    .hero-subtitle {
        color: var(--text-secondary);
        font-size: 18px;
        font-weight: 400;
        line-height: 1.6;
        position: relative;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.95) 100%) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid var(--border);
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 24px;
    }

    .sidebar-header {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: var(--glow);
        position: relative;
        overflow: hidden;
    }

    .sidebar-header::after {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200px;
        height: 200px;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.1) 0%, transparent 70%);
        animation: rotate 10s linear infinite;
    }

    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    .sidebar-logo {
        width: 56px;
        height: 56px;
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 28px;
        margin-right: 16px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
        vertical-align: middle;
    }

    .sidebar-title {
        display: inline-block;
        vertical-align: middle;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 20px;
        color: white;
    }

    .sidebar-subtitle {
        font-size: 13px;
        color: rgba(255, 255, 255, 0.7);
        margin-top: 4px;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: var(--bg-card);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid var(--border);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.5);
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.2);
    }

    div[data-testid="stMetric"] label {
        color: var(--text-secondary) !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    div[data-testid="stMetric"] > div {
        color: var(--text-primary) !important;
        font-size: 32px !important;
        font-weight: 700 !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 28px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.3) !important;
        cursor: pointer;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.4) !important;
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    /* Input Fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        background: rgba(30, 41, 59, 0.5) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
        padding: 10px 14px !important;
        backdrop-filter: blur(10px);
    }

    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(20px);
        padding: 8px;
        border-radius: 16px;
        border: 1px solid var(--border);
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: var(--text-secondary);
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        border: none;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(99, 102, 241, 0.1);
        color: var(--text-primary);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    }

    /* DataFrames */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid var(--border);
    }

    /* File Uploader */
    .stFileUploader {
        background: rgba(30, 41, 59, 0.5);
        border: 2px dashed var(--border);
        border-radius: 16px;
        padding: 32px;
        backdrop-filter: blur(10px);
    }

    .stFileUploader:hover {
        border-color: var(--primary);
        background: rgba(99, 102, 241, 0.05);
    }

    /* Divider */
    hr {
        margin: 32px 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border), transparent);
    }

    /* Success/Error/Info Messages */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 12px;
        backdrop-filter: blur(10px);
        border: 1px solid;
    }

    .stSuccess {
        background: rgba(16, 185, 129, 0.1) !important;
        border-color: var(--success) !important;
    }

    .stError {
        background: rgba(239, 68, 68, 0.1) !important;
        border-color: var(--danger) !important;
    }

    .stInfo {
        background: rgba(6, 182, 212, 0.1) !important;
        border-color: var(--accent) !important;
    }

    /* Subheaders */
    h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--text-primary) !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(15, 23, 42, 0.5);
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary);
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 12px;
        font-weight: 600;
        color: var(--text-primary);
    }

    /* Radio buttons */
    .stRadio > div {
        background: rgba(30, 41, 59, 0.3);
        border-radius: 12px;
        padding: 12px;
        border: 1px solid var(--border);
    }

    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, var(--success), #059669) !important;
    }

    /* Animation for page load */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .glass-card {
        animation: fadeInUp 0.6s ease-out;
    }

    /* Responsive */
    @media (max-width: 768px) {
        .hero-title { font-size: 32px; }
        .hero-subtitle { font-size: 16px; }
        .block-container { padding: 1rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-header">
            <div style="position: relative; z-index: 1;">
                <div class="sidebar-logo">🔆</div>
                <div style="display: inline-block; vertical-align: middle;">
                    <div class="sidebar-title">Solar Cell AI Lab</div>
                    <div class="sidebar-subtitle">Advanced Analytics Platform</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("⚙️ Configuration", expanded=True):
        folder_path = st.text_input(
            "📁 Data Directory",
            value="",
            help="Leave blank to use current directory",
            placeholder="./data"
        )
        data_dir = folder_path if folder_path else "."

    with st.expander("🤖 Model Files", expanded=False):
        rf_model_file = st.text_input("🌲 RandomForest", value="rf_model.pkl")
        xgb_model_file = st.text_input("⚡ XGBoost", value="xgb_model.json")
        iso_model_file = st.text_input("🔍 IsolationForest", value="iso_model.pkl")
        ocsvm_model_file = st.text_input("🎯 OneClassSVM", value="ocsvm_model.pkl")

    st.markdown("---")
    
    if st.button("🔄 Refresh Dashboard", use_container_width=True):
        st.rerun()

# -----------------------------
# Hero Header
# -----------------------------
st.markdown(
    """
    <div class="hero-header">
        <div class="hero-title">🔆 Solar Cell Intelligence</div>
        <div class="hero-subtitle">
            Next-generation ML platform for photovoltaic performance prediction, 
            explainability analysis, and anomaly detection powered by advanced AI models.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Model Loading
# -----------------------------
load_error = None
try:
    rf_model = joblib.load(os.path.join(data_dir, rf_model_file))
    xgb_model = xgb.Booster()
    xgb_model.load_model(os.path.join(data_dir, xgb_model_file))
    st.sidebar.success("✅ Models loaded successfully")
except Exception as e:
    load_error = e
    st.sidebar.error(f"❌ Model loading failed: {e}")

if load_error:
    st.error("🚨 Unable to load models. Please check your configuration.")
    st.stop()

# -----------------------------
# Load Test Data
# -----------------------------
x_test_file = os.path.join(data_dir, "X_test_processed.csv")
y_test_file = os.path.join(data_dir, "y_test_processed.csv")

if not (os.path.exists(x_test_file) and os.path.exists(y_test_file)):
    st.error("❌ Test data files not found in the specified directory.")
    st.stop()

X_test = pd.read_csv(x_test_file)
y_test = pd.read_csv(y_test_file)
if y_test.shape[1] == 1:
    y_test = y_test.iloc[:, 0]
else:
    if "JV_default_PCE" in y_test.columns:
        y_test = y_test["JV_default_PCE"]
    else:
        y_test = y_test.iloc[:, 0]

# -----------------------------
# Feature Alignment
# -----------------------------
def align_features(model, X):
    if hasattr(model, "feature_names_in_"):
        model_features = list(model.feature_names_in_)
        aligned = pd.DataFrame(0, index=X.index, columns=model_features)
        common = X.columns.intersection(model_features)
        aligned.loc[:, common] = X.loc[:, common]
        return aligned
    return X

X_test = align_features(rf_model, X_test)

# -----------------------------
# Predictions
# -----------------------------
rf_pred = rf_model.predict(X_test)
dtest = xgb.DMatrix(X_test)
xgb_pred = xgb_model.predict(dtest)

# -----------------------------
# Performance Metrics
# -----------------------------
metrics = {
    "RMSE": [
        np.sqrt(mean_squared_error(y_test, rf_pred)),
        np.sqrt(mean_squared_error(y_test, xgb_pred)),
    ],
    "MAE": [
        mean_absolute_error(y_test, rf_pred),
        mean_absolute_error(y_test, xgb_pred),
    ],
    "R²": [
        r2_score(y_test, rf_pred),
        r2_score(y_test, xgb_pred),
    ],
}
df_metrics = pd.DataFrame(metrics, index=["Random Forest", "XGBoost"])

# Metric Cards
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🌲 RF RMSE", f"{df_metrics.loc['Random Forest','RMSE']:.4f}")
with col2:
    st.metric("⚡ XGB RMSE", f"{df_metrics.loc['XGBoost','RMSE']:.4f}")
with col3:
    winner = "Random Forest" if df_metrics.loc["Random Forest","R²"] > df_metrics.loc["XGBoost","R²"] else "XGBoost"
    st.metric("🏆 Best Model", winner)
with col4:
    best_r2 = max(df_metrics["R²"])
    st.metric("📊 Best R²", f"{best_r2:.4f}")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("📈 Model Performance Comparison")
st.dataframe(
    df_metrics.style.highlight_min(color='#ef444420', subset=["RMSE","MAE"])
    .highlight_max(color='#10b98120', subset=["R²"])
    .format("{:.4f}"),
    use_container_width=True
)
st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# -----------------------------
# Tabs
# -----------------------------
tabs = st.tabs([
    "📊 Predictions", 
    "🌫️ Uncertainty", 
    "🔥 SHAP Analysis", 
    "🚨 Anomalies", 
    "🧩 Clusters", 
    "🔮 Predict", 
    "📊 Data Explorer"
])

# TAB 1: True vs Predicted
with tabs[0]:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🎯 Prediction Accuracy Visualization")
    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(7, 5), facecolor='none')
        ax.set_facecolor('#1e293b')
        ax.scatter(y_test, rf_pred, alpha=0.6, c='#6366f1', edgecolors='#4f46e5', linewidth=0.5)
        mn = min(float(y_test.min()), float(rf_pred.min()))
        mx = max(float(y_test.max()), float(rf_pred.max()))
        ax.plot([mn, mx], [mn, mx], 'w--', linewidth=2, alpha=0.7)
        ax.set_title("🌲 Random Forest Predictions", color='#f1f5f9', fontweight='bold', fontsize=14)
        ax.set_xlabel("True PCE", color='#94a3b8', fontsize=11)
        ax.set_ylabel("Predicted PCE", color='#94a3b8', fontsize=11)
        ax.grid(alpha=0.1, color='#475569')
        ax.tick_params(colors='#94a3b8')
        for spine in ax.spines.values():
            spine.set_edgecolor('#334155')
        st.pyplot(fig)
    with right:
        fig2, ax2 = plt.subplots(figsize=(7, 5), facecolor='none')
        ax2.set_facecolor('#1e293b')
        ax2.scatter(y_test, xgb_pred, alpha=0.6, c='#8b5cf6', edgecolors='#7c3aed', linewidth=0.5)
        mn2 = min(float(y_test.min()), float(xgb_pred.min()))
        mx2 = max(float(y_test.max()), float(xgb_pred.max()))
        ax2.plot([mn2, mx2], [mn2, mx2], 'w--', linewidth=2, alpha=0.7)
        ax2.set_title("⚡ XGBoost Predictions", color='#f1f5f9', fontweight='bold', fontsize=14)
        ax2.set_xlabel("True PCE", color='#94a3b8', fontsize=11)
        ax2.set_ylabel("Predicted PCE", color='#94a3b8', fontsize=11)
        ax2.grid(alpha=0.1, color='#475569')
        ax2.tick_params(colors='#94a3b8')
        for spine in ax2.spines.values():
            spine.set_edgecolor('#334155')
        st.pyplot(fig2)
    st.markdown('</div>', unsafe_allow_html=True)

# TAB 2: RF Uncertainty
with tabs[1]:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🌫️ Prediction Uncertainty Analysis")
    try:
        all_tree_preds = np.array([tree.predict(X_test.values) for tree in rf_model.estimators_])
        y_pred_mean = np.mean(all_tree_preds, axis=0)
        y_pred_std = np.std(all_tree_preds, axis=0)
        y_pred_mean = pd.Series(y_pred_mean, index=X_test.index)
        y_pred_std = pd.Series(y_pred_std, index=X_test.index)
        y_true_series = pd.Series(y_test.values, index=X_test.index)

        threshold = np.percentile(y_pred_std, 90)
        mask_high = y_pred_std > threshold

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Avg Uncertainty", f"{np.mean(y_pred_std):.3f}")
        with col2:
            st.metric("⚠️ High Uncertainty Threshold", f"{threshold:.3f}")
        with col3:
            st.metric("🔴 High Uncertainty Samples", f"{mask_high.sum()}")

        fig, ax = plt.subplots(figsize=(10, 6), facecolor='none')
        ax.set_facecolor('#1e293b')
        ax.errorbar(
            y_true_series, y_pred_mean, yerr=y_pred_std,
            fmt='o', ecolor='#ef4444', alpha=0.3, markersize=4, color='#6366f1',
            label='Predictions ± Uncertainty'
        )
        ax.scatter(
            y_true_series[mask_high], y_pred_mean[mask_high],
            color='#f59e0b', s=100, edgecolor='#fff', linewidth=1.5,
            label='High Uncertainty', zorder=5
        )
        mn = float(min(y_true_series.min(), y_pred_mean.min()))
        mx = float(max(y_true_series.max(), y_pred_mean.max()))
        ax.plot([mn, mx], [mn, mx], 'w--', linewidth=2, alpha=0.7)
        ax.set_xlabel("True PCE", color='#94a3b8', fontsize=12)
        ax.set_ylabel("Predicted PCE", color='#94a3b8', fontsize=12)
        ax.set_title("Uncertainty Quantification", color='#f1f5f9', fontweight='bold', fontsize=14)
        ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#f1f5f9')
        ax.grid(alpha=0.1, color='#475569')
        ax.tick_params(colors='#94a3b8')
        for spine in ax.spines.values():
            spine.set_edgecolor('#334155')
        st.pyplot(fig)

        st.subheader("🔍 Top Uncertain Predictions")
        top_uncertain = pd.DataFrame({
            "True PCE": y_true_series[mask_high],
            "Predicted PCE": y_pred_mean[mask_high],
            "Uncertainty": y_pred_std[mask_high],
        }).sort_values("Uncertainty", ascending=False)
        st.dataframe(top_uncertain.head(10).style.format("{:.3f}"), use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error computing uncertainty: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# TAB 3: SHAP
with tabs[2]:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🔥 SHAP Feature Importance")
    shap_image_path = os.path.join(data_dir, "shap.png")
    col1, col2 = st.columns([2, 1])
    with col1:
        if os.path.exists(shap_image_path):
            image = Image.open(shap_image_path)
            st.image(image, caption="Global Feature Impact Analysis", use_container_width=True)
        else:
            st.info("📊 No pre-computed SHAP visualization found.")
            if st.button("⚡ Compute SHAP Analysis", use_container_width=True):
                with st.spinner("Computing SHAP values..."):
                    explainer = shap.TreeExplainer(rf_model)
                    shap_vals = explainer.shap_values(X_test)
                    
                    # Create figure with dark theme styling
                    fig = plt.figure(figsize=(12, 8), facecolor='#1e293b')
                    shap.summary_plot(shap_vals, X_test, show=False)
                    
                    # Style for dark theme
                    ax = plt.gca()
                    ax.set_facecolor('#1e293b')
                    
                    # Fix all text colors
                    for text in ax.texts:
                        text.set_color('#f1f5f9')
                    
                    # Fix axis labels
                    ax.tick_params(colors='#f1f5f9', labelsize=10)
                    ax.xaxis.label.set_color('#f1f5f9')
                    ax.yaxis.label.set_color('#f1f5f9')
                    
                    if ax.get_title():
                        ax.set_title(ax.get_title(), color='#f1f5f9', fontsize=14, fontweight='bold')
                    
                    # Fix colorbar if present
                    if hasattr(fig, 'colorbar'):
                        fig.colorbar.ax.tick_params(colors='#f1f5f9')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
    with col2:
        st.markdown("**📖 Interpretation Guide**")
        st.markdown(
            """
            - 🔴 **Red**: High feature values
            - 🔵 **Blue**: Low feature values  
            - ➡️ **X-axis**: Impact magnitude
            - 📊 **Y-axis**: Feature importance ranking
            
            Features at the top have the strongest influence on PCE predictions.
            """
        )
    st.markdown('</div>', unsafe_allow_html=True)

# TAB 4: Anomaly
with tabs[3]:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🚨 Anomaly Detection & Analysis")

    uploaded_anomaly_file = st.file_uploader("📤 Upload dataset with anomaly columns", type=["csv"], key="anomaly_uploader")
    if uploaded_anomaly_file:
        df_anomaly = pd.read_csv(uploaded_anomaly_file)
    else:
        default_path = os.path.join(data_dir, "data.csv")
        if os.path.exists(default_path):
            df_anomaly = pd.read_csv(default_path)
        else:
            st.warning("⚠️ Please upload a dataset with anomaly labels.")
            st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("📊 Total Samples", f"{df_anomaly.shape[0]:,}")
    with col2:
        st.metric("🔢 Features", df_anomaly.shape[1])

    with st.expander("👁️ View Dataset Preview"):
        st.dataframe(df_anomaly.head(10), use_container_width=True)

    anomaly_cols = [col for col in df_anomaly.columns if "anomaly" in col.lower()]
    if not anomaly_cols:
        st.error("❌ No anomaly columns detected in dataset.")
        st.stop()

    for col in anomaly_cols:
        df_anomaly[col] = pd.to_numeric(df_anomaly[col], errors="coerce").fillna(0).astype(int)
        if df_anomaly[col].min() == -1 and df_anomaly[col].max() == 1:
            df_anomaly[col] = df_anomaly[col].replace({-1: 1, 1: 0})

    st.subheader("📊 Anomaly Distribution")
    anomaly_counts = df_anomaly[anomaly_cols].sum().to_frame("Count")
    anomaly_counts["Percentage"] = (anomaly_counts["Count"] / len(df_anomaly) * 100).round(2)
    st.dataframe(anomaly_counts.style.format({"Count": "{:,.0f}", "Percentage": "{:.2f}%"}), use_container_width=True)

    feature_cols = [c for c in df_anomaly.columns if c not in anomaly_cols]
    X = df_anomaly[feature_cols].select_dtypes(include=[float, int]).fillna(0)

    method = st.radio("🎨 Visualization Method", ["PCA", "t-SNE"], horizontal=True)
    if method == "PCA":
        reducer = PCA(n_components=2, random_state=42)
    else:
        st.info("⏳ t-SNE computation may take longer for large datasets")
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)

    with st.spinner(f"Computing {method} projection..."):
        emb = reducer.fit_transform(X)
    df_emb = pd.DataFrame(emb, columns=["Component 1", "Component 2"])
    df_emb["Anomaly"] = df_anomaly["anomaly_iso"] if "anomaly_iso" in df_anomaly.columns else 0

    fig1, ax1 = plt.subplots(figsize=(10, 7), facecolor='none')
    ax1.set_facecolor('#1e293b')
    
    normal = df_emb[df_emb["Anomaly"] == 0]
    anomalies = df_emb[df_emb["Anomaly"] == 1]
    
    ax1.scatter(normal["Component 1"], normal["Component 2"], 
                c='#6366f1', alpha=0.5, s=40, label='Normal', edgecolors='none')
    ax1.scatter(anomalies["Component 1"], anomalies["Component 2"], 
                c='#ef4444', alpha=0.8, s=80, label='Anomaly', 
                edgecolors='#fff', linewidth=1, marker='X')
    
    ax1.set_title(f"{method} Anomaly Visualization", color='#f1f5f9', fontweight='bold', fontsize=14)
    ax1.set_xlabel("Component 1", color='#94a3b8', fontsize=11)
    ax1.set_ylabel("Component 2", color='#94a3b8', fontsize=11)
    ax1.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#f1f5f9')
    ax1.grid(alpha=0.1, color='#475569')
    ax1.tick_params(colors='#94a3b8')
    for spine in ax1.spines.values():
        spine.set_edgecolor('#334155')
    st.pyplot(fig1)

    st.subheader("🔍 SHAP Explanation for Anomalies")
    anomaly_indices = df_anomaly.index[df_anomaly["anomaly_iso"] == 1].tolist() if "anomaly_iso" in df_anomaly.columns else []
    if not anomaly_indices:
        st.info("ℹ️ No anomalies detected for analysis.")
    else:
        try:
            if os.path.exists(os.path.join(data_dir, iso_model_file)):
                iso_model = joblib.load(os.path.join(data_dir, iso_model_file))
                st.success("✅ Isolation Forest model loaded")
            else:
                from sklearn.ensemble import IsolationForest
                iso_model = IsolationForest(contamination=0.05, random_state=42).fit(X)
                st.info("ℹ️ Fitted new Isolation Forest model")
        except Exception as e:
            st.warning(f"⚠️ Model error: {e}")
            st.stop()

        if hasattr(iso_model, "feature_names_in_"):
            iso_features = list(iso_model.feature_names_in_)
        else:
            iso_features = list(X.columns)

        def iso_decision_fn(x_np):
            x_df = pd.DataFrame(x_np, columns=iso_features)
            return iso_model.decision_function(x_df[iso_features])

        selected_idx = st.selectbox("Select anomaly for detailed analysis:", options=anomaly_indices)
        sample = X.loc[[selected_idx], iso_features]
        background = X[iso_features].sample(n=min(100, len(X)), random_state=42)
        try:
            explainer = shap.Explainer(iso_decision_fn, masker=shap.maskers.Independent(background), feature_names=iso_features)
            shap_values = explainer(sample.to_numpy())
            st.markdown(f"**Analyzing Sample:** `{selected_idx}`")
            
            # Create figure with proper styling for dark theme
            fig2 = plt.figure(figsize=(12, 8), facecolor='#1e293b')
            shap.plots.waterfall(shap_values[0], max_display=15, show=False)
            
            # Style the plot for dark theme
            ax2 = plt.gca()
            ax2.set_facecolor('#1e293b')
            
            # Fix text colors for visibility
            for text in ax2.texts:
                text.set_color('#f1f5f9')
                text.set_fontsize(10)
            
            # Fix axis labels and title
            ax2.tick_params(colors='#f1f5f9', labelsize=10)
            ax2.xaxis.label.set_color('#f1f5f9')
            ax2.yaxis.label.set_color('#f1f5f9')
            
            if ax2.get_title():
                ax2.set_title(ax2.get_title(), color='#f1f5f9', fontsize=12, fontweight='bold')
            
            # Fix spines
            for spine in ax2.spines.values():
                spine.set_edgecolor('#475569')
            
            # Fix grid
            ax2.grid(alpha=0.1, color='#475569')
            
            plt.tight_layout()
            st.pyplot(fig2)
        except Exception as e:
            st.error(f"❌ SHAP computation failed: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# TAB 5: Clustering
with tabs[4]:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🧩 Cluster Analysis")
    data_path = os.path.join(data_dir, "data3.csv")
    if not os.path.exists(data_path):
        st.error("❌ data3.csv not found")
    else:
        df = pd.read_csv(data_path)
        if "composition_cluster" not in df.columns:
            st.warning("⚠️ No clustering information found")
        else:
            st.subheader("📊 Cluster Distribution")
            cluster_counts = df["composition_cluster"].value_counts().sort_index()
            
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(cluster_counts.to_frame("Samples").style.format("{:,}"), use_container_width=True)
            with col2:
                if "JV_default_PCE" in df.columns:
                    cluster_stats = df.groupby("composition_cluster")["JV_default_PCE"].agg(['mean', 'std', 'count'])
                    st.dataframe(cluster_stats.style.format("{:.3f}"), use_container_width=True)
            
            static_img = os.path.join(data_dir, "plot.png")
            if os.path.exists(static_img):
                st.image(static_img, caption="Cluster Visualization", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# TAB 6: RF Prediction
with tabs[5]:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🔮 PCE Prediction & Explainability")
    st.markdown("Upload a preprocessed dataset to predict PCE values and understand feature contributions.")
    
    uploaded_file = st.file_uploader("📤 Upload Preprocessed CSV", type=["csv"], key="rf_predict_uploader")
    if uploaded_file:
        user_df = pd.read_csv(uploaded_file)
        if hasattr(rf_model, "feature_names_in_"):
            model_features = list(rf_model.feature_names_in_)
            X_user = pd.DataFrame(0, index=user_df.index, columns=model_features)
            for col in user_df.columns:
                if col in model_features:
                    X_user[col] = user_df[col]
        else:
            X_user = user_df.copy()

        st.success(f"✅ Loaded {len(X_user)} samples with {X_user.shape[1]} features")
        
        with st.expander("👁️ View & Edit Data"):
            edited_df = st.data_editor(X_user, num_rows="dynamic", use_container_width=True)

        if st.button("🚀 Generate Predictions", use_container_width=True, type="primary"):
            preds = rf_model.predict(edited_df)
            preds_df = pd.DataFrame({"Sample_Index": range(len(preds)), "Predicted_PCE": preds})
            
            st.subheader("📊 Prediction Results")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📈 Max PCE", f"{preds.max():.3f}")
            with col2:
                st.metric("📉 Min PCE", f"{preds.min():.3f}")
            with col3:
                st.metric("📊 Mean PCE", f"{preds.mean():.3f}")
            
            st.dataframe(
                preds_df.style.highlight_max(subset=['Predicted_PCE'], color='#10b98130')
                .highlight_min(subset=['Predicted_PCE'], color='#ef444430')
                .format({"Predicted_PCE": "{:.4f}"}),
                use_container_width=True
            )

            csv_bytes = preds_df.to_csv(index=False).encode()
            st.download_button("⬇️ Download Predictions", data=csv_bytes, file_name="pce_predictions.csv", mime="text/csv", use_container_width=True)
            st.subheader("🔍 SHAP Feature Contribution")

            selected_index = st.number_input(
                "Select row index for SHAP analysis:",
                0, len(edited_df)-1, 0
            )

            # Use only one sample for SHAP explanation
            sample = edited_df.iloc[[selected_index]]

            # TreeExplainer optimized for RF
            explainer = shap.TreeExplainer(rf_model)
            shap_values_single = explainer.shap_values(sample)

            st.markdown(f"**Explaining Sample Index:** `{selected_index}`")

            # --- Waterfall Plot (Single Instance) ---
            fig, ax = plt.subplots(figsize=(8, 4))
            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_values_single[0],
                    base_values=explainer.expected_value,
                    data=sample.iloc[0],
                    feature_names=sample.columns
                ),
                max_display=10,
                show=False
            )
            st.pyplot(fig)

            # --- Optional Local Feature Impact Bar ---
            st.subheader("📊 Local Feature Importance (This Sample Only)")
            shap_importance = pd.Series(shap_values_single[0], index=sample.columns).abs().sort_values(ascending=False)
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            shap_importance.head(10).plot(kind="barh", ax=ax2)
            ax2.set_xlabel("SHAP Value Magnitude (|impact|)")
            ax2.set_ylabel("Feature")
            ax2.invert_yaxis()
            st.pyplot(fig2)

    else:
        st.info("📎 Upload a CSV file to begin prediction and analysis")
    st.markdown('</div>', unsafe_allow_html=True)

# TAB 7: Data Visualization
with tabs[6]:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📊 Interactive Data Explorer")
    uploaded_data_file = st.file_uploader("📤 Upload Dataset for Exploration", type=["csv"], key="explore_uploader")
    if uploaded_data_file:
        df_vis = pd.read_csv(uploaded_data_file)
        st.success(f"✅ Dataset loaded: {df_vis.shape[0]:,} samples × {df_vis.shape[1]} features")
        
        with st.expander("👁️ Data Preview", expanded=False):
            st.dataframe(df_vis.head(20), use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Samples", f"{len(df_vis):,}")
        with col2:
            st.metric("🔢 Features", df_vis.shape[1])
        with col3:
            st.metric("❓ Missing", f"{df_vis.isna().sum().sum():,}")
        with col4:
            missing_pct = (df_vis.isna().sum().sum() / (df_vis.shape[0] * df_vis.shape[1]) * 100)
            st.metric("📉 Missing %", f"{missing_pct:.2f}%")

        st.subheader("📋 Statistical Summary")
        st.dataframe(
            df_vis.describe().T.style.highlight_max(color='#10b98130', axis=0)
            .highlight_min(color='#ef444430', axis=0)
            .format("{:.3f}"),
            use_container_width=True
        )

        st.subheader("📊 Feature Distribution")
        num_cols = df_vis.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) == 0:
            st.warning("⚠️ No numeric columns found")
        else:
            selected_feature = st.selectbox("Select feature to visualize:", num_cols, key="dist_feature")
            fig1, ax1 = plt.subplots(figsize=(12, 5), facecolor='none')
            ax1.set_facecolor('#1e293b')
            ax1.hist(df_vis[selected_feature].dropna(), bins=50, color='#6366f1', alpha=0.7, edgecolor='#4f46e5')
            ax1.set_title(f"Distribution: {selected_feature}", color='#f1f5f9', fontweight='bold', fontsize=14)
            ax1.set_xlabel(selected_feature, color='#94a3b8', fontsize=11)
            ax1.set_ylabel("Frequency", color='#94a3b8', fontsize=11)
            ax1.grid(alpha=0.1, color='#475569', axis='y')
            ax1.tick_params(colors='#94a3b8')
            for spine in ax1.spines.values():
                spine.set_edgecolor('#334155')
            st.pyplot(fig1)

            st.subheader("🔥 Correlation Heatmap")
            if len(num_cols) > 1:
                corr = df_vis[num_cols].corr()
                fig2, ax2 = plt.subplots(figsize=(14, 10), facecolor='none')
                sns.heatmap(corr, center=0, annot=True, fmt=".2f", linewidths=0.5, 
                           cmap='coolwarm', cbar_kws={'label': 'Correlation'}, ax=ax2)
                ax2.set_title("Feature Correlation Matrix", color='#f1f5f9', fontweight='bold', fontsize=14, pad=20)
                plt.xticks(rotation=45, ha='right', color='#94a3b8')
                plt.yticks(rotation=0, color='#94a3b8')
                st.pyplot(fig2)

            st.subheader("🔗 Pairwise Feature Relationships")
            pair_features = st.multiselect("Select features (max 4):", num_cols, default=num_cols[:min(3, len(num_cols))], max_selections=4)
            if len(pair_features) >= 2:
                with st.spinner("Generating pairplot..."):
                    pairplot_fig = sns.pairplot(df_vis[pair_features].dropna(), 
                                                diag_kind="kde", 
                                                plot_kws={'alpha': 0.6, 'color': '#6366f1'},
                                                diag_kws={'color': '#8b5cf6'})
                    pairplot_fig.fig.set_size_inches(12, 12)
                    st.pyplot(pairplot_fig.fig)
            else:
                st.info("ℹ️ Select at least 2 features for pairplot")

        st.subheader("📦 Category Distribution")
        cat_cols = df_vis.select_dtypes(exclude=[np.number]).columns.tolist()
        if cat_cols:
            cat_col = st.selectbox("Categorical feature:", cat_cols, key="box_cat")
            num_col = st.selectbox("Numeric feature:", num_cols, key="box_num") if num_cols else None
            if num_col:
                if df_vis[cat_col].nunique() > 20:
                    st.info(f"Showing top 20 categories (out of {df_vis[cat_col].nunique()})")
                    top_cats = df_vis[cat_col].value_counts().nlargest(20).index
                    df_box = df_vis[df_vis[cat_col].isin(top_cats)].copy()
                else:
                    df_box = df_vis.copy()
                fig4, ax4 = plt.subplots(figsize=(14, 6), facecolor='none')
                ax4.set_facecolor('#1e293b')
                bp = ax4.boxplot([df_box[df_box[cat_col]==cat][num_col].dropna() for cat in df_box[cat_col].unique()],
                                 labels=df_box[cat_col].unique(), patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor('#6366f1')
                    patch.set_alpha(0.7)
                ax4.set_title(f"{num_col} by {cat_col}", color='#f1f5f9', fontweight='bold', fontsize=14)
                ax4.set_xlabel(cat_col, color='#94a3b8', fontsize=11)
                ax4.set_ylabel(num_col, color='#94a3b8', fontsize=11)
                plt.xticks(rotation=45, ha='right', color='#94a3b8')
                ax4.tick_params(colors='#94a3b8')
                ax4.grid(alpha=0.1, color='#475569', axis='y')
                for spine in ax4.spines.values():
                    spine.set_edgecolor('#334155')
                st.pyplot(fig4)
        else:
            st.info("ℹ️ No categorical columns available")
    else:
        st.info("📎 Upload a CSV to start exploring your data")
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #94a3b8; padding: 20px; font-size: 13px;">
        <p>🔆 <strong>Solar Cell AI Lab</strong> • Powered by Machine Learning & Advanced Analytics</p>
        <p style="font-size: 11px; margin-top: 8px;">Built with Streamlit • XGBoost • Random Forest • SHAP</p>
    </div>
    """,
    unsafe_allow_html=True
)
