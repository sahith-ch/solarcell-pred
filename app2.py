# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Solar Cell Model Dashboard", layout="wide", initial_sidebar_state="expanded")

# -----------------------------
# Custom CSS: Modern Light Theme
# -----------------------------
st.markdown(
    """
    <style>
    /* Import Google fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Poppins:wght@400;600&display=swap');

    :root{
        --accent: #1f77b4;
        --accent-2: #4aa3df;
        --muted: #6c7280;
        --card-bg: #ffffff;
        --page-bg: #f7fbff; /* very light blue */
        --radius: 12px;
        --shadow: 0 6px 18px rgba(31,119,180,0.08);
    }

    html, body, [class*="css"]  {
        font-family: "Inter", "Poppins", sans-serif !important;
        background: var(--page-bg) !important;
    }

    /* Main container */
    .block-container {
        padding-top: 1rem;
        padding-left: 1.25rem;
        padding-right: 1.25rem;
    }

    /* Card style for sections */
    .card {
        background: var(--card-bg);
        border-radius: var(--radius);
        padding: 18px;
        box-shadow: var(--shadow);
        margin-bottom: 16px;
        border: 1px solid rgba(31,119,180,0.04);
    }

    /* Title */
    .big-title {
        font-family: "Poppins", sans-serif;
        font-weight: 600;
        font-size: 22px;
        color: #0f1724;
        margin-bottom: 6px;
    }
    .subtitle {
        color: var(--muted);
        margin-top: -6px;
        margin-bottom: 12px;
        font-size: 13px;
    }

    /* Sidebar styling */
    .css-1d391kg { /* wrapper for sidebar content (class may change across Streamlit versions) */
        padding-top: 12px;
    }
    .sidebar-card {
        background: linear-gradient(180deg, #ffffff, #fbfdff);
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 12px;
        box-shadow: 0 6px 16px rgba(15,23,36,0.04);
        border: 1px solid rgba(31,119,180,0.04);
    }
    .logo {
        width: 44px;
        height: 44px;
        border-radius: 10px;
        background: linear-gradient(90deg, var(--accent), var(--accent-2));
        display:inline-block;
        vertical-align:middle;
        margin-right: 10px;
        box-shadow: 0 6px 14px rgba(74,163,223,0.12);
    }
    .sidebar-title { display:inline-block; vertical-align: middle; font-weight:600; }

    /* Metrics styling */
    .stMetric { padding: 6px 0; }

    /* Tables */
    table {
        border-radius: 8px;
        overflow: hidden;
    }

    /* Buttons */
    .stButton>button {
        border-radius: 10px;
        padding: 8px 12px;
        background: linear-gradient(90deg, var(--accent), var(--accent-2));
        border: none;
        color: white;
        font-weight: 600;
    }
    .stButton>button:hover { filter: brightness(0.98); }

    /* Smaller adjustments to spacing in outputs */
    .streamlit-expanderHeader { font-weight:600; }

    /* make charts and images appear on cards cleaner */
    .element-container .stImage img {
        border-radius: 8px;
        box-shadow: 0 8px 30px rgba(31,119,180,0.06);
    }

    /* responsive tweak */
    @media (max-width: 900px) {
        .block-container { padding-left: 10px; padding-right: 10px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Sidebar: Styled Controls & Model Loading
# -----------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-card">
            <div style="display:flex; align-items:center;">
                <div class="logo"></div>
                <div>
                    <div class="sidebar-title">Solar Cell Dashboard</div><br>
                    <span style="font-size:12px; color:#6c7280">Model evaluation · SHAP · Anomaly</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown("### 🔧 Controls")
    folder_path = st.text_input(
        "Folder path (x_test/y_test, model files):",
        value="",
        help="Leave blank to use app directory."
    )
    data_dir = folder_path if folder_path else "."

    st.markdown("---")
    st.markdown("### 🧾 Model files (filenames)")
    rf_model_file = st.text_input("RandomForest (.pkl):", value="rf_model.pkl")
    xgb_model_file = st.text_input("XGBoost (.json):", value="xgb_model.json")
    iso_model_file = st.text_input("IsolationForest (.pkl):", value="iso_model.pkl")
    ocsvm_model_file = st.text_input("OneClassSVM (.pkl):", value="ocsvm_model.pkl")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="sidebar-card">
            <div style="display:flex; gap:8px; align-items:center;">
                <div style="flex:1">
                    <strong>Quick actions</strong><br>
                    <span style="font-size:12px; color:#6c7280">Model & dataset checks</span>
                </div>
            </div>
            <div style="margin-top:10px">
        """,
        unsafe_allow_html=True,
    )
    st.button("Reload app (refresh)", key="btn_refresh")
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Top header inside main page
# -----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="big-title">🔆 Solar Cell ML Model Evaluation Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Clean, modern layout — model evaluation, explainability, anomaly detection, clustering and dataset exploration.</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Attempt to load models (robust)
# -----------------------------
load_error = None
try:
    rf_model = joblib.load(os.path.join(data_dir, rf_model_file))
    xgb_model = xgb.Booster()
    xgb_model.load_model(os.path.join(data_dir, xgb_model_file))
    st.sidebar.success("✅ Models loaded.")
except Exception as e:
    load_error = e
    st.sidebar.error(f"❌ Model load failed: {e}")

if load_error:
    st.error("Failed to load models — check sidebar filenames and paths.")
    st.stop()

# -----------------------------
# Load test data
# -----------------------------
x_test_file = os.path.join(data_dir, "X_test_processed.csv")
y_test_file = os.path.join(data_dir, "y_test_processed.csv")

if not (os.path.exists(x_test_file) and os.path.exists(y_test_file)):
    st.error("❌ Missing `x_test_processed.csv` or `y_test_processed.csv` in the chosen folder.")
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
# Feature alignment
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
# Performance metrics
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

# Summary metrics cards
st.markdown('<div class="card">', unsafe_allow_html=True)
col1, col2, col3 = st.columns([1.2, 1.2, 1])
with col1:
    st.metric("RF RMSE", f"{df_metrics.loc['Random Forest','RMSE']:.4f}")
with col2:
    st.metric("XGB RMSE", f"{df_metrics.loc['XGBoost','RMSE']:.4f}")
with col3:
    winner = "Random Forest" if df_metrics.loc["Random Forest","R²"] > df_metrics.loc["XGBoost","R²"] else "XGBoost"
    st.metric("Best R²", winner)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📈 Model Performance Summary")
st.dataframe(df_metrics.style.highlight_min(color='lightcoral', subset=["RMSE","MAE"])
             .highlight_max(color='lightgreen', subset=["R²"])
             .format("{:.4f}"), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# -----------------------------
# Tabs
# -----------------------------
tabs = st.tabs(["📊 True vs Predicted", "🌫️ RF Uncertainty", "🔥 SHAP Feature Importance", "🚨 Anomaly", "🧩 Clustering", "🔮 Prediction (RF+SHAP)", "📊 Data Visualization"])

# TAB 1: True vs Predicted
with tabs[0]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("True vs Predicted")
    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(y_test, rf_pred, alpha=0.6)
        mn = min(float(y_test.min()), float(rf_pred.min()))
        mx = max(float(y_test.max()), float(rf_pred.max()))
        ax.plot([mn, mx], [mn, mx], 'k--', linewidth=1)
        ax.set_title("Random Forest — True vs Predicted")
        ax.set_xlabel("True PCE")
        ax.set_ylabel("Predicted PCE")
        ax.grid(alpha=0.15)
        st.pyplot(fig)
    with right:
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        ax2.scatter(y_test, xgb_pred, alpha=0.6)
        mn2 = min(float(y_test.min()), float(xgb_pred.min()))
        mx2 = max(float(y_test.max()), float(xgb_pred.max()))
        ax2.plot([mn2, mx2], [mn2, mx2], 'k--', linewidth=1)
        ax2.set_title("XGBoost — True vs Predicted")
        ax2.set_xlabel("True PCE")
        ax2.set_ylabel("Predicted PCE")
        ax2.grid(alpha=0.15)
        st.pyplot(fig2)
    st.markdown('</div>', unsafe_allow_html=True)

# TAB 2: RF Uncertainty
with tabs[1]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🌫️ Random Forest Prediction Uncertainty")
    try:
        all_tree_preds = np.array([tree.predict(X_test.values) for tree in rf_model.estimators_])
        y_pred_mean = np.mean(all_tree_preds, axis=0)
        y_pred_std = np.std(all_tree_preds, axis=0)
        y_pred_mean = pd.Series(y_pred_mean, index=X_test.index)
        y_pred_std = pd.Series(y_pred_std, index=X_test.index)
        y_true_series = pd.Series(y_test.values, index=X_test.index)

        threshold = np.percentile(y_pred_std, 90)
        mask_high = y_pred_std > threshold

        st.markdown(f"- **Average uncertainty (std):** `{np.mean(y_pred_std):.3f}`")
        st.markdown(f"- **Top 10% high uncertainty threshold:** `{threshold:.3f}`")

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.errorbar(
            y_true_series, y_pred_mean, yerr=y_pred_std,
            fmt='o', ecolor='lightcoral', alpha=0.4, label='Predictions ± Uncertainty'
        )
        ax.scatter(
            y_true_series[mask_high], y_pred_mean[mask_high],
            color='orange', s=70, edgecolor='k', label='High Uncertainty'
        )
        mn = float(min(y_true_series.min(), y_pred_mean.min()))
        mx = float(max(y_true_series.max(), y_pred_mean.max()))
        ax.plot([mn, mx], [mn, mx], 'k--')
        ax.set_xlabel("True PCE")
        ax.set_ylabel("Predicted PCE")
        ax.set_title("Random Forest Predictions with Uncertainty")
        ax.legend()
        ax.grid(alpha=0.15)
        st.pyplot(fig)

        st.subheader("Top uncertain predictions")
        top_uncertain = pd.DataFrame({
            "True PCE": y_true_series[mask_high],
            "Predicted PCE": y_pred_mean[mask_high],
            "Std (Uncertainty)": y_pred_std[mask_high],
        }).sort_values("Std (Uncertainty)", ascending=False)
        st.dataframe(top_uncertain.head(10).style.format("{:.3f}"), use_container_width=True)

    except Exception as e:
        st.error(f"Error computing uncertainty: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# TAB 3: SHAP Global
with tabs[2]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    from PIL import Image
    st.header("🔍 Model Explainability (SHAP Summary)")
    shap_image_path = os.path.join(data_dir, "shap.png")
    col1, col2 = st.columns([2, 1])
    with col1:
        if os.path.exists(shap_image_path):
            image = Image.open(shap_image_path)
            st.image(image, caption="SHAP Feature Importance — Global Explanation", use_container_width=True)
        else:
            st.info("No `shap.png` found. Optionally compute SHAP summary below (may take time).")
            if st.button("Compute SHAP summary (RandomForest) — may take time"):
                with st.spinner("Computing SHAP values..."):
                    explainer = shap.TreeExplainer(rf_model)
                    shap_vals = explainer.shap_values(X_test)
                    fig = plt.figure(figsize=(10, 6))
                    shap.summary_plot(shap_vals, X_test, show=False)
                    st.pyplot(fig)
    with col2:
        st.markdown("**How to read this plot**")
        st.markdown(
            """
            - Each dot = a sample; color = feature value (red = high, blue = low).  
            - X-axis: impact on model output (positive -> increases PCE).  
            - Features higher in the chart are more influential.
            """
        )
    st.markdown('</div>', unsafe_allow_html=True)

# TAB 4: Anomaly
with tabs[3]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    import seaborn as sns
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    st.header("🚨 Anomaly Detection Visualization & Explanation")

    uploaded_anomaly_file = st.file_uploader("Upload dataset with anomaly columns (CSV)", type=["csv"], key="anomaly_uploader")
    if uploaded_anomaly_file:
        df_anomaly = pd.read_csv(uploaded_anomaly_file)
    else:
        default_path = os.path.join(data_dir, "data.csv")
        if os.path.exists(default_path):
            df_anomaly = pd.read_csv(default_path)
        else:
            st.warning("Upload / provide a dataset with anomaly columns (e.g., 'anomaly_iso').")
            st.stop()

    st.write("Dataset shape:", df_anomaly.shape)
    st.dataframe(df_anomaly.head(), use_container_width=True)

    anomaly_cols = [col for col in df_anomaly.columns if "anomaly" in col.lower()]
    if not anomaly_cols:
        st.error("No anomaly columns found.")
        st.stop()

    for col in anomaly_cols:
        df_anomaly[col] = pd.to_numeric(df_anomaly[col], errors="coerce").fillna(0).astype(int)
        if df_anomaly[col].min() == -1 and df_anomaly[col].max() == 1:
            df_anomaly[col] = df_anomaly[col].replace({-1: 1, 1: 0})

    st.subheader("📊 Anomaly Counts")
    st.write(df_anomaly[anomaly_cols].sum())

    feature_cols = [c for c in df_anomaly.columns if c not in anomaly_cols]
    X = df_anomaly[feature_cols].select_dtypes(include=[float, int]).fillna(0)

    method = st.radio("Choose visualization method:", ["PCA", "t-SNE"])
    if method == "PCA":
        reducer = PCA(n_components=2, random_state=42)
    else:
        st.info("t-SNE may be slow for large datasets.")
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)

    with st.spinner(f"Computing {method} embedding..."):
        emb = reducer.fit_transform(X)
    df_emb = pd.DataFrame(emb, columns=["dim1", "dim2"])
    df_emb["anomaly_iso"] = df_anomaly["anomaly_iso"] if "anomaly_iso" in df_anomaly.columns else 0

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        x="dim1", y="dim2", hue="anomaly_iso",
        data=df_emb, palette={0: "gray", 1: "red"},
        alpha=0.7, ax=ax1
    )
    ax1.set_title(f"{method} Projection — IsolationForest Anomalies Highlighted")
    ax1.legend(title="anomaly_iso")
    st.pyplot(fig1)
    plt.close(fig1)

    static_img = os.path.join(data_dir, "anomaly.png")
    if os.path.exists(static_img):
        st.image(static_img, caption="Precomputed anomaly visualization", use_container_width=True)

    st.subheader("🔍 SHAP Explanation for Selected Anomaly (IsolationForest)")
    anomaly_indices = df_anomaly.index[df_anomaly["anomaly_iso"] == 1].tolist() if "anomaly_iso" in df_anomaly.columns else []
    if not anomaly_indices:
        st.info("No IsolationForest anomalies detected.")
    else:
        try:
            if os.path.exists(os.path.join(data_dir, iso_model_file)):
                iso_model = joblib.load(os.path.join(data_dir, iso_model_file))
                st.success("✅ Loaded pre-trained IsolationForest model.")
            else:
                from sklearn.ensemble import IsolationForest
                iso_model = IsolationForest(contamination=0.05, random_state=42).fit(X)
                st.info("Fitted a new IsolationForest model (no saved model found).")
        except Exception as e:
            st.warning(f"Could not load or fit IsolationForest: {e}")
            st.stop()

        if hasattr(iso_model, "feature_names_in_"):
            iso_features = list(iso_model.feature_names_in_)
        else:
            iso_features = list(X.columns)

        def iso_decision_fn(x_np):
            x_df = pd.DataFrame(x_np, columns=iso_features)
            return iso_model.decision_function(x_df[iso_features])

        selected_idx = st.selectbox("Select anomaly index for SHAP analysis:", options=anomaly_indices)
        sample = X.loc[[selected_idx], iso_features]
        background = X[iso_features].sample(n=min(100, len(X)), random_state=42)
        try:
            explainer = shap.Explainer(iso_decision_fn, masker=shap.maskers.Independent(background), feature_names=iso_features)
            shap_values = explainer(sample.to_numpy())
            st.markdown(f"**Explaining anomaly index:** `{selected_idx}`")
            fig2, ax2 = plt.subplots(figsize=(9, 6))
            shap.plots.waterfall(shap_values[0], max_display=15, show=False)
            st.pyplot(fig2)
            plt.close(fig2)
        except Exception as e:
            st.error(f"Error computing SHAP: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# TAB 5: Clustering
with tabs[4]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("🧩 Clustering Visualization")
    data_path = os.path.join(data_dir, "data3.csv")
    if not os.path.exists(data_path):
        st.error("❌ data3.csv not found in app folder.")
    else:
        df = pd.read_csv(data_path)
        if "composition_cluster" not in df.columns:
            st.warning("No 'composition_cluster' column found.")
        else:
            st.subheader("📊 Cluster Summary")
            cluster_counts = df["composition_cluster"].value_counts().sort_index()
            st.dataframe(cluster_counts.to_frame("count"), use_container_width=True)
            if "JV_default_PCE" in df.columns:
                st.write("**Mean JV_default_PCE per Cluster:**")
                st.dataframe(df.groupby("composition_cluster")["JV_default_PCE"].mean().to_frame("mean_PCE"), use_container_width=True)
            static_img = os.path.join(data_dir, "plot.png")
            if os.path.exists(static_img):
                st.image(static_img, caption="t-SNE Visualization of KMeans Clusters", use_container_width=True)
            else:
                st.info("No `plot.png` found — provide a visual for faster viewing.")
    st.markdown('</div>', unsafe_allow_html=True)

# TAB 6: RF Prediction + SHAP
with tabs[5]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("🌲 Random Forest – PCE Prediction & SHAP Analysis")
    st.write("Upload a preprocessed CSV (same features used to train RF) to predict PCE and compute SHAP explanations.")
    uploaded_file = st.file_uploader("Upload CSV for prediction (preprocessed)", type=["csv"], key="rf_predict_uploader")
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

        st.write("✅ Uploaded Data Preview:")
        edited_df = st.data_editor(X_user, num_rows="dynamic", use_container_width=True)

        if st.button("🔮 Predict PCE"):
            preds = rf_model.predict(edited_df)
            preds_df = pd.DataFrame({"Predicted_PCE": preds})
            st.subheader("📈 Predicted PCE Values")
            st.dataframe(preds_df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)

            # Download button for predictions
            csv_bytes = preds_df.to_csv(index=False).encode()
            st.download_button("⬇️ Download predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")

            st.subheader("🔍 SHAP Feature Contribution (single sample)")
            selected_index = st.number_input("Select row index for SHAP analysis:", 0, len(edited_df)-1, 0)
            sample = edited_df.iloc[[selected_index]]
            try:
                explainer = shap.TreeExplainer(rf_model)
                shap_values_single = explainer.shap_values(sample)
                st.markdown(f"**Explaining Sample Index:** `{selected_index}`")
                fig, ax = plt.subplots(figsize=(9, 4))
                shap.plots.waterfall(
                    shap.Explanation(
                        values=shap_values_single[0],
                        base_values=explainer.expected_value,
                        data=sample.iloc[0].values,
                        feature_names=sample.columns
                    ),
                    max_display=10, show=False
                )
                st.pyplot(fig)

                shap_importance = pd.Series(shap_values_single[0], index=sample.columns).abs().sort_values(ascending=False)
                fig2, ax2 = plt.subplots(figsize=(7, 4))
                shap_importance.head(10).plot(kind="barh", ax=ax2)
                ax2.set_xlabel("SHAP Value Magnitude (|impact|)")
                ax2.invert_yaxis()
                st.pyplot(fig2)
            except Exception as e:
                st.error(f"Error computing SHAP for RF: {e}")
    else:
        st.info("📎 Upload a CSV file to start predicting and analyzing PCE.")
    st.markdown('</div>', unsafe_allow_html=True)

# TAB 7: Data Visualization
with tabs[6]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    import seaborn as sns
    st.header("📊 Dataset Exploration & Visualization")
    uploaded_data_file = st.file_uploader("📤 Upload dataset (CSV) for exploration", type=["csv"], key="explore_uploader")
    if uploaded_data_file:
        df_vis = pd.read_csv(uploaded_data_file)
        st.success(f"✅ Loaded dataset with shape: {df_vis.shape}")
        st.subheader("🔍 Data Preview")
        st.dataframe(df_vis.head(), use_container_width=True)

        st.subheader("📈 Dataset Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Samples", len(df_vis))
        with col2:
            st.metric("Features", df_vis.shape[1])
        with col3:
            st.metric("Missing Values", int(df_vis.isna().sum().sum()))

        st.subheader("📋 Summary Statistics")
        st.dataframe(df_vis.describe().T.style.highlight_max(color='lightgreen', axis=0), use_container_width=True)

        st.subheader("📊 Feature Distribution")
        num_cols = df_vis.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) == 0:
            st.warning("No numeric columns found.")
        else:
            selected_feature = st.selectbox("Select feature:", num_cols, key="dist_feature")
            fig1, ax1 = plt.subplots(figsize=(10, 4))
            sns.histplot(df_vis[selected_feature].dropna(), kde=True, ax=ax1)
            ax1.set_title(f"Distribution of {selected_feature}")
            st.pyplot(fig1)

            st.subheader("🔥 Correlation Heatmap")
            corr = df_vis[num_cols].corr()
            fig2, ax2 = plt.subplots(figsize=(12, 8))
            sns.heatmap(corr, center=0, annot=True, fmt=".2f", linewidths=0.4, ax=ax2)
            ax2.set_title("Feature Correlation Matrix")
            st.pyplot(fig2)

            st.subheader("🔗 Pairwise Relationships")
            pair_features = st.multiselect("Select up to 4 features for pairplot:", num_cols, default=num_cols[:min(4, len(num_cols))], max_selections=4)
            if len(pair_features) >= 2:
                st.info("Generating pairplot (may take a few seconds)...")
                pairplot_fig = sns.pairplot(df_vis[pair_features].dropna(), diag_kind="kde", plot_kws={'alpha': 0.6})
                pairplot_fig.fig.set_size_inches(10, 10)
                st.pyplot(pairplot_fig.fig)
            else:
                st.warning("Select at least 2 numeric features to show pairwise relationships.")

        st.subheader("📦 Boxplot for Feature Distribution by Category")
        cat_cols = df_vis.select_dtypes(exclude=[np.number]).columns.tolist()
        if cat_cols:
            cat_col = st.selectbox("Select categorical feature (x-axis):", cat_cols, key="box_cat")
            num_col = st.selectbox("Select numeric feature (y-axis):", num_cols, key="box_num")
            if df_vis[cat_col].nunique() > 20:
                st.warning(f"{cat_col} has {df_vis[cat_col].nunique()} unique categories — showing top 20 by frequency.")
                top_cats = df_vis[cat_col].value_counts().nlargest(20).index
                df_box = df_vis[df_vis[cat_col].isin(top_cats)].copy()
            else:
                df_box = df_vis.copy()
            fig4, ax4 = plt.subplots(figsize=(12, 6))
            sns.boxplot(data=df_box, x=cat_col, y=num_col, ax=ax4, fliersize=3)
            ax4.set_title(f"{num_col} Distribution by {cat_col}")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig4)
        else:
            st.info("No categorical columns found for boxplot.")
    else:
        st.info("Upload a CSV file to begin exploring your dataset.")
    st.markdown('</div>', unsafe_allow_html=True)

