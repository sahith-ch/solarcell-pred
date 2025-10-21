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
st.set_page_config(page_title="Solar Cell Model Dashboard", layout="wide")
st.title("🔆 Solar Cell ML Model Evaluation Dashboard")
import os
import streamlit as st

st.write("Current working directory:", os.getcwd())
st.write("Files in current directory:", os.listdir())

# -----------------------------
# Input: Folder Path
# -----------------------------
folder_path = st.text_input(
    "📁 Enter folder path containing x_test_processed.csv and y_test_processed.csv (leave blank for current directory):",
    ""
)
data_dir = folder_path if folder_path else "."
x_test_file = os.path.join(data_dir, "x_test_processed.csv")
y_test_file = os.path.join(data_dir, "y_test_processed.csv")

# -----------------------------
# Load Models
# -----------------------------
try:
    rf_model = joblib.load("rf_model.pkl")
    xgb_model = xgb.Booster()
    xgb_model.load_model("xgb_model.json")
    st.sidebar.success("✅ Models loaded successfully.")
except Exception as e:
    st.sidebar.error(f"❌ Failed to load models: {e}")
    st.stop()

# -----------------------------
# Load Data
# -----------------------------
if not (os.path.exists(x_test_file) and os.path.exists(y_test_file)):
    st.error("❌ Missing test CSV files. Please check folder path.")
    st.stop()

X_test = pd.read_csv(x_test_file)
y_test = pd.read_csv(y_test_file)
if y_test.shape[1] == 1:
    y_test = y_test.iloc[:, 0]

# -----------------------------
# Feature Alignment
# -----------------------------
def align_features(model, X):
    if hasattr(model, "feature_names_in_"):
        model_features = model.feature_names_in_
        aligned = pd.DataFrame(0, index=X.index, columns=model_features)
        common = X.columns.intersection(model_features)
        aligned[common] = X[common]
        return aligned
    return X

X_test = align_features(rf_model, X_test)

# -----------------------------
# Compute Predictions
# -----------------------------
rf_pred = rf_model.predict(X_test)
dtest = xgb.DMatrix(X_test)
xgb_pred = xgb_model.predict(dtest)

# -----------------------------
# Metrics Table
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
st.subheader("📈 Model Performance Summary")
st.dataframe(df_metrics.style.highlight_min(color='lightcoral', subset=["RMSE", "MAE"])
             .highlight_max(color='lightgreen', subset=["R²"])
             .format("{:.4f}"))

# -----------------------------
# Tabs for Plots
# -----------------------------
tabs = st.tabs(["📊 True vs Predicted", "🌫️ RF Uncertainty", "🔥 SHAP Feature Importance","Anamoly","Clustering","Prediction","Data Visualization"])

# ================================================
# TAB 1: TRUE vs PREDICTED
# ================================================
with tabs[0]:
    st.subheader("True vs Predicted")
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    # RF Plot
    ax[0].scatter(y_test, rf_pred, alpha=0.6, color="#0072B2")
    ax[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    ax[0].set_title("Random Forest")
    ax[0].set_xlabel("True PCE")
    ax[0].set_ylabel("Predicted PCE")
    # XGB Plot
    ax[1].scatter(y_test, xgb_pred, alpha=0.6, color="#E69F00")
    ax[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    ax[1].set_title("XGBoost")
    ax[1].set_xlabel("True PCE")
    ax[1].set_ylabel("Predicted PCE")
    st.pyplot(fig)

# ================================================
# TAB 2: RANDOM FOREST UNCERTAINTY
# ================================================
with tabs[1]:
    st.subheader("🌫️ Random Forest Prediction Uncertainty")

    try:
        all_tree_preds = np.array([tree.predict(X_test.values) for tree in rf_model.estimators_])
        y_pred_mean = np.mean(all_tree_preds, axis=0)
        y_pred_std = np.std(all_tree_preds, axis=0)

        threshold = np.percentile(y_pred_std, 90)
        mask_high = y_pred_std > threshold

        st.markdown(f"- **Average uncertainty (std):** `{np.mean(y_pred_std):.3f}`")
        st.markdown(f"- **Top 10% high uncertainty threshold:** `{threshold:.3f}`")

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.errorbar(
            y_test, y_pred_mean, yerr=y_pred_std,
            fmt='o', ecolor='lightcoral', alpha=0.5, label='Predictions ± Uncertainty'
        )
        ax.scatter(
            y_test[mask_high], y_pred_mean[mask_high],
            color='orange', s=70, edgecolor='k', label='High Uncertainty'
        )
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
        ax.set_xlabel("True PCE")
        ax.set_ylabel("Predicted PCE")
        ax.set_title("Random Forest Predictions with Uncertainty")
        ax.legend()
        st.pyplot(fig)

        # Show uncertain samples
        st.write("Top uncertain predictions:")
        top_uncertain = pd.DataFrame({
            "True PCE": y_test[mask_high].values,
            "Predicted PCE": y_pred_mean[mask_high],
            "Std (Uncertainty)": y_pred_std[mask_high],
        }).sort_values("Std (Uncertainty)", ascending=False)
        st.dataframe(top_uncertain.head(10).style.format("{:.3f}"))

    except Exception as e:
        st.error(f"Error computing uncertainty: {e}")

# ================================================
# TAB 3: SHAP FEATURE IMPORTANCE
# ================================================
with tabs[2]:
    import streamlit as st
    from PIL import Image
    import os

    st.header("🔍 Model Explainability (SHAP Summary Plot)")

    # Path to your pre-generated SHAP summary image
    shap_image_path = "shap.png"  # change to your filename if needed

    if os.path.exists(shap_image_path):
        image = Image.open(shap_image_path)
        st.image(image, caption="SHAP Feature Importance — Global Explanation", use_container_width=True)
        st.markdown("""
        **Interpretation:**
        - Each dot is a sample; color = feature value (red = high, blue = low).
        - Position on the x-axis shows impact on model output (PCE ↑ or ↓).
        - Features higher on the chart are more influential on predictions.
        """)
    else:
        st.error(f"SHAP summary plot not found at `{shap_image_path}`.")
        st.info("Please ensure the SHAP image is saved in the app folder.")

# ---------- Anomaly Detection ----------
# --- TAB 3: Anomaly Detection + SHAP (Index Based) ---
# --- TAB 3: Anomaly Detection + SHAP (IsolationForest + AE Comparison) ---
# ---------- TAB 3: Anomaly Detection ----------
with tabs[3]:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import os
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import shap

    st.header("🚨 Anomaly Detection Visualization & Explanation")

    # --- Load dataset ---
    uploaded_anomaly_file = st.file_uploader("Upload dataset with anomaly columns", type=["csv"])
    if uploaded_anomaly_file:
        df_anomaly = pd.read_csv(uploaded_anomaly_file)
    else:
        default_path = "data.csv"  # your saved anomaly dataset
        if os.path.exists(default_path):
            df_anomaly = pd.read_csv(default_path)
        else:
            st.warning("Upload or provide a dataset containing anomaly columns.")
            st.stop()

    st.write("Dataset shape:", df_anomaly.shape)
    st.dataframe(df_anomaly.head())

    # --- Detect anomaly columns ---
    anomaly_cols = [col for col in df_anomaly.columns if "anomaly" in col.lower()]
    if not anomaly_cols:
        st.error("No anomaly columns found (expected 'anomaly_iso', 'anomaly_svm', 'anomaly_ae').")
        st.stop()

    # ✅ Ensure anomaly columns are numeric (and fix sign)
    for col in anomaly_cols:
        df_anomaly[col] = pd.to_numeric(df_anomaly[col], errors="coerce").fillna(0).astype(int)
        # Normalize: make sure anomalies = 1, normal = 0
        if df_anomaly[col].min() == -1:
            df_anomaly[col] = df_anomaly[col].replace({-1: 1, 1: 0})

    st.subheader("📊 Anomaly Counts")
    st.write(df_anomaly[anomaly_cols].sum())

    # --- Prepare feature data ---
    feature_cols = [c for c in df_anomaly.columns if c not in anomaly_cols]
    X = df_anomaly[feature_cols].select_dtypes(include=[float, int]).fillna(0)

    # --- Choose visualization method ---
    method = st.radio("Choose visualization method:", ["PCA", "t-SNE"])
    reducer = PCA(n_components=2, random_state=42) if method == "PCA" else TSNE(n_components=2, perplexity=30, random_state=42)

    with st.spinner(f"Computing {method} embedding..."):
        emb = reducer.fit_transform(X)
    df_emb = pd.DataFrame(emb, columns=["dim1", "dim2"])
    df_emb["anomaly_iso"] = df_anomaly["anomaly_iso"] if "anomaly_iso" in df_anomaly.columns else 0

    # --- t-SNE / PCA Plot ---
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        x="dim1", y="dim2", hue="anomaly_iso",
        data=df_emb, palette={0: "gray", 1: "red"},
        alpha=0.6, ax=ax1
    )
    ax1.set_title(f"{method} Projection — IsolationForest Anomalies Highlighted")
    st.pyplot(fig1)
    plt.close(fig1)

    # --- Optional precomputed image (static demo) ---
    static_img = "anomaly.png"
    if os.path.exists(static_img):
        st.image(static_img, caption="Precomputed anomaly visualization", use_container_width=True)

    # --- SHAP Analysis for Selected Anomaly ---
    st.subheader("🔍 SHAP Explanation for Selected Anomaly (IsolationForest)")
    anomaly_indices = df_anomaly.index[df_anomaly["anomaly_iso"] == 1].tolist()
    if not anomaly_indices:
        st.info("No IsolationForest anomalies detected in dataset.")
        st.stop()


    # Load your saved IsolationForest model (pkl)
    import pickle
    model_path = "iso_model.pkl"
    if os.path.exists(model_path):
        from joblib import load

        iso_model = load("iso_model.pkl")
        ocsvm_model = load("ocsvm_model.pkl")

        st.success("✅ Loaded pre-trained IsolationForest model.")
    else:
        st.warning("No iso_forest.pkl found — using a new IsolationForest model (less accurate).")
        from sklearn.ensemble import IsolationForest
        iso_model = IsolationForest(contamination=0.05, random_state=42).fit(X)

# Only use features seen during training
    iso_features = iso_model.feature_names_in_

    def iso_decision_fn(x_np):
        x_df = pd.DataFrame(x_np, columns=iso_features)
        return iso_model.decision_function(x_df[iso_features])

    selected_idx = st.selectbox("Select anomaly index for SHAP analysis:", options=anomaly_indices)
    sample = X.loc[[selected_idx], iso_features]


    # Create SHAP explainer
    background = X[iso_features].sample(n=min(100, len(X)), random_state=42)
    explainer = shap.Explainer(iso_decision_fn, masker=shap.maskers.Independent(background), feature_names=iso_features)

    shap_values = explainer(sample.to_numpy())

    # --- Waterfall plot isolated ---
    st.markdown(f"**Explaining anomaly index:** `{selected_idx}`")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    shap.plots.waterfall(shap_values[0], max_display=15, show=False)
    st.pyplot(fig2)
    plt.close(fig2)
with tabs[4]:
    import pandas as pd
    import streamlit as st
    import os

    st.header("🧩 Clustering Visualization")

    data_path = "data3.csv"
    if not os.path.exists(data_path):
        st.error("❌ data.csv not found.")
        st.stop()

    df = pd.read_csv(data_path)

    # Check for clustering results
    if "composition_cluster" not in df.columns:
        st.warning("⚠️ No 'cluster' column found in data. Please ensure clustering has been performed.")
        st.stop()

    # ---- Cluster stats ----
    st.subheader("📊 Cluster Summary")
    cluster_counts = df["composition_cluster"].value_counts().sort_index()
    st.write("**Cluster Counts:**")
    st.dataframe(cluster_counts.to_frame("count"))

    if "JV_default_PCE" in df.columns:
        st.write("**Mean JV_default_PCE per Cluster:**")
        st.dataframe(df.groupby("composition_cluster")["JV_default_PCE"].mean().to_frame("mean_PCE"))
    else:
        st.info("Column 'JV_default_PCE' not found for mean calculation.")

    # ---- Static Visualization ----
    static_img = "plot.png"  # <-- your PNG file here
    if os.path.exists(static_img):
        st.image(static_img, caption="t-SNE Visualization of KMeans Clusters", use_container_width=True)
    else:
        st.warning(f"⚠️ Could not find {static_img}. Please ensure the image is saved in the same directory.")

    st.markdown("The visualization above shows how different KMeans clusters group together in t-SNE space.")
# with tabs[5]:
#     import torch
#     import torch.nn as nn
#     import pandas as pd
#     import numpy as np
#     import streamlit as st
#     import shap
#     import matplotlib.pyplot as plt
#     from sklearn.preprocessing import StandardScaler

#     st.header("🔮 Joint Prediction – Hybrid Model for Solar Cell Performance")

#     # ===== Model Definition (must match training) =====
#     class HybridJointModel(nn.Module):
#         def __init__(self, input_dim, output_dim=4):
#             super().__init__()
#             self.shared = nn.Sequential(
#                 nn.Linear(input_dim, 512),
#                 nn.BatchNorm1d(512),
#                 nn.ReLU(),
#                 nn.Linear(512, 256),
#                 nn.BatchNorm1d(256),
#                 nn.ReLU(),
#                 nn.Linear(256, 128),
#                 nn.ReLU()
#             )
#             self.task_interaction = nn.Linear(128, 128)
#             self.input_skip = nn.Linear(input_dim, 128)
#             self.heads = nn.ModuleList([nn.Linear(128, 1) for _ in range(output_dim)])
#             self.log_vars = nn.Parameter(torch.zeros(output_dim))

#         def forward(self, x):
#             shared = self.shared(x)
#             shared = shared + self.input_skip(x)
#             h = shared + torch.tanh(self.task_interaction(shared))
#             outputs = [head(h) for head in self.heads]
#             return torch.cat(outputs, dim=1)

#     # ===== Load Model and Scaler =====
#     model = HybridJointModel(X.shape[1]).to("cpu")
#     model.load_state_dict(torch.load("hybrid_joint_model.pth", map_location="cpu"))
#     model.eval()

#     scaler_y = StandardScaler()
#     scaler_y.fit(df[['JV_default_Voc', 'JV_default_Jsc', 'JV_default_FF', 'JV_default_PCE']])
#     targets = ['JV_default_Voc', 'JV_default_Jsc', 'JV_default_FF', 'JV_default_PCE']

#     # ===== File Upload =====
#     st.subheader("📤 Upload CSV for Prediction")
#     uploaded_file = st.file_uploader("Upload a preprocessed CSV file (matching training columns)", type=["csv"])

#     if uploaded_file:
#         user_df = pd.read_csv(uploaded_file)

#         # Ensure same columns as training data
#         missing_cols = set(X.columns) - set(user_df.columns)
#         if missing_cols:
#             st.error(f"Missing columns in uploaded file: {missing_cols}")
#             st.stop()

#         # Reorder columns to match training
#         user_df = user_df[X.columns]

#         st.write("✅ Uploaded Data Preview:")
#         edited_df = st.data_editor(user_df, num_rows="dynamic", use_container_width=True)

#         # Predict Button
#         if st.button("🔮 Predict Performance"):
#             input_tensor = torch.tensor(edited_df.values, dtype=torch.float32)
#             with torch.no_grad():
#                 scaled_preds = model(input_tensor).numpy()
#                 preds = scaler_y.inverse_transform(scaled_preds)

#             preds_df = pd.DataFrame(preds, columns=targets)
#             st.subheader("📈 Predicted Performance Metrics")
#             st.dataframe(preds_df.style.highlight_max(axis=0, color='lightgreen'))

#             # Allow user to pick a sample for SHAP
#             # Allow user to pick a sample for SHAP
#             st.subheader("🔍 SHAP Explanation")
#             selected_index = st.number_input("Select row index for SHAP analysis:", 0, len(edited_df)-1, 0)
#             # Explain SHAP for one selected sample
#             sample = edited_df.iloc[[selected_index]].values

#             def predict_fn(x):
#                 with torch.no_grad():
#                     return model(torch.tensor(x, dtype=torch.float32)).numpy()

#             # Use a smaller background for speed
#             background = X.sample(100, random_state=42).values
#             explainer = shap.Explainer(predict_fn, background)

#             # Compute SHAP values for the selected row only
#             shap_values = explainer(sample)

#             # --- Handle multi-output case manually ---
#             # shap_values.values shape: (1, n_features, n_outputs)
#             # Extract SHAP values for PCE (output index 3)
#             shap_values_pce = shap_values.values[:, :, 3]
#             base_value_pce = shap_values.base_values[:, 3]

#             # Wrap it into a SHAP Explanation object for plotting
#             pce_explanation = shap.Explanation(
#                 values=shap_values_pce,
#                 base_values=base_value_pce,
#                 data=sample,
#                 feature_names=list(X.columns)
#             )

#             # Waterfall plot for this row
#             fig, ax = plt.subplots(figsize=(8, 4))
#             shap.plots.waterfall(pce_explanation[0], max_display=10, show=False)
#             st.pyplot(fig)



#     else:
#         st.info("Please upload a CSV file to begin predictions.")
with tabs[5]:
    import pandas as pd
    import numpy as np
    import streamlit as st
    import shap
    import matplotlib.pyplot as plt

    st.header("🌲 Random Forest – PCE Prediction & SHAP Analysis")

    # ⚙️ Reuse rf_model loaded earlier
    try:
        rf_model  # already loaded above
        st.success("✅ Random Forest model loaded from main section.")
    except NameError:
        st.error("❌ RF model not found. Make sure rf_model is loaded earlier.")
        st.stop()

    # ===== Upload User Data =====
    st.subheader("📤 Upload CSV for PCE Prediction")
    uploaded_file = st.file_uploader("Upload CSV file (preprocessed, same features as training)", type=["csv"])

    if uploaded_file:
        user_df = pd.read_csv(uploaded_file)

        # Align with training features
        if hasattr(rf_model, "feature_names_in_"):
            model_features = rf_model.feature_names_in_
            X_user = pd.DataFrame(0, index=user_df.index, columns=model_features)
            for col in user_df.columns:
                if col in model_features:
                    X_user[col] = user_df[col]
        else:
            X_user = user_df.copy()

        st.write("✅ Uploaded Data Preview:")
        edited_df = st.data_editor(X_user, num_rows="dynamic", use_container_width=True)

        # ===== Prediction =====
        if st.button("🔮 Predict PCE"):
            preds = rf_model.predict(edited_df)
            preds_df = pd.DataFrame({"Predicted_PCE": preds})
            st.subheader("📈 Predicted PCE Values")
            st.dataframe(preds_df.style.highlight_max(axis=0, color='lightgreen'))

            # ===== SHAP ANALYSIS =====
            # ===== SHAP ANALYSIS (single-row) =====
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
        st.info("📎 Upload a CSV file to start predicting and analyzing PCE.")
with tabs[6]:
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import streamlit as st

    st.header("📊 Dataset Exploration & Visualization")

    uploaded_data_file = st.file_uploader("📤 Upload your dataset (CSV)", type=["csv"])

    if uploaded_data_file:
        df_vis = pd.read_csv(uploaded_data_file)
        st.success(f"✅ Loaded dataset with shape: {df_vis.shape}")

        # --- Data Preview ---
        st.subheader("🔍 Data Preview")
        st.dataframe(df_vis.head())

        # --- Basic Info ---
        st.subheader("📈 Dataset Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Samples", len(df_vis))
        with col2:
            st.metric("Features", df_vis.shape[1])
        with col3:
            st.metric("Missing Values", int(df_vis.isna().sum().sum()))

        # --- Summary Statistics ---
        st.subheader("📋 Summary Statistics")
        st.dataframe(df_vis.describe().T.style.highlight_max(color='lightgreen', axis=0))

        # --- Distribution Plot ---
        st.subheader("📊 Feature Distribution")
        num_cols = df_vis.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) == 0:
            st.warning("No numeric columns found.")
            st.stop()

        selected_feature = st.selectbox("Select feature to visualize distribution:", num_cols)
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        sns.histplot(df_vis[selected_feature], kde=True, color="#1f77b4", ax=ax1)
        ax1.set_title(f"Distribution of {selected_feature}", fontsize=14, pad=10)
        ax1.set_xlabel(selected_feature)
        ax1.set_ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig1)

        # --- Correlation Heatmap ---
        st.subheader("🔥 Correlation Heatmap")
        corr = df_vis[num_cols].corr()

        fig2, ax2 = plt.subplots(figsize=(14, 10))
        sns.heatmap(
            corr,
            cmap="coolwarm",
            center=0,
            annot=True,           # ✅ Show correlation numbers
            fmt=".2f",
            annot_kws={"size": 8},
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax2
        )
        ax2.set_title("Feature Correlation Matrix", fontsize=16, pad=12)
        plt.tight_layout()
        st.pyplot(fig2)

        # --- Pairplot / Relationship ---
        st.subheader("🔗 Pairwise Relationships")
        pair_features = st.multiselect("Select up to 4 features for pairplot:", num_cols, default=num_cols[:min(4, len(num_cols))])
        if len(pair_features) >= 2:
            st.info("Generating pairplot... may take a few seconds ⏳")
            fig3 = sns.pairplot(df_vis[pair_features], diag_kind="kde", plot_kws={'alpha': 0.6})
            fig3.fig.set_size_inches(10, 10)
            st.pyplot(fig3)
        else:
            st.warning("Select at least 2 numeric features to show pairwise relationships.")

# --- Boxplot ---
        st.subheader("📦 Boxplot for Feature Distribution by Category")
        cat_cols = df_vis.select_dtypes(exclude=[np.number]).columns.tolist()

        if cat_cols:
            cat_col = st.selectbox("Select categorical feature (x-axis):", cat_cols)
            num_col = st.selectbox("Select numeric feature (y-axis):", num_cols)

            # Limit categories if too many
            if df_vis[cat_col].nunique() > 20:
                st.warning(f"⚠️ {cat_col} has {df_vis[cat_col].nunique()} unique categories — showing top 20 by frequency.")
                top_cats = df_vis[cat_col].value_counts().nlargest(20).index
                df_box = df_vis[df_vis[cat_col].isin(top_cats)].copy()
            else:
                df_box = df_vis.copy()

            fig4, ax4 = plt.subplots(figsize=(12, 6))
            sns.boxplot(data=df_box, x=cat_col, y=num_col, ax=ax4, palette="Set2", fliersize=3)
            ax4.set_title(f"{num_col} Distribution by {cat_col}", fontsize=14, pad=10)
            ax4.set_xlabel(cat_col, fontsize=12)
            ax4.set_ylabel(num_col, fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig4)

            st.caption("📉 Showing only top categories by frequency for better readability.")
        else:
            st.info("No categorical columns found for boxplot.")
