# train_solarcell.py
import os
import joblib
import json
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import shap

# --- Config ---
DATA_PATH = "Solar Cell data set.csv"
OUT_DIR = "models_artifacts"
os.makedirs(OUT_DIR, exist_ok=True)
RANDOM_STATE = 42

# --- Helper functions (same logic as your notebook) ---
def clean_numerical(value):
    import numpy as _np
    if pd.isna(value) or value in ["nan", "None", "", "unknown"]:
        return _np.nan
    if isinstance(value, str):
        for sep in ['|', ',', ';']:
            if sep in value:
                vals = [v.strip() for v in value.split(sep) if v.strip().lower() not in ['nan','none','']]
                try:
                    vals = [float(v) for v in vals if v]
                    return float(np.mean(vals)) if vals else _np.nan
                except Exception:
                    return _np.nan
        try:
            return float(value)
        except Exception:
            return _np.nan
    try:
        return float(value)
    except Exception:
        return _np.nan

def parse_composition_short_form_series(s):
    a_ions = ['Cs', 'FA', 'MA', 'Rb', 'K']
    b_ions = ['Pb', 'Sn']
    c_ions = ['I', 'Br', 'Cl']
    def parse(formula):
        if pd.isna(formula):
            formula = ""
        return {f'has_{ion}': int(ion in str(formula)) for ion in a_ions + b_ions + c_ions}
    parsed = s.fillna("").apply(parse).apply(pd.Series)
    return parsed

# --- Load data safely ---
print("Loading data...")
df = pd.read_csv(DATA_PATH, low_memory=False)
print("Rows:", len(df), "Cols:", len(df.columns))

# --- Basic cleaning & numeric parsing ---
numerical_cols = ['Perovskite_thickness',"Perovskite_additives_concentrations", 
                  'Perovskite_deposition_thermal_annealing_temperature',
                  'Perovskite_deposition_thermal_annealing_time', 'Backcontact_thickness_list',
                  'JV_default_Voc', 'JV_default_Jsc', 'JV_default_FF', 'JV_default_PCE']
for c in numerical_cols:
    if c in df.columns:
        df[c] = df[c].apply(clean_numerical)

# Boolean-ish
boolean_cols = ['Perovskite_deposition_quenching_induced_crystallisation', 'Encapsulation', 'JV_measured']
for c in boolean_cols:
    if c in df.columns:
        df[c] = df[c].map(lambda x: 1 if str(x).strip().lower() in ['1','true','yes','y','t']
                          else (0 if str(x).strip().lower() in ['0','false','no','n','f'] else np.nan))

# Drop rows without target
df = df.dropna(subset=['JV_default_PCE'])
print("After dropping missing PCE:", len(df))

# --- Parse composition short form
if 'Perovskite_composition_short_form' in df.columns:
    parsed = parse_composition_short_form_series(df['Perovskite_composition_short_form'])
    df = pd.concat([df.reset_index(drop=True), parsed.reset_index(drop=True)], axis=1)

# --- Categorical selection
categorical_cols = [c for c in [
    'Cell_stack_sequence', 'Cell_architecture', 'Substrate_stack_sequence',
    'ETL_stack_sequence', 'ETL_deposition_procedure', 'Perovskite_composition_a_ions',
    'Perovskite_composition_b_ions', 'Perovskite_composition_c_ions',
    'Perovskite_composition_short_form', 'Perovskite_additives_compounds',
    'Perovskite_deposition_procedure', 'Perovskite_deposition_solvents',
    'HTL_stack_sequence', 'HTL_deposition_procedure', 'Backcontact_stack_sequence',
    'Backcontact_deposition_procedure', 'Encapsulation_stack_sequence'] if c in df.columns]

# --- Numeric impute
numeric_cols_present = df.select_dtypes(include=[np.number]).columns.tolist()
print("Numeric columns:", numeric_cols_present)
if numeric_cols_present:
    try:
        imp = IterativeImputer(random_state=RANDOM_STATE, max_iter=10)
        df[numeric_cols_present] = imp.fit_transform(df[numeric_cols_present])
    except Exception as e:
        print("IterativeImputer failed, using mean imputer:", e)
        df[numeric_cols_present] = SimpleImputer(strategy='mean').fit_transform(df[numeric_cols_present])

# --- Categorical impute & encode ---
if categorical_cols:
    df[categorical_cols] = SimpleImputer(strategy='most_frequent').fit_transform(df[categorical_cols])
    # OneHotEncoder compatibility
    try:
        enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore', max_categories=10)
    except TypeError:
        enc = OneHotEncoder(sparse=False, handle_unknown='ignore', max_categories=10)
    enc_arr = enc.fit_transform(df[categorical_cols])
    enc_cols = enc.get_feature_names_out(categorical_cols)
    df_enc = pd.DataFrame(enc_arr, columns=enc_cols, index=df.index)
    df = pd.concat([df.drop(columns=categorical_cols), df_enc], axis=1)
else:
    enc = None

# --- Fill any remaining na
df = df.fillna(df.median(numeric_only=True)).fillna("missing")

# --- Feature / target split
target = 'JV_default_PCE'
drop_cols = [c for c in ['JV_default_Voc','JV_default_Jsc','JV_default_FF', target] if c in df.columns]
X = df.drop(columns=drop_cols)
y = df[target].astype(float)

# Save feature list (column order)
feature_columns = X.columns.tolist()
joblib.dump(feature_columns, os.path.join(OUT_DIR, "feature_columns.pkl"))
print("Saved feature column list.")

# --- Scaling (fit scaler on numeric columns only) ---
num_cols = [c for c in feature_columns if df[c].dtype in [np.float64, np.int64]]
scaler = StandardScaler()
if num_cols:
    X[num_cols] = scaler.fit_transform(X[num_cols])
joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.pkl"))
print("Saved scaler.")

# Save encoder if exists
if enc is not None:
    joblib.dump(enc, os.path.join(OUT_DIR, "onehot_encoder.pkl"))
    print("Saved OneHotEncoder.")

# --- Train/test split (you can adjust test_size) ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
print("Train size:", len(X_train), "Test size:", len(X_test))

# --- RandomForest ---
print("Training RandomForest...")
rf = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("RF RMSE:", np.sqrt(mean_squared_error(y_test, rf_pred)), "R2:", r2_score(y_test, rf_pred))
joblib.dump(rf, os.path.join(OUT_DIR, "rf_model.pkl"))
print("Saved RandomForest model.")

# --- XGBoost (native train) ---
print("Training XGBoost (native)...")
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.01,
    'max_depth': 8,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_lambda': 1.5,
    'tree_method': 'hist',
    'eval_metric': 'rmse',
    'seed': RANDOM_STATE,
    'nthread': 1
}
num_round = 500
xgb_model = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_round, evals=[(dtest, 'test')],
                      early_stopping_rounds=30, verbose_eval=50)
xgb_pred = xgb_model.predict(dtest)
print("XGB RMSE:", np.sqrt(mean_squared_error(y_test, xgb_pred)), "R2:", r2_score(y_test, xgb_pred))
xgb_model.save_model(os.path.join(OUT_DIR, "xgb_model.json"))
print("Saved XGBoost Booster to JSON.")

# --- Save background sample for SHAP (small) ---
bg = X_train.sample(n=min(200, len(X_train)), random_state=RANDOM_STATE)
joblib.dump(bg, os.path.join(OUT_DIR, "shap_background.pkl"))
print("Saved SHAP background sample.")

# --- Precompute global SHAP for XGBoost and save PNGs ---
print("Computing SHAP (global) for XGBoost. This may take a bit...")
def xgb_predict_for_shap(X_in):
    d = xgb.DMatrix(X_in)
    return xgb_model.predict(d)

explainer = shap.Explainer(xgb_predict_for_shap, bg)
# use a modest sample for speed
X_shap_sample = X_test.sample(n=min(200, len(X_test)), random_state=RANDOM_STATE)
shap_values = explainer(X_shap_sample)

# summary plot
plt.figure(figsize=(10,6))
shap.summary_plot(shap_values, X_shap_sample, show=False)
plt.title("SHAP Summary (XGBoost)")
plt.savefig(os.path.join(OUT_DIR, "shap_summary_xgb.png"), bbox_inches='tight')
plt.close()

# bar plot
plt.figure(figsize=(10,6))
shap.plots.bar(shap_values, show=False)
plt.title("SHAP Mean |SHAP| (XGBoost)")
plt.savefig(os.path.join(OUT_DIR, "shap_bar_xgb.png"), bbox_inches='tight')
plt.close()
print("Saved SHAP global images.")

# --- Save metadata (e.g., columns, params, etc.) ---
meta = {
    'feature_columns': feature_columns,
    'numeric_columns': num_cols,
    'target': target,
    'xgb_params': params,
    'random_state': RANDOM_STATE
}
with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("Training complete — artifacts saved in", OUT_DIR)
