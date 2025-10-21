# precompute_shap.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import shap
import joblib
import pickle

# Preprocess data
def preprocess_data():
    df = pd.read_csv("Solar Cell data set.csv")
    
    numerical_cols = ['Perovskite_thickness', 'Perovskite_additives_concentrations', 
                     'Perovskite_deposition_thermal_annealing_temperature',
                     'Perovskite_deposition_thermal_annealing_time', 'Backcontact_thickness_list',
                     'JV_default_Voc', 'JV_default_Jsc', 'JV_default_FF', 'JV_default_PCE']
    
    categorical_cols = ['Cell_stack_sequence', 'Cell_architecture', 'Substrate_stack_sequence',
                       'ETL_stack_sequence', 'ETL_deposition_procedure', 'Perovskite_composition_a_ions',
                       'Perovskite_composition_b_ions', 'Perovskite_composition_c_ions',
                       'Perovskite_composition_short_form', 'Perovskite_additives_compounds',
                       'Perovskite_deposition_procedure', 'Perovskite_deposition_solvents',
                       'HTL_stack_sequence', 'HTL_deposition_procedure', 'Backcontact_stack_sequence',
                       'Backcontact_deposition_procedure', 'Encapsulation_stack_sequence']
    
    boolean_cols = ['Perovskite_deposition_quenching_induced_crystallisation', 'Encapsulation', 'JV_measured']
    
    def clean_numerical(value):
        if pd.isna(value) or value in ["nan", "None", "", "unknown"]:
            return np.nan
        if isinstance(value, str):
            for sep in ['|', ',', ';']:
                if sep in value:
                    values = [val.strip() for val in value.split(sep) if val.strip().lower() not in ['nan', 'none', '']]
                    try:
                        values = [float(val) for val in values if val]
                        return np.mean(values) if values else np.nan
                    except (ValueError, TypeError):
                        return np.nan
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return np.nan
        return float(value)
    
    for col in numerical_cols:
        df[col] = df[col].apply(clean_numerical).astype(float)
    
    # Imputation
    imp = IterativeImputer(random_state=0)
    df[numerical_cols] = imp.fit_transform(df[numerical_cols])
    
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    
    for col in boolean_cols:
        df[col] = df[col].astype(bool).astype(int)
    
    df = df.dropna(subset=['JV_default_PCE'])
    
    # Feature Engineering
    a_ions = ['Cs', 'FA', 'MA', 'Rb', 'K']
    b_ions = ['Pb', 'Sn']
    c_ions = ['I', 'Br', 'Cl']
    
    def parse_short_form(formula):
        features = {}
        for ion in a_ions + b_ions + c_ions:
            features[f'has_{ion}'] = 1 if ion in str(formula) else 0
        return features
    
    parsed_df = df['Perovskite_composition_short_form'].apply(parse_short_form).apply(pd.Series)
    df = pd.concat([df, parsed_df], axis=1)
    
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', max_categories=10)
    encoded_cats = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out())
    df = pd.concat([df, encoded_df], axis=1).drop(categorical_cols, axis=1)
    
    scaler = StandardScaler()
    df[numerical_cols[:-4]] = scaler.fit_transform(df[numerical_cols[:-4]])
    
    # Save processed data
    df.to_csv("processed_solar_data.csv", index=False)
    joblib.dump(encoder, "encoder.pkl")
    joblib.dump(scaler, "scaler.pkl")
    
    return df, encoder, scaler

# Preprocess data
df, encoder, scaler = preprocess_data()

# Prepare features and target
y = df['JV_default_PCE']
X = df.drop(['JV_default_Voc', 'JV_default_Jsc', 'JV_default_FF', 'JV_default_PCE'], axis=1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model (to match scikit-learn 1.7.2)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Load existing XGBoost model
xgb_model = xgb.Booster()
xgb_model.load_model("xgb_model.json")

# Save new Random Forest model
joblib.dump(rf_model, "rf_model_new.pkl")

# Compute SHAP values
rf_explainer = shap.TreeExplainer(rf_model, X_train)
rf_shap_values = rf_explainer.shap_values(X_test)

xgb_explainer = shap.Explainer(xgb_model, X_train)
xgb_shap_values = xgb_explainer(X_test)

# Save SHAP values and test data
with open("rf_shap_values.pkl", "wb") as f:
    pickle.dump(rf_shap_values, f)
with open("xgb_shap_values.pkl", "wb") as f:
    pickle.dump(xgb_shap_values, f)
with open("X_test.pkl", "wb") as f:
    pickle.dump(X_test, f)
with open("y_test.pkl", "wb") as f:
    pickle.dump(y_test, f)

print("Preprocessing, Random Forest training, and SHAP computation completed. Files saved.")
