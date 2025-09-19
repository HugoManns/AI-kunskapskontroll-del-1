
import os
import streamlit as st
import pandas as pd
import joblib
import numpy as np


MODEL_FILE = "model_stl.pkl"
META = joblib.load(MODEL_FILE)

# 1. Modell-val
st.sidebar.title("Modell")
model_choice = st.sidebar.radio(
    "Välj modell för att beräkna sannolikheten",
    options=["Random Forest", "XGBoost", "Logistic Regression"],
    index=0,
)
if model_choice == "Random Forest":
    model = META["rf_model"]
elif model_choice == "XGBoost":
    model = META["xgb_model"]
else:
    model = META["lr_model"]

# 2. Läs kandidat-fil
csv_path = "candidates.csv"
if not os.path.exists(csv_path):
    st.info(f"Fil `{csv_path}` finns inte. Lägg in kandidater via *Lägg till kandidat*-sidan.")
    st.stop()

df_raw = pd.read_csv(csv_path, header=None)

X_FEATURES = META["features"]
X_FEATURES_LIST = list(X_FEATURES)


if len(df_raw.columns) != len(X_FEATURES):
    st.warning(
        "Stämmer ej"
    )
    missing = set(X_FEATURES) - set(df_raw.columns)
    for col in missing:
        df_raw[col] = 0
    df_raw = df_raw.reindex(columns=X_FEATURES)

df_raw.columns = X_FEATURES 


# “Employed”-kolumn
if "Employed" not in df_raw.columns:
    df_raw["Employed"] = "-"


# modellen
st.title("Inkomna kandidater")
st.write(f"Antal kandidater: **{len(df_raw)}**")
if len(df_raw) == 0:
    st.info("Ingen kandidat att visa.")
    st.stop()

X = df_raw[X_FEATURES].copy()
prob = model.predict_proba(X)[:, 1]
df_raw["Probability"] = prob
df_raw["Prediction"]  = np.where(prob >= 0.5, "Anställningsbar", "EJ anställningsbar")
df_raw["Percentile"]  = pd.Series(prob).rank(pct=True) * 100
df_raw = df_raw.reset_index(drop=True)
df_raw["Kandidatindex"] = df_raw.index + 1



display_cols = ["Kandidatindex", "Probability", "Prediction"]
df_view = (
    df_raw[display_cols]
    .sort_values("Probability", ascending=False)
    .reset_index(drop=True)
)

def color_prediction(val):
    if val == "Anställningsbar":
        return "background-color: green; color:black"
    elif val == "EJ anställningsbar":
        return "background-color: red; color:white"
    return ""

st.subheader("Kandidater")
st.dataframe(
    df_view.style.applymap(color_prediction, subset=["Prediction"]),
    hide_index=True
)


# ──────────────────────────────────────
# 7. Ladda ned fil
csv_bytes = df_raw.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Ladda ned CSV med kandidater",
    data=csv_bytes,
    file_name="potential_candidates.csv",
    mime="text/csv",
)
