# recruiter_page.py
import os
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import functools

MODEL_FILE = "model_stl.pkl"
META = joblib.load(MODEL_FILE)

# ──────────────────────────────────────
# 1. Modell‑val
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

# ──────────────────────────────────────
# 2. Läs kandidat‑fil
csv_path = "candidates.csv"
if not os.path.exists(csv_path):
    st.info(f"Fil `{csv_path}` finns inte. Lägg in kandidater via *Lägg till kandidat*‑sidan.")
    st.stop()

df_raw = pd.read_csv(csv_path, header=None)

# ──────────────────────────────────────
# 3. Sätt kolumnnamn
X_FEATURES = META["features"]           # 1‑D array eller pd.Index
X_FEATURES_LIST = list(X_FEATURES)      # <‑‑ kritiskt!

# Om kolumnantalet inte stämmer, lägg till 0‑kolumner
if len(df_raw.columns) != len(X_FEATURES):
    st.warning(
        f"CSV har {len(df_raw.columns)} kolumner, men vi förväntar oss {len(X_FEATURES)}. "
        "Fyller med 0 för saknade kolumner."
    )
    missing = set(X_FEATURES) - set(df_raw.columns)
    for col in missing:
        df_raw[col] = 0
    df_raw = df_raw.reindex(columns=X_FEATURES)

df_raw.columns = X_FEATURES  # Säkrar rätt namn

# ──────────────────────────────────────
# 4. Säkerställ “Employed”‑kolumn (om den saknas)
if "Employed" not in df_raw.columns:
    df_raw["Employed"] = "-"

# ──────────────────────────────────────
# 5. Kör modellen för varje kandidat
st.title("Kandidatlista – Sannolikheter & Pre‑bedömning")
st.write(f"Antal kandidater: **{len(df_raw)}**")

if len(df_raw) == 0:
    st.info("Ingen kandidat att visa.")
    st.stop()

X = df_raw[X_FEATURES].copy()
prob = model.predict_proba(X)[:, 1]
df_raw["Probability"] = prob
df_raw["Prediction"]   = np.where(prob >= 0.5, "Anställd", "Ej anställd")
df_raw["Percentile"]   = pd.Series(prob).rank(pct=True) * 100
df_raw = df_raw.reset_index(drop=True)
df_raw["#"] = df_raw.index + 1

# ──────────────────────────────────────
# 6. Visa resultatet – använd listan X_FEATURES_LIST
display_cols = [
    "#",
    "Probability",
    "Prediction",
    "Percentile",
    "Employed",
] + X_FEATURES_LIST

st.dataframe(
    df_raw[display_cols]
    .sort_values("Probability", ascending=False)
    .reset_index(drop=True)
)

# ──────────────────────────────────────
# 7. Laddar ned fil med resultat
csv_bytes = df_raw.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Ladda ned CSV med sannolikheter och percentiler",
    data=csv_bytes,
    file_name="candidates_with_results.csv",
    mime="text/csv",
)
