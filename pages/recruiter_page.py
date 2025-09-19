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
# 2. Läs kandidat‑fil (endast en gång – python‑motor)
csv_path = "candidates.csv"
if not os.path.exists(csv_path):
    st.info(f"Fil `{csv_path}` finns inte. Lägg in kandidater via *Lägg till kandidat*‑sidan.")
    st.stop()

# Feature‑lista som vi kommer att använda
X_FEATURES = META["features"]
# Lägg till Email‑kolumnen i listan av förväntade kolumner
ALL_COLUMNS = list(X_FEATURES) + ["Email"]

# Läser med python‑motorn – hanterar ojämna kolumner
df_raw = pd.read_csv(
    csv_path,
    header=None,  # inga header‑rader i våra CSVs
    engine="python",  # python‑parser kan hantera varierande kolumner
    on_bad_lines="skip"  # hoppar över helt ogiltiga rader
)

# 0) Byt NaN till 0 så att alla rader får fullständig in‑feature‑matris
df_raw = df_raw.fillna(0)

# 1) Reindexa så att DataFrame har rätt antal kolumner för att matcha CSV:n
df_raw = df_raw.reindex(columns=range(len(ALL_COLUMNS)), fill_value=0)
df_raw.columns = ALL_COLUMNS  # Sätter alla kolumnnamn, inklusive 'Email'

# ──────────────────────────────────────
# 3. Beräkna sannolikheter
st.title("Kandidatlista – Sannolikheter & Pre‑bedömning")
if len(df_raw) == 0:
    st.info("Ingen kandidat att visa.")
    st.stop()

# Använd X_FEATURES för att predikera, men behåll Email‑kolumnen
X = df_raw[X_FEATURES].copy()
prob = model.predict_proba(X)[:, 1]
df_raw["Probability"] = prob
df_raw["Prediction"] = np.where(prob >= 0.5, "Anställd", "Ej anställd")
df_raw["Percentile"] = pd.Series(prob).rank(pct=True) * 100
df_raw = df_raw.reset_index(drop=True)
df_raw["#"] = df_raw.index + 1

# ──────────────────────────────────────
# 4. Visa resultatet – med fullständiga feature‑kolumner
# Lägg till Email i listan över kolumner som ska visas
display_cols = [
    "#",
    "Probability",
    "Prediction",
    "Percentile",
    "Email",
] + list(X_FEATURES)

st.dataframe(
    df_raw[display_cols]
    .sort_values("Probability", ascending=False)
    .reset_index(drop=True)
)

# ──────────────────────────────────────
# 5. Laddar ned fil med resultat
csv_bytes = df_raw.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Ladda ned CSV med sannolikheter och percentiler",
    data=csv_bytes,
    file_name="candidates_with_results.csv",
    mime="text/csv",
)