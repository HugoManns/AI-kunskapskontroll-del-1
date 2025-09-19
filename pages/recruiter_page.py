# --- imports -------------------------------------------------------------
import os
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- metadata -------------------------------------------------------------
MODEL_FILE = "model_stl.pkl"

# Use st.cache_resource to load the model once
@st.cache_resource
def load_meta():
    return joblib.load(MODEL_FILE)

META = load_meta()

# -----------------------------------------------
# 1. Modell (sparad i session_state så den finns när vi räknar)
if "model" not in st.session_state:
    st.sidebar.title("Modell")
    model_choice = st.sidebar.radio(
        "Välj modell för att beräkna sannolikheten",
        options=["Random Forest", "XGBoost", "Logistic Regression"],
        index=0,
    )
    if model_choice == "Random Forest":
        st.session_state.model = META["rf_model"]
    elif model_choice == "XGBoost":
        st.session_state.model = META["xgb_model"]
    else:
        st.session_state.model = META["lr_model"]

model = st.session_state.model

# -----------------------------------------------
# 2. Läs CSV – Korrigerat sätt att läsa in data
csv_path = "candidates.csv"
if not os.path.exists(csv_path):
    st.info(f"Fil `{csv_path}` finns inte. Lägg in kandidater via *Lägg till kandidat*‑sidan.")
    st.stop()

# Skapa en lista med alla förväntade kolumnnamn från modellen + de manuella
ALL_COLUMNS = list(META["features"]) + ["Email"]

# Läs filen med pandas, nu med rätt antal kolumner
df_all = pd.read_csv(
    csv_path,
    header=None,
    names=ALL_COLUMNS, # Ange de exakta kolumnnamnen vid inläsning
    engine="python",
    on_bad_lines="skip",
    na_values=[0] # Behandla nollor som saknade värden om det är så de sparas
)

# -----------------------------------------------
# 3. Hantera "Employed"-kolumnen
# Denna kolumn är inte i din `candidates.csv`, så vi skapar den
# och ger den ett standardvärde (t.ex. "-")
if "Employed" not in df_all.columns:
    df_all["Employed"] = "-"

# -----------------------------------------------
# 4. Beräkna sannolikheter
X = df_all[META["features"]].copy()
prob = model.predict_proba(X)[:, 1]
df_all["Probability"] = prob
df_all["Prediction"] = np.where(prob >= 0.5, "Anställd", "Ej anställd")
df_all["Percentile"] = pd.Series(prob).rank(pct=True) * 100
df_all = df_all.reset_index(drop=True)
df_all["Kandidatindex"] = df_all.index + 1

# -----------------------------------------------
# 5. Visa tabell (inkludera Email om den finns)
display_cols = ["Kandidatindex", "Probability", "Prediction", "Percentile", "Email", "Employed"]

def color_pred(val):
    if val == "Anställd":
        return "background-color:#b7f5a7; color:black"
    if val == "Ej anställd":
        return "background-color:#f5a7a7; color:white"
    return ""

st.title("Inkomna kandidater")
st.write(f"Antal kandidater: **{len(df_all)}**")
if len(df_all) == 0:
    st.info("Ingen kandidat att visa.")
    st.stop()

st.subheader("Kandidater")
df_view = (
    df_all[display_cols]
    .sort_values("Probability", ascending=False)
    .reset_index(drop=True)
)

st.dataframe(
    df_view.style.applymap(color_pred, subset=["Prediction"]),
    hide_index=True
)

# -----------------------------------------------
# 6. Ladda ned fil
csv_bytes = df_all.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Ladda ned CSV med kandidater",
    data=csv_bytes,
    file_name="potential_candidates.csv",
    mime="text/csv",
)