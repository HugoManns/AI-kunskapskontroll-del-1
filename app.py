#!/usr/bin/env python
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import joblib
import numpy as np

MODEL_FILE = "model_stl.pkl"

# ------------------------------------------------------------------
#  Ladda modell + metadata
# ------------------------------------------------------------------
meta = joblib.load(MODEL_FILE)
model          = meta["model"]
FEATURES       = meta["features"]          # kolumn‑ordning
DUMMY_COLS     = meta["dummy_cols"]        # HaveWorkedWith‑dummy‑kolumner
CAT_OPTIONS    = meta["cat_opts"]           # dict: kolumn -> lista av unika värden

st.set_page_config(page_title="Anställnings‑predictor", layout="wide")
st.title("🔍 Rekryterare – Anställnings‑sannolikhet")

# ------------------------------------------------------------------
#  Kandidatens attribut (input‑formulär)
# ------------------------------------------------------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Kategoriska attribut")
    age        = st.selectbox("Ålder",                     options=sorted(CAT_OPTIONS["Age"]))
    ed_level   = st.selectbox("Utbildningsnivå",           options=sorted(CAT_OPTIONS["EdLevel"]))
    country    = st.selectbox("Land (grupperat)",          options=sorted(CAT_OPTIONS["Country_grouped"]))
    main_branch= st.selectbox("Huvudgren",                 options=sorted(CAT_OPTIONS["MainBranch"]))
    # Employment är numerisk (0 eller 1) – vi använder en radiobox
    employment = st.radio("Anställningstyp (1 = aktiv", options=["No Employment", "Employed"], index=1)

with col2:
    st.subheader("Numeriska attribut & färdigheter")
    years_code     = st.number_input("Totala års kodning", min_value=0, max_value=60, value=5)
    years_code_pro = st.number_input("År professionell kodning", min_value=0, max_value=60, value=3)
    salary_norm    = st.number_input("Normaliserad lön (förhållande till landets median)",
                                     min_value=0.0, max_value=6.0, value=1.0, step=0.01)

    st.subheader("Färdigheter (separera med ';')")
    skills_selected = st.multiselect(
        "Har arbetat med",
        options=sorted(DUMMY_COLS),
        default=[]
    )

# ------------------------------------------------------------------
# Bygg en rad som matchar modellens kolumner
# ------------------------------------------------------------------
def build_candidate_row():
    # a) Bas‑data
    row = {
        "Age": age,
        "EdLevel": ed_level,
        "Country_grouped": country,
        "MainBranch": main_branch,
        "Employment": employment,
        "YearsCode": years_code,
        "YearsCodePro": years_code_pro,
        "PreviousSalary_norm": salary_norm,
    }

    # b) HaveWorkedWith‑dummies
    for col in DUMMY_COLS:
        row[col] = 1 if col in skills_selected else 0

    raw_df = pd.DataFrame([row])

    # c) One‑hot koda de återstående kategoriska kolumner
    encoded = pd.get_dummies(raw_df, drop_first=True)

    # d) Justera till modellens columns
    candidate = encoded.reindex(columns=FEATURES, fill_value=0)

    return candidate

# ------------------------------------------------------------------
#   Prediktion
# ------------------------------------------------------------------
if st.button("Beräkna sannolikhet"):
    candidate_df = build_candidate_row()
    prob  = model.predict_proba(candidate_df)[0, 1]
    pred  = model.predict(candidate_df)[0]

    st.success(
        f"**Sannolikhet för att bli anställd:** {prob:.2%}  "
        f"({'Anställd' if pred == 1 else 'Ej anställd'})"
    )
    st.json(candidate_df.iloc[0].to_dict())

# ------------------------------------------------------------------
#  (Valfritt) Visa topp 10 från en uppladdad CSV
# ------------------------------------------------------------------
st.markdown("---")
st.subheader("Topp 10 från en uppladdad CSV (valfri)")
uploaded = st.file_uploader("Ladda CSV med fler kandidater", type="csv")
if uploaded:
    batch = pd.read_csv(uploaded)

    # CSV:n måste ha exakt samma kolumner som modellen
    batch = batch[FEATURES]
    probs = model.predict_proba(batch)[:, 1]
    top10 = batch.assign(prob=probs).sort_values("prob", ascending=False).head(10)
    st.dataframe(top10)
