# candidate_page.py
import os
import streamlit as st
import pandas as pd
import joblib

MODEL_FILE = "model_stl.pkl"
meta = joblib.load(MODEL_FILE)

# ----- Meta‑variabler
X_FEATURES    = meta["features"]
DUMMY_COLS    = meta["dummy_cols"]
CAT_OPTIONS   = meta["cat_opts"]

# ----- Streamlit‑layout
st.set_page_config(page_title="Nytt kandidatinlägg", layout="wide")
st.title("Lägg till en ny kandidat")

# ---- 1. Formulär
with st.form(key="candidate_form"):
    col1, col2 = st.columns([1, 2])

    # Kategoriska attribut
    with col1:
        st.subheader("Kategoriska attribut")
        age          = st.selectbox("Ålder",      sorted(CAT_OPTIONS["Age"]))
        ed_level     = st.selectbox("Utbildningsnivå", sorted(CAT_OPTIONS["EdLevel"]))
        country      = st.selectbox("Land (grupperat)", sorted(CAT_OPTIONS["Country_grouped"]))
        main_branch  = st.selectbox("Huvudgren",         sorted(CAT_OPTIONS["MainBranch"]))
        employment   = st.radio("Anställningstyp", [0, 1], index=1)

    # Numeriska attribut & färdigheter
    with col2:
        st.subheader("Numeriska attribut & färdigheter")
        years_code        = st.number_input("Totala års kodning", min_value=0, max_value=60, value=5)
        years_code_pro    = st.number_input("År professionell kodning", min_value=0, max_value=60, value=3)
        salary_norm       = st.number_input("Normaliserad lön", min_value=0.0, max_value=6.0, value=1.0, step=0.01)

        st.subheader("Färdigheter (max 15)")
        skills_selected   = st.multiselect(
            "Har arbetat med",
            options=sorted(DUMMY_COLS),
            default=[],
            max_selections=15
        )
        computer_skills   = len(skills_selected)
        st.caption(f"Valda färdigheter: {computer_skills}")

    # –– 2. Klicka för att spara
    submitted = st.form_submit_button("Spara kandidat till CSV")

# ---- 3. Bygg DataFrame
def build_candidate_df():
    row = {
        "Age": age,
        "EdLevel": ed_level,
        "Country_grouped": country,
        "MainBranch": main_branch,
        "Employment": employment,
        "YearsCode": years_code,
        "YearsCodePro": years_code_pro,
        "PreviousSalary_norm": salary_norm,
        "ComputerSkills": computer_skills,
    }

    # Dummy‑kolumner för färdigheter
    for col in DUMMY_COLS:
        row[col] = 1 if col in skills_selected else 0

    raw_df   = pd.DataFrame([row])
    encoded  = pd.get_dummies(raw_df, drop_first=True)
    candidate = encoded.reindex(columns=X_FEATURES, fill_value=0)
    return candidate

# ---- 4. Spara till CSV
if submitted:
    cand_df = build_candidate_df()

    csv_path = "candidates.csv"

    # Skapa filen första gången
    if not os.path.isfile(csv_path):
        cand_df.to_csv(csv_path, index=False)
    else:   # Append utan header
        cand_df.to_csv(csv_path, mode="a", header=False, index=False)

    st.success("Kandidaten sparades i `candidates.csv`")
    st.dataframe(cand_df)   # Visa den sparade raden
