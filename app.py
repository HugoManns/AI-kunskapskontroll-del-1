import streamlit as st
import pandas as pd
import joblib

MODEL_FILE = "model_stl.pkl"


# 1.  Ladda modell

meta = joblib.load(MODEL_FILE)

X_FEATURES = meta["features"]
DUMMY_COLS = meta["dummy_cols"]
CAT_OPTIONS = meta["cat_opts"]


st.set_page_config(page_title="Rekryterings‑sannolikhet", layout="wide")
st.title("Rekryterare – Anställnings‑sannolikhet")



#model val på vänsterfliken

model_choice = st.radio(
    "Välj modell",
    options=["Random Forest", "XGBoost"],
    index=0
)


# Ladda rätt modell

model = meta["rf_model"] if model_choice == "Random Forest" else meta["xgb_model"]



# Kandidatens attribut

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Kategoriska attribut")
    age        = st.selectbox("Ålder",                     options=sorted(CAT_OPTIONS["Age"]))
    ed_level   = st.selectbox("Utbildningsnivå",          options=sorted(CAT_OPTIONS["EdLevel"]))
    country    = st.selectbox("Land (grupperat)",           options=sorted(CAT_OPTIONS["Country_grouped"]))
    main_branch= st.selectbox("Huvudgren",                 options=sorted(CAT_OPTIONS["MainBranch"]))
    employment = st.radio("Anställningstyp", [0, 1], index=1)

with col2:
    st.subheader("Numeriska attribut & färdigheter")
    years_code     = st.number_input("Totala års kodning", min_value=0, max_value=60, value=5)
    years_code_pro = st.number_input("År professionell kodning", min_value=0, max_value=60, value=3)
    salary_norm    = st.number_input(
        "Normaliserad lön (förhållande till landets median)",
        min_value=0.0, max_value=6.0, value=1.0, step=0.01
    )

    st.subheader("Färdigheter max 15")
    skills_selected = st.multiselect(
        "Har arbetat med",
        options=sorted(DUMMY_COLS),
        default=[],
        max_selections=15
    )
    computer_skills = len(skills_selected)
    st.caption(f"Valda färdigheter: {computer_skills}")



# Bygg kandidatrad
def build_candidate_row():
    row = {
        "Age": age,
        "EdLevel": ed_level,
        "Country_grouped": country,
        "MainBranch": main_branch,
        "Employment": employment,
        "YearsCode": years_code,
        "YearsCodePro": years_code_pro,
        "PreviousSalary_norm": salary_norm,
        "ComputerSkills": computer_skills
    }



# HaveWorkedWith‑dummies
    for col in DUMMY_COLS:
        row[col] = 1 if col in skills_selected else 0

    raw_df = pd.DataFrame([row])
    encoded = pd.get_dummies(raw_df, drop_first=True)
    candidate = encoded.reindex(columns=X_FEATURES, fill_value=0)
    return candidate



# Prediktion

if st.button("Beräkna sannolikhet"):
    candidate_df = build_candidate_row()
    prob  = model.predict_proba(candidate_df)[0, 1]
    pred  = model.predict(candidate_df)[0]

    st.subheader(f"Vald modell: **{model_choice}**")
    st.success(f"**Sannolikhet för anställning:** {prob:.2%}")
    st.caption(
        f"Pre‑bedömning: **{'Anställd' if pred == 1 else 'Ej anställd'}**"
    )
    
    # st.json(candidate_df.iloc[0].to_dict())


