# --- 1/2  – import, meta, state, layout  --------------------------------------------------------
import os
import streamlit as st
import pandas as pd
import joblib

MODEL_FILE = "model_stl.pkl"
META = joblib.load(MODEL_FILE)

X_FEATURES   = META["features"] 
DUMMY_COLS   = META["dummy_cols"]   
CAT_OPTIONS  = META["cat_opts"]        

# state‑initialisering
if "step" not in st.session_state:
    st.session_state.step = 0
if "answers" not in st.session_state:
    st.session_state.answers = {}

st.set_page_config(page_title="Nytt kandidatinlägg", layout="wide")
st.title("Lägg till en ny kandidat – steg för steg")
# ----------------------------------------------------------------------

def next_step():  st.session_state.step += 1
def prev_step():  st.session_state.step -= 1

def finish_and_save():
    "Dataframe & sparar till candidates.csv även email."
    # --- samla in svar --------------------------------------------------
    age           = st.session_state.get("age", sorted(CAT_OPTIONS["Age"])[0])
    ed_level      = st.session_state.get("ed_level", sorted(CAT_OPTIONS["EdLevel"])[0])
    country       = st.session_state.get("country", sorted(CAT_OPTIONS["Country_grouped"])[0])
    main_branch   = st.session_state.get("main_branch", sorted(CAT_OPTIONS["MainBranch"])[0])
    employment    = st.session_state.get("employment", 1)
    years_code    = st.session_state.get("years_code", 5)
    years_code_pro= st.session_state.get("years_code_pro", 3)
    salary_norm   = st.session_state.get("salary_norm", 1.0)
    skills_selected = st.session_state.get("skills_selected", [])
    computer_skills  = len(skills_selected)

    
    user_email = st.session_state.get("email")

   
    row = {
        "Email": user_email,
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

    for col in DUMMY_COLS:   
        row[col] = 1 if col in skills_selected else 0

    raw_df   = pd.DataFrame([row])
    encoded  = pd.get_dummies(raw_df, drop_first=True)
    candidate_df = encoded.reindex(columns=X_FEATURES, fill_value=0)

   
    candidate_df["Email"] = user_email

    # skriv till csv
    csv_path = "candidates.csv"
    if not os.path.isfile(csv_path):
        candidate_df.to_csv(csv_path, index=False)        
    else:
        candidate_df.to_csv(csv_path, mode="a", header=False, index=False)

    st.success("Kandidaten sparades i `candidates.csv`")
    st.subheader("Sparad rad")
    st.dataframe(candidate_df)


step = st.session_state.step

if step == 0:
    st.subheader("Ålder")

    age_num = st.number_input(
        "Ålder",
        min_value=0,
        max_value=120,
        value=25,
        key="age_num"
    )

    def age_to_cat(age: int) -> str:
        cats = sorted(CAT_OPTIONS["Age"])
        for cat in cats:
            if str(age) in cat:
                return cat
        return "Under 35" if age < 35 else "Över 35"

    st.session_state["age"] = age_to_cat(age_num)

    col_next, _ = st.columns([1, 1])
    with col_next:
        st.button("Nästa", on_click=next_step)


elif step == 1:
    st.subheader("Emailadress")
    user_email = st.text_input(
        "Ange kandidatens email",
        value="",
        key="email"
    )
    if st.session_state.get("email") and "@" not in st.session_state["email"]:
        st.warning("Ogiltig emailadress.")

    col_prev, col_next = st.columns([1, 1])
    with col_prev:
        st.button("← Back", on_click=prev_step)
    with col_next:
        st.button("Nästa", on_click=next_step)


elif step == 2:
    st.subheader("Utbildningsnivå")
    st.selectbox(
        "Utbildningsnivå",
        sorted(CAT_OPTIONS["EdLevel"]),
        key="ed_level"
    )
    col_prev, col_next = st.columns([1, 1])
    with col_prev:
        st.button("← Back", on_click=prev_step)
    with col_next:
        st.button("Nästa", on_click=next_step)


elif step == 3:
    st.subheader("Land (grupperat)")
    st.selectbox(
        "Land (grupperat)",
        sorted(CAT_OPTIONS["Country_grouped"]),
        key="country"
    )
    col_prev, col_next = st.columns([1, 1])
    with col_prev:
        st.button("← Back", on_click=prev_step)
    with col_next:
        st.button("Nästa", on_click=next_step)


elif step == 4:
    st.subheader("Huvudgren")
    st.selectbox(
        "Huvudgren",
        sorted(CAT_OPTIONS["MainBranch"]),
        key="main_branch"
    )
    col_prev, col_next = st.columns([1, 1])
    with col_prev:
        st.button("← Back", on_click=prev_step)
    with col_next:
        st.button("Nästa", on_click=next_step)


elif step == 5:
    employment_labels = [
        "Ja, Jag har en anställning",
        "Nej, Jag är arbetslös",
    ]
    selected_label = st.radio(
        "Anställningstyp",
        options=employment_labels,
        index=1,
        key="employment_label"
    )
    employment_num = 0 if selected_label == employment_labels[0] else 1
    st.session_state["employment"] = employment_num

    col_prev, col_next = st.columns([1, 1])
    with col_prev:
        st.button("← Back", on_click=prev_step)
    with col_next:
        st.button("Nästa", on_click=next_step)


elif step == 6:
    st.subheader("Numeriska attribut")
    st.number_input(
        "Totala års kodning",
        min_value=0,
        max_value=60,
        value=5,
        key="years_code"
    )
    st.number_input(
        "År professionell kodning",
        min_value=0,
        max_value=60,
        value=3,
        key="years_code_pro"
    )
    if st.session_state.get("years_code_pro", 0) > st.session_state.get("years_code", 0):
        st.warning("År professionell kodning kan inte vara större än totala års kodning.")

    st.number_input(
        "Normaliserad lön",
        min_value=0.0,
        max_value=6.0,
        value=1.0,
        step=0.01,
        key="salary_norm"
    )
    col_prev, col_next = st.columns([1, 1])
    with col_prev:
        st.button("← Back", on_click=prev_step)
    with col_next:
        st.button("Nästa", on_click=next_step)


elif step == 7:
    st.subheader("Färdigheter (max 15)")
    skills_selected = st.multiselect(
        "Har arbetat med",
        options=sorted(DUMMY_COLS),
        default=[],
        max_selections=15,
        key="skills_selected"
    )
    st.caption(f"Valda färdigheter: {len(skills_selected)}")

    col_prev, col_save = st.columns([1, 1])
    with col_prev:
        st.button("← Back", on_click=prev_step)
    with col_save:
        st.button(
            "Spara kandidat till CSV",
            on_click=finish_and_save,
            type="primary"
        )

if st.session_state.step > 0:
    st.markdown("---")
    if st.button("Lägg till en annan kandidat", key="new_candidate"):
        st.session_state.step = 0
