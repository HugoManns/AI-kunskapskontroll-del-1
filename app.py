import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
MODEL_FILE = "model_stl.pkl"


# 1.  Ladda modell + metadata

meta = joblib.load(MODEL_FILE)

X_FEATURES = meta["features"]
DUMMY_COLS = meta["dummy_cols"]
CAT_OPTIONS = meta["cat_opts"]



# 2.  Streamlit‑layout

st.set_page_config(page_title="Rekryterings‑sannolikhet", layout="wide")
st.title("Rekryterare – Anställnings‑sannolikhet")





# Radio‑val för model

model_choice = st.radio(
    "Välj modell",
    options=["Random Forest", "XGBoost", "Logistic Regression"],
    index=0
)


# Ladda rätt modell

if model_choice == "Random Forest":
    model = meta["rf_model"]
elif model_choice == "XGBoost":
    model = meta["xgb_model"]
else:
    model = meta["lr_model"]



# 3.  Kandidatens attribut

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

    st.subheader("Färdigheter (max 15)")
    skills_selected = st.multiselect(
        "Har arbetat med",
        options=sorted(DUMMY_COLS),
        default=[],
        max_selections=15
    )
    computer_skills = len(skills_selected)
    st.caption(f"Valda färdigheter: {computer_skills}")



# 4.  Bygg kandidat‑row
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



# 5.  Prediktion

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

if set(["X_val", "y_val"]).issubset(meta):
    X_val = meta["X_val"]
    y_val = meta["y_val"]

    col1, col2 = st.columns(2)

# ----- Plot 1: ROC-kurva -----
with col1:
    st.write("### ROC-kurva (Validering)")
    y_proba_val = model.predict_proba(X_val)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, y_proba_val)
    auc_val = roc_auc_score(y_val, y_proba_val)

    fig_roc, ax_roc = plt.subplots(figsize=(5, 4)) # Justera storleken här
    ax_roc.plot(fpr, tpr, color="darkorange", lw=2,
                label=f"ROC kurva (AUC={auc_val:.3f})")
    ax_roc.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    ax_roc.set_xlabel("Falsk positivt förhållande (FPR)")
    ax_roc.set_ylabel("Säker positivt förhållande (TPR)")
    ax_roc.set_title("Receiver Operating Characteristic")
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)
    plt.close(fig_roc) # Viktigt för att undvika att plotarna ritas om

# ----- Plot 2: Confusion-matrix -----
with col2:
    st.write("### Confusion-matrix (Validering)")
    y_pred_val = (y_proba_val >= 0.5).astype(int)
    cm = confusion_matrix(y_val, y_pred_val, labels=[0, 1])

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ej anställd", "Anställd"])
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4)) # Justera storleken här
    disp.plot(ax=ax_cm, cmap="viridis", xticks_rotation='vertical')
    st.pyplot(fig_cm)
    plt.close(fig_cm)
