#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score
)

TRAIN_CSV  = "stackoverflow_full.csv"
MODEL_FILE = "model_stl.pkl"


def main():
    df = pd.read_csv(TRAIN_CSV)

    # -------------------------------------------------------------
    # 1. Data‑rengöring
    # -------------------------------------------------------------
    df["Country_raw"] = df["Country"]
    counts = df["Country_raw"].value_counts()
    rare   = counts[counts < 2].index
    df["Country_grouped"] = df["Country_raw"].replace(rare, "Other")
    valid  = counts[counts >= 2].index
    df = df[df["Country_raw"].isin(valid)]

    cols_drop = ["Gender", "MentalHealth", "Accessibility", "Unnamed: 0"]
    df = df.drop(columns=[c for c in cols_drop if c in df.columns])

    df = df[df["YearsCodePro"] <= df["YearsCode"]]
    df = df.drop(df[(df["Age"] == "<35") & (df["YearsCode"] > 35)].index)

    median_salary_by_country = df.groupby("Country")["PreviousSalary"].median()
    df["PreviousSalary_norm"] = df.apply(
        lambda r: r["PreviousSalary"] / median_salary_by_country[r["Country"]], axis=1
    )
    df = df.drop(columns="PreviousSalary")

    # -------------------------------------------------------------
    # 2. Dummy‑kolumner för HaveWorkedWith
    # -------------------------------------------------------------
    haveworked_dummies = df["HaveWorkedWith"].str.get_dummies(sep=";")
    df = pd.concat([df.drop(columns="HaveWorkedWith"), haveworked_dummies], axis=1)

    computer_skills = haveworked_dummies.columns.tolist()
    df["ComputerSkills"] = haveworked_dummies.sum(axis=1)

    # -------------------------------------------------------------
    # 3. Features / target  (före one‑hot)
    # -------------------------------------------------------------
    X = df.drop(columns=["Employed", "Country_raw"])
    y = df["Employed"]

    # Bevar unika värden för ursprungliga kategoriska kolumner
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    cat_opts = {c: df[c].unique().tolist() for c in cat_cols}

    # One‑hot encode alla kvarstående objekt‑kolumner
    X = pd.get_dummies(X, drop_first=True)

    # -------------------------------------------------------------
    # 4. Split & träna
    # -------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=df["Country_raw"]
    )

    # ----- RandomForest -----
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # ----- XGBoost -----
    xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)

    
    ''' SVM modell med GridSearchCV -------------------------------------------------
    Vi skapar en pipeline som först skalar datan med StandardScaler och sedan tränar
    en SVC-modell. Vi använder GridSearchCV för att hitta de bästa hyperparametrarna
    för SVC genom att testa olika värden för C och gamma.
    ------------------------------------------------------------------------------
    
    # ----- SVM -----
    # 1) Skapa en pipeline så att scaling sker inuti modellen
    scaler = StandardScaler()
    svc = SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced"
    )

    pipe = make_pipeline(scaler, svc)

    param_grid = {
        "svc__C": [0.1, 1, 10],
        "svc__gamma": ["scale", "auto", 0.01, 0.1, 1]
    }

    grid = GridSearchCV(
        pipe,
        param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)
    

    print("\nBest SVM parameters:", grid.best_params_)
    print("Best cross‑val AUC (SVM):", grid.best_score_)
    '''

    # -------------------------------------------------------------
    # 5. Utvärdera och skriv ut confusion matrix
    # -------------------------------------------------------------
    for name, model in (
            ("Random Forest", rf),
            ("XGBoost", xgb),
            #("SVM", grid)     # <‑  här läggs SVM in
    ):
        print(f"\n=== {name} ===")
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        print(f"Tränings‑precision:  {model.score(X_train, y_train):.4f}")
        print(f"Test‑precision:     {model.score(X_test, y_test):.4f}")

        print("\n Confusion‑matrix (Test)")
        print(confusion_matrix(y_test, y_pred))

        print("\n Klassifikations‑rapport (Test)")
        print(classification_report(y_test, y_pred, zero_division=0))

        print(f"\n ROC‑AUC (Test): {roc_auc_score(y_test, y_proba):.4f}")

    # -------------------------------------------------------------
    # 6. Spara modeller + metadata
    # -------------------------------------------------------------
    joblib.dump(
        {
            "rf_model": rf,
            "xgb_model": xgb,
            #"svm_model": grid,              # <‑  lagra det bästa SVM‑pipelinen
            "features": X.columns,
            "dummy_cols": haveworked_dummies.columns.tolist(),
            "cat_opts": cat_opts,
        },
        MODEL_FILE
    )
    print(f"\n✔️  Modellpaketet sparat som {MODEL_FILE}")


if __name__ == "__main__":
    main()
