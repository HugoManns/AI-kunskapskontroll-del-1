#!/usr/bin/env python

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

TRAIN_CSV  = "stackoverflow_full.csv"
MODEL_FILE = "model_stl.pkl"

def main():
    df = pd.read_csv(TRAIN_CSV)

    # ---------------------------------------------------------------
    #  Data‑rengöring
    # ---------------------------------------------------------------
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

    # HaveWorkedWith → dummy‑kolumner
    haveworked_dummies = df["HaveWorkedWith"].str.get_dummies(sep=";")
    df = pd.concat([df.drop(columns="HaveWorkedWith"), haveworked_dummies], axis=1)

    # ---------------------------------------------------------------
    #  Features / target **(före one‑hot)**
    # ---------------------------------------------------------------
    X = df.drop(columns=["Employed", "Country_raw"])
    y = df["Employed"]

    # ---- Bevara unika värden för de ursprungliga kategoriska kolumnerna ----
    cat_cols       = X.select_dtypes(include="object").columns.tolist()
    cat_opts = {c: df[c].unique().tolist() for c in cat_cols}

    # ---- One‑hot encode alla kvarstående objekt‑kolumner ----
    X = pd.get_dummies(X, drop_first=True)

    # ---------------------------------------------------------------
    #  Split & träna
    # ---------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=df["Country_raw"]
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # ---------------------------------------------------------------
    #  Spara modell + metadata
    # ---------------------------------------------------------------
    joblib.dump(
        {
            "model":      rf,
            "features":   list(X.columns),            # kolumn‑ordning efter one‑hot
            "dummy_cols": haveworked_dummies.columns.tolist(),  # HaveWorkedWith‑dummy‑kolumner
            "cat_opts":   cat_opts,                                 # *viktigt* – original kategoriska värden
        },
        MODEL_FILE
    )
    print(f"✔️  Modell & metadata sparad som {MODEL_FILE}")

if __name__ == "__main__":
    main()
