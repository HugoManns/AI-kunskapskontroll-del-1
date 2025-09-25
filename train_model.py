import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    accuracy_score
)

TRAIN_CSV = "stackoverflow_full.csv"
MODEL_FILE = "model_stl.pkl"


def main():
    df = pd.read_csv(TRAIN_CSV)

    # Cleaning
    df["Country_raw"] = df["Country"]
    counts = df["Country_raw"].value_counts()
    rare = counts[counts < 2].index
    df["Country_grouped"] = df["Country_raw"].replace(rare, "Other")
    valid = counts[counts >= 2].index
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

    haveworked_dummies = df["HaveWorkedWith"].str.get_dummies(sep=";")
    df = pd.concat([df.drop(columns="HaveWorkedWith"), haveworked_dummies], axis=1)

    computer_skills = haveworked_dummies.columns.tolist()
    df["ComputerSkills"] = haveworked_dummies.sum(axis=1)

    # Features / target före one-hot
    X = df.drop(columns=["Employed", "Country_raw"])
    y = df["Employed"]

    strat_col = df["Country_grouped"]

    cat_cols = X.select_dtypes(include="object").columns.tolist()
    cat_options = {c: df[c].unique().tolist() for c in cat_cols}

    # One-hot encode alla kvarstående objekt-kolumner
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=strat_col
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        random_state=42,
        stratify=strat_col.loc[X_train.index]
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=None,
        subsample=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)


    auc_scores = {}

    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    import seaborn as sns

    for name, model in (("Random Forest", rf), ("XGBoost", xgb)):
        print(f"\n{name}")

        y_val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        print(f"Validation Accuracy: {val_acc:.4f}")

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        test_acc = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {test_acc:.4f}")

        print("\n Confusion-matrix (Test)")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

        print("\n Klassifikations-rapport (Test)")
        print(classification_report(y_test, y_pred, zero_division=0))
        auc_score = roc_auc_score(y_test, y_proba)
        auc_scores[name] = auc_score
        print(f"\n ROC-AUC (Test): {auc_score:.4f}")


        plt.figure(figsize=(5, 4))
        sns.heatmap(cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=["Not Hired", "Hired"],
                    yticklabels=["Not Hired", "Hired"])
        plt.title(f"Confusion Matrix – {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(f"confusion_matrix_{name.replace(' ', '_')}.png", dpi=300)
        plt.close()


    plt.figure(figsize=(8, 6))
    for name, model in (("Random Forest", rf), ("XGBoost", xgb)):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.title("ROC Curve – Random Forest vs XGBoost")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("roc_curve_models.png", dpi=300)
    plt.show()

    # Välj bästa modell
    best_name = max(auc_scores, key=auc_scores.get)
    best_model = rf if best_name == "Random Forest" else xgb
    print(f"\nBästa modell: {best_name} (ROC-AUC: {auc_scores[best_name]:.4f})")

    # Träna om bästa modellen på hela datan för produktion
    X_full = pd.get_dummies(df.drop(columns=["Employed", "Country_raw"]), drop_first=True)
    y_full = df["Employed"]
    best_model.fit(X_full, y_full)


    joblib.dump(
        {
            "best_model": best_model,
            "features": X.columns,
            "dummy_cols": haveworked_dummies.columns.tolist(),
            "cat_opts": cat_options,
        },
        MODEL_FILE
    )
    print(f"\n Modellpaketet sparat som {MODEL_FILE}")


if __name__ == "__main__":
    main()
