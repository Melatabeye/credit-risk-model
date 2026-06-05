import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import mlflow
import mlflow.sklearn

from data_processing import create_proxy_target, build_preprocessor


def load_data():
    """
    Load dataset (update path if needed).
    """
    from sklearn.datasets import fetch_openml

    data = fetch_openml(data_id=31, as_frame=True)
    df = data.frame
    return df


def main():

    # Step 1: Load data
    df = load_data()

    # Step 2: Create proxy target
    df = create_proxy_target(df)

    # Step 3: Split features/target
    X = df.drop(columns=["is_high_risk"])
    y = df["is_high_risk"]

    # Step 4: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Step 5: Preprocessing
    preprocessor = build_preprocessor(X_train)

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    # =========================
    # Model 1: Logistic Regression
    # =========================
    mlflow.set_experiment("credit_risk")

    with mlflow.start_run(run_name="Logistic Regression"):

        log_model = LogisticRegression(max_iter=1000)
        log_model.fit(X_train, y_train)

        preds = log_model.predict(X_test)
        probs = log_model.predict_proba(X_test)[:, 1]

        mlflow.log_metric("accuracy", accuracy_score(y_test, preds))
        mlflow.log_metric("f1", f1_score(y_test, preds))
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, probs))

        mlflow.sklearn.log_model(log_model, "logistic_model")

    # =========================
    # Model 2: Random Forest
    # =========================

    with mlflow.start_run(run_name="Random Forest"):

        rf_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        rf_model.fit(X_train, y_train)

        preds = rf_model.predict(X_test)
        probs = rf_model.predict_proba(X_test)[:, 1]

        mlflow.log_metric("accuracy", accuracy_score(y_test, preds))
        mlflow.log_metric("f1", f1_score(y_test, preds))
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, probs))

        mlflow.sklearn.log_model(rf_model, "random_forest_model")


if __name__ == "__main__":
    main()
    