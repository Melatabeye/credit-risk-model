import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn

from data_processing import create_proxy_target, build_preprocessor

# =========================
# Constants
# =========================
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_ITER = 1000
N_ESTIMATORS = 100
EXPERIMENT_NAME = "credit_risk"
TARGET_COLUMN = "is_high_risk"


def load_data() -> pd.DataFrame:
    """
    Load the German Credit dataset from OpenML.

    Returns:
        pd.DataFrame: The dataset as a pandas DataFrame.
    """
    data = fetch_openml(data_id=31, as_frame=True)
    return data.frame


def main() -> None:
    """
    Train and evaluate credit risk prediction models using MLflow.
    """

    # Step 1: Load data
    df = load_data()

    # Step 2: Create proxy target
    df = create_proxy_target(df)

    # Step 3: Split features and target
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Step 4: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Step 5: Preprocessing
    preprocessor = build_preprocessor(X_train)

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    # =========================
    # Logistic Regression
    # =========================
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="Logistic Regression"):

        log_model = LogisticRegression(
            max_iter=MAX_ITER,
            random_state=RANDOM_STATE,
        )

        log_model.fit(X_train, y_train)

        preds = log_model.predict(X_test)
        probs = log_model.predict_proba(X_test)[:, 1]

        mlflow.log_metric("accuracy", accuracy_score(y_test, preds))
        mlflow.log_metric("f1", f1_score(y_test, preds))
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, probs))

        mlflow.sklearn.log_model(log_model, "logistic_model")

    # =========================
    # Random Forest
    # =========================
    with mlflow.start_run(run_name="Random Forest"):

        rf_model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
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