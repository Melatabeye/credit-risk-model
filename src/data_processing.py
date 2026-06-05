import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def create_proxy_target(df):
    """
    Create proxy high-risk target.
    """

    df = df.copy()

    df["is_high_risk"] = (
        (df["credit_amount"] > df["credit_amount"].median()) &
        (df["duration"] > df["duration"].median())
    ).astype(int)

    return df


def build_preprocessor(df):
    """
    Build preprocessing pipeline.
    """

    numeric_features = df.select_dtypes(
        include=["int64", "float64", "uint8"]
    ).columns.tolist()

    categorical_features = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor
