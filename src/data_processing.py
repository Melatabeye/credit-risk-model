import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Constants
CREDIT_AMOUNT = "credit_amount"
DURATION = "duration"
TARGET_COLUMN = "is_high_risk"


def create_proxy_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a proxy target variable indicating high-risk borrowers.

    Args:
        df: Input DataFrame containing borrower information.

    Returns:
        A copy of the DataFrame with the proxy target column added.
    """

    df = df.copy()

    df[TARGET_COLUMN] = (
        (df[CREDIT_AMOUNT] > df[CREDIT_AMOUNT].median())
        & (df[DURATION] > df[DURATION].median())
    ).astype(int)

    return df


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """
    Build a preprocessing pipeline for numeric and categorical features.

    Args:
        df: Input DataFrame.

    Returns:
        A fitted ColumnTransformer preprocessing pipeline.
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
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor