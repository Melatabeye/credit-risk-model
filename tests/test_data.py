import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from sklearn.compose import ColumnTransformer

from data_processing import create_proxy_target, build_preprocessor
from sklearn.compose import ColumnTransformer

from data_processing import create_proxy_target, build_preprocessor

def test_create_proxy_target_adds_target_column():
    df = pd.DataFrame(
        {
            "credit_amount": [1000, 2000, 3000],
            "duration": [6, 12, 24],
        }
    )

    result = create_proxy_target(df)

    assert "is_high_risk" in result.columns


def test_create_proxy_target_returns_same_number_of_rows():
    df = pd.DataFrame(
        {
            "credit_amount": [1000, 2000, 3000],
            "duration": [6, 12, 24],
        }
    )

    result = create_proxy_target(df)

    assert len(result) == len(df)


def test_proxy_target_contains_only_binary_values():
    df = pd.DataFrame(
        {
            "credit_amount": [1000, 2000, 3000],
            "duration": [6, 12, 24],
        }
    )

    result = create_proxy_target(df)

    assert set(result["is_high_risk"].unique()).issubset({0, 1})


def test_build_preprocessor_returns_column_transformer():
    df = pd.DataFrame(
        {
            "credit_amount": [1000, 2000],
            "duration": [6, 12],
            "purpose": ["car", "business"],
        }
    )

    preprocessor = build_preprocessor(df)

    assert isinstance(preprocessor, ColumnTransformer)


def test_original_dataframe_not_modified():
    df = pd.DataFrame(
        {
            "credit_amount": [1000, 2000],
            "duration": [6, 12],
        }
    )

    create_proxy_target(df)

    assert "is_high_risk" not in df.columns