import numpy as np                
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from predict import CreditRiskPredictor


class MockModel:
    def predict_proba(self, data):
        return np.array([[0.2, 0.8] for _ in range(len(data))])


def test_predict_returns_dataframe():
    predictor = CreditRiskPredictor.__new__(CreditRiskPredictor)
    predictor.model = MockModel()

    data = pd.DataFrame([[1, 2, 3]])

    result = predictor.predict(data)

    assert isinstance(result, pd.DataFrame)


def test_prediction_output_columns_exist():
    predictor = CreditRiskPredictor.__new__(CreditRiskPredictor)
    predictor.model = MockModel()

    data = pd.DataFrame([[1, 2, 3]])

    result = predictor.predict(data)

    assert "risk_probability" in result.columns
    assert "prediction" in result.columns


def test_prediction_values_are_binary():
    predictor = CreditRiskPredictor.__new__(CreditRiskPredictor)
    predictor.model = MockModel()

    data = pd.DataFrame([[1, 2, 3]])

    result = predictor.predict(data)

    assert result["prediction"].iloc[0] == 1