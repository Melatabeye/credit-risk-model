import numpy as np
import mlflow
import pandas as pd


class CreditRiskPredictor:
    def __init__(self):
        # Load latest Random Forest model from MLflow
        self.model = mlflow.sklearn.load_model(
            "models:/random_forest_model/latest"
        )

    def predict(self, input_data):
        """
        input_data: pandas DataFrame
        """

        proba = self.model.predict_proba(input_data)[:, 1]
        preds = (proba > 0.5).astype(int)

        return pd.DataFrame({
            "risk_probability": proba,
            "prediction": preds
        })


if __name__ == "__main__":
    # simple test run
    sample = pd.DataFrame([[0]*10])  # placeholder input
    predictor = CreditRiskPredictor()
    print(predictor.predict(sample))
