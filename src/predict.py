import pandas as pd
import mlflow


DEFAULT_THRESHOLD = 0.5


class CreditRiskPredictor:
    """
    Load a trained credit risk model and generate predictions.
    """

    def __init__(self) -> None:
        """
        Load the latest Random Forest model from MLflow.
        """
        self.model = mlflow.sklearn.load_model(
            "models:/random_forest_model/latest"
        )

    def predict(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate credit risk predictions.

        Args:
            input_data: DataFrame containing customer features.

        Returns:
            DataFrame containing risk probability and prediction.
        """

        probabilities = self.model.predict_proba(input_data)[:, 1]

        predictions = (
            probabilities > DEFAULT_THRESHOLD
        ).astype(int)

        return pd.DataFrame(
            {
                "risk_probability": probabilities,
                "prediction": predictions,
            }
        )