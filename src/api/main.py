from fastapi import FastAPI
import numpy as np

from src.api.pydantic_models import CustomerInput, PredictionOutput

app = FastAPI()

# Dummy model so API always runs
class DummyModel:
    def predict_proba(self, X):
        return np.array([[0.2, 0.8]])

model = DummyModel()


@app.get("/")
def home():
    return {"message": "Credit Risk API is running"}


@app.post("/predict", response_model=PredictionOutput)
def predict(data: CustomerInput):

    X = np.array(data.features).reshape(1, -1)

    proba = float(model.predict_proba(X)[0][1])
    pred = int(proba > 0.5)

    return PredictionOutput(
        risk_probability=proba,
        prediction=pred
    )
