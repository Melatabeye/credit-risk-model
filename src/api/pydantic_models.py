from pydantic import BaseModel
from typing import List


class CustomerInput(BaseModel):
    features: List[float]


class PredictionOutput(BaseModel):
    risk_probability: float
    prediction: int
    