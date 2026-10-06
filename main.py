"""
Simple FastAPI app serving a scikit-learn model.

Run locally:
    uvicorn main:app --reload

Test it:
    curl -X POST http://127.0.0.1:8000/predict \
         -H "Content-Type: application/json" \
         -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

app = FastAPI(
    title="Getting Started with ML in Production API",
    description="A minimal prediction API built for the workshop.",
    version="1.0.0",
)

# Load the model once at startup
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None


class PredictionRequest(BaseModel):
    features: list[float] = Field(
        ..., min_length=4, max_length=4,
        description="Iris features: [sepal_length, sepal_width, petal_length, petal_width]"
    )


class PredictionResponse(BaseModel):
    prediction: int
    class_name: str


IRIS_CLASSES = ["setosa", "versicolor", "virginica"]


@app.get("/")
def root():
    return {"message": "Workshop ML API is running. See /docs for usage."}


@app.get("/health")
def health():
    """Basic health check endpoint — useful for deployment platforms & load balancers."""
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Did you run the training notebook?")

    features = np.array(request.features).reshape(1, -1)
    pred = int(model.predict(features)[0])

    return PredictionResponse(prediction=pred, class_name=IRIS_CLASSES[pred])

