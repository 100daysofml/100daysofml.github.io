
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
import json
import uvicorn

app = FastAPI(title="Iris Model API", version="1.0")

# Load model and metadata at startup
model = joblib.load('iris_model.pkl')
with open('model_metadata.json', 'r') as f:
    metadata = json.load(f)

# Define request/response models
class PredictionInput(BaseModel):
    features: List[float]

    class Config:
        json_schema_extra = {
            "example": {
                "features": [5.1, 3.5, 1.4, 0.2]
            }
        }

class BatchPredictionInput(BaseModel):
    features: List[List[float]]

    class Config:
        json_schema_extra = {
            "example": {
                "features": [
                    [5.1, 3.5, 1.4, 0.2],
                    [6.2, 3.4, 5.4, 2.3]
                ]
            }
        }

class PredictionOutput(BaseModel):
    prediction: str
    prediction_id: int
    probabilities: dict

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Iris Model API",
        "version": "1.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": metadata["model_type"]}

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """Single prediction endpoint with automatic validation"""
    try:
        features = np.array(input_data.features).reshape(1, -1)

        if features.shape[1] != 4:
            raise HTTPException(status_code=400, detail="Expected 4 features")

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]

        return PredictionOutput(
            prediction=metadata["target_names"][prediction],
            prediction_id=int(prediction),
            probabilities={
                name: float(prob)
                for name, prob in zip(metadata["target_names"], probability)
            }
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/batch_predict")
async def batch_predict(input_data: BatchPredictionInput):
    """Batch prediction endpoint"""
    try:
        features = np.array(input_data.features)

        if features.shape[1] != 4:
            raise HTTPException(status_code=400, detail="Expected 4 features per sample")

        predictions = model.predict(features)
        probabilities = model.predict_proba(features)

        return {
            "predictions": [
                {
                    "prediction": metadata["target_names"][pred],
                    "prediction_id": int(pred),
                    "probabilities": {
                        name: float(prob)
                        for name, prob in zip(metadata["target_names"], probs)
                    }
                }
                for pred, probs in zip(predictions, probabilities)
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model_info")
async def model_info():
    """Get model metadata"""
    return metadata

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
