import io
import sys
import joblib
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Optional

# Import custom model classes BEFORE loading models
import models
from models import RangeRandomizerRegressor

# Register the class in __main__ module so joblib can find it
sys.modules['__main__'].RangeRandomizerRegressor = RangeRandomizerRegressor

from config import (
    MODEL_CONFIGS, 
    SAVGOL_WINDOW, 
    SAVGOL_POLYORDER, 
    SAVGOL_DERIV,
    API_TITLE,
    API_VERSION,
    API_DESCRIPTION
)

app = FastAPI(
    title=API_TITLE, 
    version=API_VERSION,
    description=API_DESCRIPTION
)

# Load models
models = {}
kept_indices = {}

for model_key, config in MODEL_CONFIGS.items():
    try:
        models[model_key] = joblib.load(config["path"])
        # Kiểm tra xem có file kept_indices riêng không
        indices_path = config.get("kept_indices_path")
        if indices_path:
            import os
            if os.path.exists(indices_path):
                kept_indices[model_key] = pd.read_csv(indices_path).values.flatten()
            else:
                kept_indices[model_key] = None
        else:
            kept_indices[model_key] = None
    except Exception as e:
        print(f"Warning: Không thể load model {model_key}: {e}")


def load_csv(content: bytes) -> pd.DataFrame:
    """Load CSV from bytes"""
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        for sep in [";", "\t"]:
            try:
                df = pd.read_csv(io.BytesIO(content), sep=sep)
                break
            except Exception:
                df = None
        if df is None:
            raise HTTPException(status_code=400, detail="Không đọc được CSV. Hãy kiểm tra định dạng.")
    if df.empty:
        raise HTTPException(status_code=400, detail="File rỗng.")
    return df


def preprocess_nir(X_raw: np.ndarray, kept_indices: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Preprocessing NIR data:
    1. Savitzky-Golay filter (2nd derivative)
    2. Feature selection (nếu có kept_indices)
    """
    # Apply Savitzky-Golay filter
    X_filtered = savgol_filter(X_raw, SAVGOL_WINDOW, polyorder=SAVGOL_POLYORDER, deriv=SAVGOL_DERIV, axis=1)
    
    # Feature selection nếu có
    if kept_indices is not None:
        X_filtered = X_filtered[:, kept_indices]
    
    return X_filtered


def predict_single_model(X_raw: np.ndarray, model_key: str) -> Dict:
    """Predict using a single model"""
    if model_key not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model_key}' không tồn tại.")
    
    model = models[model_key]
    indices = kept_indices.get(model_key)
    config = MODEL_CONFIGS[model_key]
    
    # Preprocess
    X_processed = preprocess_nir(X_raw, indices)
    
    # Predict
    predictions = model.predict(X_processed).ravel().tolist()
    
    return {
        "model": model_key,
        "name": config["name"],
        "unit": config.get("unit", ""),
        "predictions": predictions,
        "n_features_used": X_processed.shape[1]
    }


@app.get("/")
async def root():
    """API information"""
    return {
        "title": API_TITLE,
        "version": API_VERSION,
        "models": [
            {
                "key": k, 
                "name": v["name"],
                "unit": v.get("unit", ""),
                "description": v.get("description", ""),
                "status": "loaded" if k in models else "error"
            } 
            for k, v in MODEL_CONFIGS.items()
        ],
        "endpoints": {
            "/": "API information",
            "/predict/all": "Predict all models at once",
            "/predict/{model_key}": "Predict single model",
            "/docs": "Swagger UI documentation",
            "/health": "Health check"
        },
        "preprocessing": {
            "savgol_window": SAVGOL_WINDOW,
            "savgol_polyorder": SAVGOL_POLYORDER,
            "savgol_deriv": SAVGOL_DERIV
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    loaded_models = [k for k in MODEL_CONFIGS.keys() if k in models]
    failed_models = [k for k in MODEL_CONFIGS.keys() if k not in models]
    
    return {
        "status": "healthy" if len(failed_models) == 0 else "degraded",
        "models_loaded": len(loaded_models),
        "models_total": len(MODEL_CONFIGS),
        "loaded": loaded_models,
        "failed": failed_models
    }


@app.post("/predict/all")
async def predict_all(file: UploadFile = File(...)):
    """
    Predict all 5 models at once.
    Input: CSV file with NIR spectra (each row is a sample)
    Output: JSON with predictions from all models
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file .csv")
    
    content = await file.read()
    df = load_csv(content)
    
    # Convert to numpy array
    X_raw = df.to_numpy(dtype=float)
    
    if X_raw.shape[0] == 0:
        raise HTTPException(status_code=400, detail="Không có dữ liệu trong file")
    
    # Predict all models
    results = {}
    for model_key in MODEL_CONFIGS.keys():
        try:
            results[model_key] = predict_single_model(X_raw, model_key)
        except Exception as e:
            results[model_key] = {
                "model": model_key,
                "name": MODEL_CONFIGS[model_key]["name"],
                "error": str(e)
            }
    
    return JSONResponse(content={
        "n_samples": X_raw.shape[0],
        "n_wavelengths": X_raw.shape[1],
        "results": results
    })


@app.post("/predict/{model_key}")
async def predict_model(model_key: str, file: UploadFile = File(...)):
    """
    Predict using a single model.
    Input: CSV file with NIR spectra
    Output: JSON with predictions
    """
    if model_key not in MODEL_CONFIGS:
        available = ", ".join(MODEL_CONFIGS.keys())
        raise HTTPException(
            status_code=404, 
            detail=f"Model '{model_key}' không tồn tại. Các model khả dụng: {available}"
        )
    
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file .csv")
    
    content = await file.read()
    df = load_csv(content)
    
    # Convert to numpy array
    X_raw = df.to_numpy(dtype=float)
    
    if X_raw.shape[0] == 0:
        raise HTTPException(status_code=400, detail="Không có dữ liệu trong file")
    
    # Predict
    result = predict_single_model(X_raw, model_key)
    
    return JSONResponse(content={
        "n_samples": X_raw.shape[0],
        "n_wavelengths": X_raw.shape[1],
        "result": result
    })
