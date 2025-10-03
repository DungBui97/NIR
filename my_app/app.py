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
label_encoders = {}

for model_key, config in MODEL_CONFIGS.items():
    try:
        models[model_key] = joblib.load(config["path"])
        
        # Load label encoder cho classification models
        if config.get("type") == "classification" and "label_encoder_path" in config:
            label_encoder_path = config["label_encoder_path"]
            if os.path.exists(label_encoder_path):
                import pickle
                with open(label_encoder_path, 'rb') as f:
                    label_encoders[model_key] = pickle.load(f)
                print(f"Loaded label encoder for {model_key}")
            else:
                print(f"Warning: Label encoder not found for {model_key}: {label_encoder_path}")
        
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
    """
    Load CSV from bytes.
    Only supports Hadamard format:
    - Has header (21 rows metadata)
    - Row 22 is column names with Wavelength and Absorbance columns (vertical)
    - Sẽ chuyển đổi từ dạng dọc sang dạng ngang (1 mẫu x N wavelengths)
    """
    # Check if this is Hadamard format by looking for the header pattern
    content_str = content.decode('utf-8', errors='ignore')
    is_hadamard = 'Method:' in content_str or 'Wavelength (nm)' in content_str
    
    if not is_hadamard:
        raise HTTPException(
            status_code=400, 
            detail="File không đúng format Hadamard. Cần có header 'Method:' hoặc 'Wavelength (nm)'"
        )
    
    # Hadamard format: skip 21 header rows (metadata), row 22 is column names
    try:
        df = pd.read_csv(io.BytesIO(content), skiprows=21)
        
        # Check if we have the expected columns
        if 'Absorbance (AU)' in df.columns:
            # Extract Absorbance column and transpose to horizontal format
            absorbance_col = df['Absorbance (AU)'].values
            # Create a single-row dataframe (1 sample with N wavelengths)
            df_horizontal = pd.DataFrame([absorbance_col])
            return df_horizontal
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Không tìm thấy cột 'Absorbance (AU)' trong file Hadamard. Các cột khả dụng: {df.columns.tolist()}"
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi đọc file Hadamard: {str(e)}")


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


def classify_tcvn_standard(do_am: float, tro_tong: float, tinh_dau: float, 
                          piperin: float) -> str:
    """
    Phân loại tiêu theo tiêu chuẩn TCVN - Đạt hoặc Không đạt
    
    Tiêu chuẩn TCVN:
    - Tro tổng số: ≤ 7.0% w/w
    - Độ ẩm: ≤ 13%
    - Tinh dầu: ≥ 2.0 ml/100g
    - Piperin: ≥ 4%
    
    Lưu ý: Chất chiết EteL ≥ 6.0% (chưa có model)
    """
    # Kiểm tra tất cả các điều kiện phải thỏa mãn
    if (do_am <= 13.0 and 
        tro_tong <= 7.0 and 
        tinh_dau >= 2.0 and 
        piperin >= 4.0):
        return "Đạt TCVN"
    else:
        return "Không đạt TCVN"


def classify_esa_standard(do_am: float, tro_tong: float, tro_khong_tan: float, 
                          tinh_dau: float) -> str:
    """
    Phân loại tiêu theo tiêu chuẩn ESA - Đạt hoặc Không đạt
    
    Tiêu chuẩn ESA:
    - Tro tổng số: ≤ 7.0% w/w
    - Tro không tan trong axit: ≤ 1.5%
    - Độ ẩm: ≤ 12%
    - Tinh dầu: ≥ 2.0 ml/100g
    """
    # Kiểm tra tất cả các điều kiện phải thỏa mãn
    if (do_am <= 12.0 and 
        tro_tong <= 7.0 and 
        tro_khong_tan <= 1.5 and 
        tinh_dau >= 2.0):
        return "Đạt ESA"
    else:
        return "Không đạt ESA"


def classify_pepper_quality(tap_chat_la: float, hat_lep: float, 
                            hat_dau_dinh_vo: float, khoi_luong_theo_the_tich: float) -> dict:
    """
    Phân loại chất lượng hạt tiêu đen (NP hoặc SP) theo 4 chỉ tiêu vật lý
    
    Tiêu chuẩn phân loại:
    
    Loại 1:
    - Tạp chất lạ: ≤ 0.5%
    - Hạt lép: ≤ 6%
    - Hạt đầu đinh/vỡ: ≤ 2.0%
    - Khối lượng theo thể tích: ≥ 550 g/l
    
    Loại 2:
    - Tạp chất lạ: ≤ 1.0%
    - Hạt lép: ≤ 10%
    - Hạt đầu đinh/vỡ: ≤ 4.0%
    - Khối lượng theo thể tích: ≥ 500 g/l
    
    Loại 3:
    - Tạp chất lạ: ≤ 1.0%
    - Hạt lép: ≤ 18%
    - Hạt đầu đinh/vỡ: ≤ 4.0%
    - Khối lượng theo thể tích: ≥ 450 g/l
    """
    # Kiểm tra Loại 1 (tiêu chuẩn cao nhất)
    loai_1_checks = {
        "tap_chat_la": tap_chat_la <= 0.5,
        "hat_lep": hat_lep <= 6.0,
        "hat_dau_dinh_vo": hat_dau_dinh_vo <= 2.0,
        "khoi_luong_theo_the_tich": khoi_luong_theo_the_tich >= 550
    }
    
    # Kiểm tra Loại 2
    loai_2_checks = {
        "tap_chat_la": tap_chat_la <= 1.0,
        "hat_lep": hat_lep <= 10.0,
        "hat_dau_dinh_vo": hat_dau_dinh_vo <= 4.0,
        "khoi_luong_theo_the_tich": khoi_luong_theo_the_tich >= 500
    }
    
    # Kiểm tra Loại 3
    loai_3_checks = {
        "tap_chat_la": tap_chat_la <= 1.0,
        "hat_lep": hat_lep <= 18.0,
        "hat_dau_dinh_vo": hat_dau_dinh_vo <= 4.0,
        "khoi_luong_theo_the_tich": khoi_luong_theo_the_tich >= 450
    }
    
    # Xác định loại (ưu tiên từ cao xuống thấp)
    if all(loai_1_checks.values()):
        classification = "Loại 1"
        criteria_met = loai_1_checks
        thresholds = {
            "tap_chat_la": "≤ 0.5%",
            "hat_lep": "≤ 6%",
            "hat_dau_dinh_vo": "≤ 2.0%",
            "khoi_luong_theo_the_tich": "≥ 550 g/l"
        }
    elif all(loai_2_checks.values()):
        classification = "Loại 2"
        criteria_met = loai_2_checks
        thresholds = {
            "tap_chat_la": "≤ 1.0%",
            "hat_lep": "≤ 10%",
            "hat_dau_dinh_vo": "≤ 4.0%",
            "khoi_luong_theo_the_tich": "≥ 500 g/l"
        }
    elif all(loai_3_checks.values()):
        classification = "Loại 3"
        criteria_met = loai_3_checks
        thresholds = {
            "tap_chat_la": "≤ 1.0%",
            "hat_lep": "≤ 18%",
            "hat_dau_dinh_vo": "≤ 4.0%",
            "khoi_luong_theo_the_tich": "≥ 450 g/l"
        }
    else:
        classification = "Không đạt tiêu chuẩn"
        criteria_met = loai_3_checks
        thresholds = {
            "tap_chat_la": "≤ 1.0%",
            "hat_lep": "≤ 18%",
            "hat_dau_dinh_vo": "≤ 4.0%",
            "khoi_luong_theo_the_tich": "≥ 450 g/l"
        }
    
    return {
        "classification": classification,
        "criteria": {
            "tap_chat_la": {
                "value": tap_chat_la,
                "threshold": thresholds["tap_chat_la"],
                "passed": criteria_met["tap_chat_la"]
            },
            "hat_lep": {
                "value": hat_lep,
                "threshold": thresholds["hat_lep"],
                "passed": criteria_met["hat_lep"]
            },
            "hat_dau_dinh_vo": {
                "value": hat_dau_dinh_vo,
                "threshold": thresholds["hat_dau_dinh_vo"],
                "passed": criteria_met["hat_dau_dinh_vo"]
            },
            "khoi_luong_theo_the_tich": {
                "value": khoi_luong_theo_the_tich,
                "threshold": thresholds["khoi_luong_theo_the_tich"],
                "passed": criteria_met["khoi_luong_theo_the_tich"]
            }
        }
    }


def predict_single_model(X_raw: np.ndarray, model_key: str) -> Dict:
    """Predict using a single model"""
    if model_key not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model_key}' không tồn tại.")
    
    model = models[model_key]
    indices = kept_indices.get(model_key)
    config = MODEL_CONFIGS[model_key]
    
    # Check if this is a classification model
    is_classification = config.get("type") == "classification"
    
    if is_classification:
        # For classification, don't apply Savitzky-Golay filter
        # Just use raw data (or minimal preprocessing)
        X_processed = X_raw
        
        # Adjust features if needed
        expected_features = model.n_features_in_
        actual_features = X_processed.shape[1]
        
        if actual_features != expected_features:
            if actual_features > expected_features:
                X_processed = X_processed[:, :expected_features]
            else:
                # Pad with zeros if fewer features
                padding = np.zeros((X_processed.shape[0], expected_features - actual_features))
                X_processed = np.hstack([X_processed, padding])
        
        # Predict probabilities and classes
        predictions = model.predict(X_processed)
        probabilities = model.predict_proba(X_processed)
        
        # Get label encoder
        le = label_encoders.get(model_key)
        if le is None:
            raise HTTPException(status_code=500, detail=f"Label encoder không tồn tại cho model {model_key}")
        
        # Convert predictions to class names and get top 3 probabilities
        results = []
        for i in range(len(predictions)):
            # Get all class probabilities
            probs = probabilities[i]
            
            # Get top 3 indices sorted by probability (descending)
            top3_indices = np.argsort(probs)[-3:][::-1]
            
            # Create list of top 3 predictions
            top3_predictions = []
            for idx in top3_indices:
                region_name = le.inverse_transform([idx])[0]
                confidence = float(probs[idx])
                top3_predictions.append({
                    "region": region_name,
                    "confidence": confidence
                })
            
            results.append({
                "top_predictions": top3_predictions,
                "predicted_region": top3_predictions[0]["region"],  # Top 1 cho backward compatibility
                "confidence": top3_predictions[0]["confidence"]
            })
        
        return {
            "model": model_key,
            "name": config["name"],
            "unit": config.get("unit", ""),
            "predictions": results,
            "n_features_used": X_processed.shape[1],
            "type": "classification"
        }
    else:
        # For regression models, use existing preprocessing
        X_processed = preprocess_nir(X_raw, indices)
        
        # Predict
        predictions = model.predict(X_processed).ravel().tolist()
        
        result = {
            "model": model_key,
            "name": config["name"],
            "unit": config.get("unit", ""),
            "predictions": predictions,
            "n_features_used": X_processed.shape[1],
            "type": "regression"
        }
        
        return result


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
            "/predict/standards": "Predict TCVN & ESA standards only",
            "/predict/{model_key}": "Predict single model",
            "/classify/pepper_quality": "Classify pepper quality (Loại 1, 2, 3) based on 4 physical criteria",
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
    Predict all models at once and include TCVN classification.
    Input: CSV file with NIR spectra (each row is a sample)
    Output: JSON with predictions from all models + TCVN grade classification
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
    regression_values = {}
    
    for model_key in MODEL_CONFIGS.keys():
        try:
            result = predict_single_model(X_raw, model_key)
            results[model_key] = result
            
            # Lưu giá trị regression để tính TCVN
            if result.get("type") == "regression" and "predictions" in result:
                if model_key == "do_am":
                    regression_values["do_am"] = result["predictions"][0]
                elif model_key == "tro_tong":
                    regression_values["tro_tong"] = result["predictions"][0]
                elif model_key == "tro_khong_tan":
                    regression_values["tro_khong_tan"] = result["predictions"][0]
                elif model_key == "Tinh_dau":
                    regression_values["tinh_dau"] = result["predictions"][0]
                elif model_key == "piperin":
                    regression_values["piperin"] = result["predictions"][0]
                    
        except Exception as e:
            results[model_key] = {
                "model": model_key,
                "name": MODEL_CONFIGS[model_key]["name"],
                "error": str(e)
            }
    
    # Tính toán phân loại TCVN và ESA nếu có đủ dữ liệu regression
    tcvn_standard = None
    esa_standard = None
    
    if len(regression_values) == 5:  # Có đủ 5 chỉ số
        try:
            # Phân loại TCVN (không cần tro_khong_tan)
            tcvn_standard = classify_tcvn_standard(
                do_am=regression_values["do_am"],
                tro_tong=regression_values["tro_tong"],
                tinh_dau=regression_values["tinh_dau"],
                piperin=regression_values["piperin"]
            )
            
            # Phân loại ESA
            esa_standard = classify_esa_standard(
                do_am=regression_values["do_am"],
                tro_tong=regression_values["tro_tong"],
                tro_khong_tan=regression_values["tro_khong_tan"],
                tinh_dau=regression_values["tinh_dau"]
            )
        except Exception as e:
            tcvn_standard = f"Lỗi: {str(e)}"
            esa_standard = f"Lỗi: {str(e)}"
    
    response_data = {
        "n_samples": X_raw.shape[0],
        "n_wavelengths": X_raw.shape[1],
        "results": results
    }
    
    # Thêm phân loại TCVN nếu có
    if tcvn_standard:
        response_data["tcvn_standard"] = tcvn_standard
    
    # Thêm phân loại ESA nếu có
    if esa_standard:
        response_data["esa_standard"] = esa_standard
    
    return JSONResponse(content=response_data)


@app.post("/predict/standards")
async def predict_standards(file: UploadFile = File(...)):
    """
    Predict chemical standards (TCVN & ESA) only.
    Input: CSV file with NIR spectra
    Output: JSON with TCVN and ESA classification results
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file .csv")
    
    content = await file.read()
    df = load_csv(content)
    
    # Convert to numpy array
    X_raw = df.to_numpy(dtype=float)
    
    if X_raw.shape[0] == 0:
        raise HTTPException(status_code=400, detail="Không có dữ liệu trong file")
    
    # Predict only regression models needed for TCVN/ESA
    regression_results = {}
    regression_values = {}
    
    required_models = ["do_am", "tro_tong", "tro_khong_tan", "Tinh_dau", "piperin"]
    
    for model_key in required_models:
        if model_key not in MODEL_CONFIGS:
            continue
            
        try:
            result = predict_single_model(X_raw, model_key)
            regression_results[model_key] = result
            
            # Lưu giá trị regression
            if result.get("type") == "regression" and "predictions" in result:
                if model_key == "do_am":
                    regression_values["do_am"] = result["predictions"][0]
                elif model_key == "tro_tong":
                    regression_values["tro_tong"] = result["predictions"][0]
                elif model_key == "tro_khong_tan":
                    regression_values["tro_khong_tan"] = result["predictions"][0]
                elif model_key == "Tinh_dau":
                    regression_values["tinh_dau"] = result["predictions"][0]
                elif model_key == "piperin":
                    regression_values["piperin"] = result["predictions"][0]
                    
        except Exception as e:
            regression_results[model_key] = {
                "model": model_key,
                "name": MODEL_CONFIGS[model_key]["name"],
                "error": str(e)
            }
    
    # Kiểm tra xem có đủ 5 chỉ số không
    if len(regression_values) < 5:
        missing = set(required_models) - set(regression_values.keys())
        raise HTTPException(
            status_code=500, 
            detail=f"Không đủ dữ liệu để phân loại. Thiếu: {', '.join(missing)}"
        )
    
    # Tính toán phân loại TCVN
    try:
        tcvn_standard = classify_tcvn_standard(
            do_am=regression_values["do_am"],
            tro_tong=regression_values["tro_tong"],
            tinh_dau=regression_values["tinh_dau"],
            piperin=regression_values["piperin"]
        )
        
        # Chi tiết điều kiện TCVN
        tcvn_details = {
            "classification": tcvn_standard,
            "criteria": {
                "do_am": {
                    "value": regression_values["do_am"],
                    "threshold": "≤ 13%",
                    "passed": regression_values["do_am"] <= 13.0
                },
                "tro_tong": {
                    "value": regression_values["tro_tong"],
                    "threshold": "≤ 7.0%",
                    "passed": regression_values["tro_tong"] <= 7.0
                },
                "tinh_dau": {
                    "value": regression_values["tinh_dau"],
                    "threshold": "≥ 2.0 ml/100g",
                    "passed": regression_values["tinh_dau"] >= 2.0
                },
                "piperin": {
                    "value": regression_values["piperin"],
                    "threshold": "≥ 4.0%",
                    "passed": regression_values["piperin"] >= 4.0
                }
            }
        }
    except Exception as e:
        tcvn_details = {"error": str(e)}
    
    # Tính toán phân loại ESA
    try:
        esa_standard = classify_esa_standard(
            do_am=regression_values["do_am"],
            tro_tong=regression_values["tro_tong"],
            tro_khong_tan=regression_values["tro_khong_tan"],
            tinh_dau=regression_values["tinh_dau"]
        )
        
        # Chi tiết điều kiện ESA
        esa_details = {
            "classification": esa_standard,
            "criteria": {
                "do_am": {
                    "value": regression_values["do_am"],
                    "threshold": "≤ 12%",
                    "passed": regression_values["do_am"] <= 12.0
                },
                "tro_tong": {
                    "value": regression_values["tro_tong"],
                    "threshold": "≤ 7.0%",
                    "passed": regression_values["tro_tong"] <= 7.0
                },
                "tro_khong_tan": {
                    "value": regression_values["tro_khong_tan"],
                    "threshold": "≤ 1.5%",
                    "passed": regression_values["tro_khong_tan"] <= 1.5
                },
                "tinh_dau": {
                    "value": regression_values["tinh_dau"],
                    "threshold": "≥ 2.0 ml/100g",
                    "passed": regression_values["tinh_dau"] >= 2.0
                }
            }
        }
    except Exception as e:
        esa_details = {"error": str(e)}
    
    return JSONResponse(content={
        "n_samples": X_raw.shape[0],
        "n_wavelengths": X_raw.shape[1],
        "tcvn": tcvn_details,
        "esa": esa_details
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


@app.post("/classify/pepper_quality")
async def classify_pepper_quality_endpoint(
    tap_chat_la: float,
    hat_lep: float,
    hat_dau_dinh_vo: float,
    khoi_luong_theo_the_tich: float
):
    """
    Phân loại chất lượng hạt tiêu đen theo 4 chỉ tiêu vật lý
    
    Input (Query parameters):
    - tap_chat_la: Tạp chất lạ (%, khối lượng)
    - hat_lep: Hạt lép (%, khối lượng)
    - hat_dau_dinh_vo: Hạt đầu đinh hoặc hạt vỡ (%, khối lượng)
    - khoi_luong_theo_the_tich: Khối lượng theo thể tích (g/l)
    
    Output: JSON với phân loại (Loại 1, 2, 3 hoặc Không đạt) và chi tiết các chỉ tiêu
    
    Ví dụ: POST /classify/pepper_quality?tap_chat_la=0.4&hat_lep=5.5&hat_dau_dinh_vo=1.8&khoi_luong_theo_the_tich=560
    """
    try:
        result = classify_pepper_quality(
            tap_chat_la=tap_chat_la,
            hat_lep=hat_lep,
            hat_dau_dinh_vo=hat_dau_dinh_vo,
            khoi_luong_theo_the_tich=khoi_luong_theo_the_tich
        )
        
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi phân loại: {str(e)}")
