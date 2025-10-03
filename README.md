# NIR Spectroscopy Analysis - PLS Models & Inference API

This repository contains code and models for Near-Infrared (NIR) spectroscopy analysis using Partial Least Squares (PLS) regression to predict various quality parameters of pepper samples.

## 🚀 Quick Start

### Option 1: Use the API (Recommended)
```bash
# Start the API server
cd my_app
./start_api.sh

# Or manually:
cd my_app
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Then access:
- API: http://localhost:8000
- Swagger UI: http://localhost:8000/docs

### Option 2: Use Jupyter Notebook
```bash
jupyter notebook Code.ipynb
```

## 📁 Project Structure

```
├── my_app/                     # API Application
│   ├── app.py                 # FastAPI application
│   ├── config.py              # API configuration
│   ├── test_api.py            # API testing script
│   ├── start_api.sh           # Startup script
│   ├── requirements.txt       # Python dependencies
│   ├── API_README.md          # API documentation
│   ├── Dockerfile             # Docker configuration
│   └── docker-compose.yml     # Docker Compose setup
├── Code.ipynb                 # Main analysis notebook
├── mean_by_ma_mau.csv        # Input data file
├── mo_hinh/                   # Trained models directory
    ├── do_am.pkl             # Moisture content model
    ├── tro_tong.pkl          # Total ash content model
    ├── tro_khong_tan.pkl     # Acid-insoluble ash model
    ├── piperin.pkl           # Piperine content model
    ├── Tinh_dau.pkl          # Essential oil model
    ├── random_forest_3regions_smote.pkl  # Origin classification model
    └── label_encoder_3regions_smote.pkl  # Label encoder for regions
```

## Overview

This project uses NIR spectroscopy data to predict quality parameters in pepper samples using PLS regression with variable selection. The analysis includes data preprocessing, model optimization, and result validation.

## 📊 Quality Parameters Analyzed

### Chemical Parameters
1. **Độ ẩm (Moisture Content)**: 10-13%
2. **Tro tổng (Total Ash)**: 3-7%  
3. **Tro không tan (Acid-insoluble Ash)**: 0.5-1.5%
4. **Piperin (Piperine Content)**: 2-4%
5. **Tinh dầu (Essential Oil)**: 1-4%

### Origin Classification
- **3 Main Regions**: Quảng Trị, Đắk Lắk, Gia Lai
- **Model**: Random Forest with SMOTE (95.17% accuracy)

## 🔬 API Usage

### 1. Predict All Parameters (with Standards Classification)
```bash
curl -X POST "http://localhost:8000/predict/all" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_nir_data.csv"
```

Response includes:
- All 5 chemical parameters
- Origin classification (3 regions)
- TCVN standard classification
- ESA standard classification

### 2. Predict TCVN & ESA Standards Only
```bash
curl -X POST "http://localhost:8000/predict/standards" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_nir_data.csv"
```

Returns:
- **TCVN Classification**: Based on moisture (≤13%), ash (≤7%), essential oil (≥2.0ml/100g), piperine (≥4%)
- **ESA Classification**: Based on moisture (≤12%), ash (≤7%), acid-insoluble ash (≤1.5%), essential oil (≥2.0ml/100g)

### 3. Classify Pepper Quality (Physical Standards)
```bash
curl -X POST "http://localhost:8000/classify/pepper_quality?tap_chat_la=0.4&hat_lep=5.5&hat_dau_dinh_vo=1.8&khoi_luong_theo_the_tich=560"
```

Parameters:
- `tap_chat_la`: Foreign matter (%)
- `hat_lep`: Light berries (%)
- `hat_dau_dinh_vo`: Pinheads/broken (%)
- `khoi_luong_theo_the_tich`: Bulk density (g/l)

Returns classification: **Loại 1**, **Loại 2**, **Loại 3**, or **Không đạt tiêu chuẩn**

### 4. Predict Origin Classification
```bash
curl -X POST "http://localhost:8000/predict/origin_classification" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_nir_data.csv"
```

Returns top 3 predictions with confidence scores for regions: Quảng Trị, Đắk Lắk, Gia Lai

### 5. Predict Single Chemical Parameter
```bash
curl -X POST "http://localhost:8000/predict/do_am" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_nir_data.csv"
```

Available models: `do_am`, `tro_tong`, `tro_khong_tan`, `piperin`, `Tinh_dau`

### Test the API
```bash
cd my_app
python test_api.py
```

## 🏆 Model Performance

### Origin Classification (Random Forest with SMOTE)
- **Overall Accuracy**: 95.17%
- **Quảng Trị**: 96.31% accuracy (11,788 samples)
- **Đắk Lắk**: 94.32% accuracy (5,708 samples)
- **Gia Lai**: 92.84% accuracy (3,730 samples)

### Standards Classification
- **TCVN Standards**: Automatic classification based on 4 criteria
- **ESA Standards**: Automatic classification based on 4 criteria
- **Physical Quality**: 3-tier grading system (Loại 1, 2, 3)

See [my_app/API_README.md](my_app/API_README.md) for detailed API documentation.

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
cd my_app
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

## 📋 API Endpoints Summary

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information and available models |
| `/predict/all` | POST | Predict all parameters + standards classification |
| `/predict/standards` | POST | TCVN & ESA standards classification only |
| `/predict/origin_classification` | POST | Origin classification (3 regions) |
| `/predict/{model_key}` | POST | Single parameter prediction |
| `/classify/pepper_quality` | POST | Physical quality grading (Loại 1, 2, 3) |
| `/health` | GET | Health check |
| `/docs` | GET | Interactive API documentation (Swagger UI) |

## Data Source

- **Input Data**: `mean_by_ma_mau.csv`
  - Contains averaged NIR spectral data by sample code (`ma_mau`)
  - Spectral range: typically 838-1750 nm
  - Quality parameters in the last 5 columns

## How to Use

### 1. Open the Analysis Notebook
```bash
jupyter notebook Code.ipynb
```

### 2. Run the Analysis Sections

The notebook is organized into the following sections:

#### Data Preprocessing
- Load and clean data from `mean_by_ma_mau.csv`
- Apply Savitzky-Golay filter with 2nd derivative
- Handle infinite values and missing data

#### Model Development
For each quality parameter, the notebook:
- Applies PLS variable selection with cross-validation
- Optimizes number of components and wavelength selection
- Fits the final model with range constraints
- Saves the trained model to `mo_hinh/` folder

#### Key Functions
- `pls_variable_selection()`: Optimizes PLS components and wavelength selection
- `RangeRandomizerRegressor`: Wrapper class that constrains predictions within valid ranges
- `print_wavelengths_kept()`: Shows which wavelengths are retained in the final model

### 3. Model Results

Each model section provides:
- Optimal number of PLS components
- Number of wavelengths selected
- Cross-validation R² and MSE
- Calibration R² and MSE
- Scatter plot of actual vs predicted values

### 4. Saved Models

Trained models are saved in the `mo_hinh/` folder:
- Models include both the PLS core and range constraints
- Can be loaded using `joblib.load()` for predictions
- Each model is optimized for its specific quality parameter range

## Model Performance Metrics

The analysis reports:
- **R² Calibration**: Coefficient of determination for training data
- **R² CV**: Cross-validation coefficient of determination  
- **MSE Calibration**: Mean squared error for training data
- **MSE CV**: Cross-validation mean squared error

## Requirements

```python
numpy
pandas
matplotlib
scikit-learn
scipy
joblib
```

## Usage Example

```python
import joblib
import numpy as np
from scipy.signal import savgol_filter

# Load a trained model
model = joblib.load('mo_hinh/do_am.pkl')

# Prepare new spectral data (apply same preprocessing)
X_new = savgol_filter(X_raw, 15, polyorder=2, deriv=2)
X_new_selected = X_new[:, kept_wavelength_indices]

# Make predictions
predictions = model.predict(X_new_selected)
```

## Notes

- All models use 2nd derivative Savitzky-Golay preprocessing (window=15, polyorder=2)
- Cross-validation uses 10-fold CV with shuffling
- Variable selection removes less important wavelengths to improve model robustness
- Range constraints ensure predictions stay within physically meaningful bounds
- Models are trained on averaged data by sample code to reduce noise

## Contact

For questions about the analysis or models, please refer to the detailed code in `Code.ipynb`.
