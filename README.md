# NIR Spectroscopy Analysis - PLS Models

This repository contains code and models for Near-Infrared (NIR) spectroscopy analysis using Partial Least Squares (PLS) regression to predict various quality parameters of pepper samples.

## Project Structure

```
├── Code.ipynb                    # Main analysis notebook
├── mean_by_ma_mau.csv           # Input data file (averaged by sample code)
├── mo_hinh/                     # Folder containing trained models
│   ├── do_am.pkl               # Moisture content model (10-13%)
│   ├── tro_tong.pkl            # Total ash content model (3-7%)
│   ├── tro_khong_tan.pkl       # Acid-insoluble ash model (0.5-1.5%)
│   └── piperin.pkl             # Piperine content model (1-4%)
│   └── tinh_dau.pkl             # Tinh_dau (2-4%)
└── README.md                   # This file
```

## Overview

This project uses NIR spectroscopy data to predict quality parameters in pepper samples using PLS regression with variable selection. The analysis includes data preprocessing, model optimization, and result validation.

## Quality Parameters Analyzed

1. **Độ ẩm (Moisture Content)**: 10-13%
2. **Tro tổng (Total Ash)**: 3-7%  
3. **Tro không tan (Acid-insoluble Ash)**: 0.5-1.5%
4. **Piperin (Piperine Content)**: 2-4%
5. **Tinh dầu (Essential Oil)**: 1-4%

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
