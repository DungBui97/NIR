# 🔬 NIR Inference API

FastAPI application for predicting pepper quality parameters from Near-Infrared (NIR) spectroscopy data.

## 📊 Models

API dự đoán 5 chỉ tiêu chất lượng hạt tiêu từ phổ NIR:

| Model | Parameter | Range | Unit |
|-------|-----------|-------|------|
| `do_am` | Độ ẩm | 10.0 - 13.0 | % |
| `tro_tong` | Tro tổng | 3.0 - 7.0 | % |
| `tro_khong_tan` | Tro không tan | 0.5 - 1.5 | % |
| `piperin` | Piperin | 2.0 - 4.0 | % |
| `Tinh_dau` | Tinh dầu | 1.0 - 4.0 | % |

## 🚀 Quick Start

### Docker (Recommended)
```bash
docker-compose up --build -d
curl http://localhost:8000/health
python3 test_real_data.py
```

### Local
```bash
pip install -r requirements.txt
./start_api.sh
```

## 📁 Files

```
my_app/
├── app.py              # Main API
├── config.py           # Configuration  
├── models.py           # Custom classes
├── requirements.txt    # Dependencies
├── Dockerfile          # Docker config
├── docker-compose.yml  # Docker Compose
├── start_api.sh        # Startup script
├── test_real_data.py   # Test script
├── test_nir_data.csv   # Sample data
└── indices/            # Wavelength indices
    ├── do_am_indices.csv
    ├── tro_tong_indices.csv
    ├── tro_khong_tan_indices.csv
    ├── piperin_indices.csv
    └── Tinh_dau_indices.csv
```

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/predict/all` | POST | Predict all 5 models |
| `/predict/{model_key}` | POST | Predict single model |
| `/docs` | GET | Swagger UI |

## 💻 Usage

### Check Health
```bash
curl http://localhost:8000/health
```

### Predict All Models
```bash
curl -X POST "http://localhost:8000/predict/all" -F "file=@test_nir_data.csv"
```

### Predict Single Model
```bash
curl -X POST "http://localhost:8000/predict/do_am" -F "file=@test_nir_data.csv"
```

### Python
```python
import requests

with open('test_nir_data.csv', 'rb') as f:
    response = requests.post('http://localhost:8000/predict/all', files={'file': f})
    print(response.json())
```

### Swagger UI
http://localhost:8000/docs

## 📊 Input/Output

### Input CSV
- No header
- Each row = 1 sample
- Each column = 1 wavelength (228 expected)
- Numeric values only
- Comma-separated

### Output JSON
```json
{
  "n_samples": 3,
  "n_wavelengths": 228,
  "results": {
    "do_am": {
      "name": "Độ ẩm",
      "unit": "%",
      "valid_range": {"low": 10.0, "high": 13.0},
      "predictions": [12.09, 10.28, 12.93],
      "n_features_used": 27
    }
  }
}
```

## ⚙️ Preprocessing

```
CSV → Numpy → Savgol Filter (15,2,2) → Feature Selection → PLS Model → Range Check → JSON
```

- **Savitzky-Golay**: window=15, polyorder=2, deriv=2
- **Feature Selection**: Model-specific wavelength indices
- **Range Validation**: Replace out-of-range predictions with random value in valid range

## 🛠️ Configuration

Edit `config.py` to modify:
- Model paths and settings
- Preprocessing parameters
- Valid ranges

## 🧪 Testing

```bash
python3 test_real_data.py
```

## 🐳 Docker

```bash
# Start
docker-compose up -d

# Logs
docker-compose logs -f

# Stop
docker-compose down

# Rebuild
docker-compose down -v && docker-compose up --build -d
```

## 📦 Dependencies

- FastAPI 0.104.1
- Uvicorn 0.24.0
- Pandas 2.1.3
- NumPy 1.26.2
- SciPy 1.11.4
- Scikit-learn 1.3.2

## 🔧 Troubleshooting

### Models not loading
```bash
docker-compose logs nir-api | grep "Warning"
ls -lh ../mo_hinh/*.pkl
```

### API not responding
```bash
docker-compose ps
netstat -tuln | grep 8000
docker-compose restart
```

### Prediction errors
- Verify CSV has 228 columns
- Check format (no header, comma-separated)
- Ensure numeric values only

## 👨‍💻 Author

NIR Spectroscopy Team - Sun Asterisk Vietnam
