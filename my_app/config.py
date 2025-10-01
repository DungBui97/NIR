"""
Configuration file cho NIR Inference API
Nếu mỗi model có file kept_wavelength_indices riêng, cập nhật path ở đây
"""

import os

# Thư mục chứa models
# Sử dụng đường dẫn tuyệt đối trong Docker hoặc tương đối khi chạy local
MODEL_DIR = os.getenv("MODEL_DIR", "../mo_hinh")

# Cấu hình cho từng model
# Nếu có file kept_indices riêng, điền path vào 'kept_indices_path'
MODEL_CONFIGS = {
    "do_am": {
        "path": os.path.join(MODEL_DIR, "do_am.pkl"),
        "kept_indices_path": "indices/do_am_indices.csv",
        "name": "Độ ẩm",
        "unit": "%",
        "description": "Độ ẩm của mẫu hạt tiêu",
        "range_low": 10.0,
        "range_high": 13.0
    },
    "tro_tong": {
        "path": os.path.join(MODEL_DIR, "tro_tong.pkl"),
        "kept_indices_path": "indices/tro_tong_indices.csv",
        "name": "Tro tổng",
        "unit": "%",
        "description": "Hàm lượng tro tổng",
        "range_low": 3.0,
        "range_high": 7.0
    },
    "tro_khong_tan": {
        "path": os.path.join(MODEL_DIR, "tro_khong_tan.pkl"),
        "kept_indices_path": "indices/tro_khong_tan_indices.csv",
        "name": "Tro không tan",
        "unit": "%",
        "description": "Hàm lượng tro không tan trong acid",
        "range_low": 0.5,
        "range_high": 1.5
    },
    "piperin": {
        "path": os.path.join(MODEL_DIR, "piperin.pkl"),
        "kept_indices_path": "indices/piperin_indices.csv",
        "name": "Piperin",
        "unit": "%",
        "description": "Hàm lượng piperin (chất cay)",
        "range_low": 2.0,
        "range_high": 4.0
    },
    "Tinh_dau": {
        "path": os.path.join(MODEL_DIR, "Tinh_dau.pkl"),
        "kept_indices_path": "indices/Tinh_dau_indices.csv",
        "name": "Tinh dầu",
        "unit": "%",
        "description": "Hàm lượng tinh dầu",
        "range_low": 1.0,
        "range_high": 4.0
    }
}

# Tham số preprocessing
SAVGOL_WINDOW = 15
SAVGOL_POLYORDER = 2
SAVGOL_DERIV = 2

# API settings
API_TITLE = "NIR Inference API"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
API để dự đoán các chỉ tiêu chất lượng hạt tiêu từ phổ NIR.

## Features:
- Dự đoán 5 chỉ tiêu: Độ ẩm, Tro tổng, Tro không tan, Piperin, Tinh dầu
- Preprocessing tự động với Savitzky-Golay filter
- Feature selection tùy chỉnh cho từng model
- Support nhiều định dạng CSV

## Models:
- **do_am**: Độ ẩm (%)
- **tro_tong**: Tro tổng (%)
- **tro_khong_tan**: Tro không tan (%)
- **piperin**: Piperin (%)
- **Tinh_dau**: Tinh dầu (%)
"""
