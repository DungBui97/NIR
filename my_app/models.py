"""
Custom model classes for NIR prediction
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class RangeRandomizerRegressor(BaseEstimator, RegressorMixin):
    """
    Wrapper cho bất kỳ mô hình hồi quy (vd: PLSRegression).
    Nếu y_pred ngoài [low, high] -> thay bằng giá trị random trong (low, high).
    Giữ nguyên nếu trong khoảng.
    """
    def __init__(self, base_model, low=10.0, high=13.0, random_state=None):
        self.base_model = base_model
        self.low = float(low)
        self.high = float(high)
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    def fit(self, X, y):
        self.base_model.fit(X, y)
        return self

    def predict(self, X):
        y = self.base_model.predict(X)
        y = np.asarray(y).reshape(-1)

        # mask ngoài [low, high]
        mask = (y < self.low) | (y > self.high)
        if np.any(mask):
            # Tạo seed động dựa trên giá trị dự đoán để đảm bảo:
            # 1. Cùng input → cùng output (reproducible)
            # 2. Khác input → khác output (không trùng lặp giữa các mẫu)
            seed = int(np.abs(y[mask]).sum() * 1000000) % (2**31 - 1)
            
            # Nếu có random_state, mix với seed động để tăng tính ngẫu nhiên
            if self.random_state is not None:
                seed = (seed + self.random_state) % (2**31 - 1)
            
            rng = np.random.default_rng(seed)
            
            # sinh ngẫu nhiên trong (low, high) (loại trừ biên)
            lo = np.nextafter(self.low, self.high)
            hi = np.nextafter(self.high, self.low)
            y_rand = rng.uniform(lo, hi, size=mask.sum())
            y_adj = y.copy()
            y_adj[mask] = y_rand
        else:
            y_adj = y

        return y_adj.reshape(-1, 1)  # sklearn quy ước trả về (n, 1)

    # Tùy chọn: expose thuộc tính bên trong khi cần (ví dụ để lưu/inspect)
    def __getattr__(self, name):
        # Cho phép truy cập thuộc tính của base_model (coef_, n_components, ...)
        return getattr(self.base_model, name)
