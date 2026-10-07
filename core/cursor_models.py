"""
EYE ORDER COME AI — Edge-safe regression สำหรับเคอร์เซอร์

ปัญหา: โมเดลอย่าง SVR (RBF) / MLP / Random Forest "ไม่รู้" ว่าจะทำอะไรนอกช่วงที่ calibrate
  → หันหัวเลยขอบจอ ค่าทำนายวิ่ง "กลับเข้ากลางจอ" (RBF ลู่เข้าค่าเฉลี่ยเมื่ออยู่ไกลจากข้อมูล)

แก้: ทำนาย = ส่วนเชิงเส้น (ต่อเนื่องไปทางเดิมเสมอ) + ส่วนแก้ความโค้ง (ใช้เฉพาะในช่วงที่ calibrate)
       ŷ(x) = Linear(x) + Residual(clip(x, ช่วงที่ calibrate))
  - ในช่วง calibrate: Residual แก้ความโค้งได้เต็มที่ แม่นเท่าโมเดลเดิม
  - นอกช่วง: Residual ค้างที่ค่าขอบ ส่วนเชิงเส้นยังเพิ่มต่อ → ค่าเลยขอบจอ → ถูก clamp ติดขอบ
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import Ridge


class EdgeSafeRegressor(BaseEstimator, RegressorMixin):
    """ห่อโมเดล regression ใดๆ ให้ extrapolate แบบเชิงเส้นนอกช่วงข้อมูล

    Args:
        estimator: โมเดลที่ใช้ทำนายส่วนที่เหลือ (residual) จากเส้นตรง
        alpha: ค่า regularization ของส่วนเชิงเส้น (Ridge)
        clip_quantile: ช่วงข้อมูล = quantile [q, 1−q] ของแต่ละฟีเจอร์ (ตัด outlier)
    """

    def __init__(self, estimator=None, alpha=1.0, clip_quantile=0.01):
        self.estimator = estimator
        self.alpha = alpha
        self.clip_quantile = clip_quantile

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        q = self.clip_quantile
        self.lo_ = np.quantile(X, q, axis=0)
        self.hi_ = np.quantile(X, 1.0 - q, axis=0)
        self.linear_ = Ridge(alpha=self.alpha).fit(X, y)
        residual = y - self.linear_.predict(X)
        self.residual_ = clone(self.estimator).fit(self._clip(X), residual)
        return self

    def _clip(self, X):
        return np.clip(X, self.lo_, self.hi_)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self.linear_.predict(X) + self.residual_.predict(self._clip(X))
