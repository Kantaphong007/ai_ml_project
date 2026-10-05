"""
EYE ORDER COME AI — Cursor Predictor (Regression Model)
ทำนายพิกัดเมาส์ (Screen_X, Screen_Y) จาก 7 features ปลายจมูกและหัว:
  [nose_x, nose_y, pitch, yaw, roll, nose_offset_x, nose_offset_y]
"""
import os
import time
import numpy as np
import joblib

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    CURSOR_MODEL_PATH, MOUSE_SMOOTHING_FRAMES, CURSOR_SPEED_MULTIPLIER,
    INVERT_CURSOR_X, INVERT_CURSOR_Y, SCREEN_WIDTH, SCREEN_HEIGHT
)
from utils.one_euro_filter import OneEuroFilter

# One-Euro: beta คือความไวต่อความเร็ว (หน่วย 1/px) — เคลื่อนเร็วกรองน้อยลงเพื่อไม่ให้หน่วง
_BETA = 0.012
# ขยับน้อยกว่านี้ (px) ถือว่าเป็นสัญญาณรบกวน ไม่ต้องขยับเคอร์เซอร์
_STILL_RADIUS_PX = 2.0


class CursorPredictor:
    """ทำนายตำแหน่งเคอร์เซอร์จากฟีเจอร์ท่าทางศีรษะและปลายจมูก

    ขั้นตอน: โมเดล regression → ขยายรอบกึ่งกลางจอ (speed) → One-Euro filter
    การแมปเป็นเชิงเส้นล้วน ไม่มีการบิดโค้ง/deadzone จึงตรงกับตำแหน่งที่ calibrate ไว้
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self._smoothing_level = MOUSE_SMOOTHING_FRAMES  # 1 = ไวสุด, 15 = นิ่งสุด
        self.speed_multiplier = CURSOR_SPEED_MULTIPLIER
        self.invert_x = INVERT_CURSOR_X
        self.invert_y = INVERT_CURSOR_Y
        self._fx = OneEuroFilter(beta=_BETA)
        self._fy = OneEuroFilter(beta=_BETA)
        self._last_out = None
        self._apply_smoothing_level()

    def load(self, model_path=None):
        """โหลดโมเดลจากไฟล์"""
        if model_path is None:
            model_path = CURSOR_MODEL_PATH

        if not os.path.exists(model_path):
            print(f"  ❌ ไม่พบโมเดล: {model_path}")
            return False

        data = joblib.load(model_path)
        self.model = data["model"]
        self.scaler = data.get("scaler", None)
        print(f"  ✅ โหลดโมเดล cursor: {data.get('name', 'unknown')}")
        return True

    def set_inversion(self, invert_x: bool, invert_y: bool):
        """ตั้งค่ากลับทิศ X / Y"""
        self.invert_x = invert_x
        self.invert_y = invert_y

    def set_speed(self, speed: float):
        """ตั้งค่าความไวเมาส์ (0.2 ถึง 3.0) — ขยายระยะรอบกึ่งกลางจอ"""
        self.speed_multiplier = max(0.2, min(3.0, float(speed)))

    def set_smoothing_frames(self, n: int):
        """ตั้งค่าระดับ Smoothing (1 = ไวสุด, 15 = นิ่งสนิท)"""
        self._smoothing_level = max(1, min(15, int(n)))
        self._apply_smoothing_level()

    def _apply_smoothing_level(self):
        # level 1 → 3.0 Hz, 5 → 1.07 Hz, 10 → 0.59 Hz, 15 → 0.41 Hz
        cutoff = 3.0 / (1.0 + 0.45 * (self._smoothing_level - 1))
        self._fx.min_cutoff = cutoff
        self._fy.min_cutoff = cutoff

    def predict_raw(self, features):
        """ทำนายพิกัดจากโมเดลโดยตรง (ยังไม่ขยาย/กรอง) — คืน (x, y) หรือ None"""
        if self.model is None:
            return None

        X = np.asarray(features, dtype=np.float64).reshape(1, -1)
        if self.scaler is not None:
            X = self.scaler.transform(X)

        prediction = self.model.predict(X)[0]
        return float(prediction[0]), float(prediction[1])

    def predict(self, features, smooth=True):
        """ทำนายพิกัดเมาส์ (พิกเซลบนหน้าจอ) พร้อม clamp ในขอบเขตจอ"""
        raw = self.predict_raw(features)
        if raw is None:
            return None
        pred_x, pred_y = raw

        # ขยายเชิงเส้นรอบกึ่งกลางจอตาม speed (1.0 = ตรงตามที่ calibrate)
        cx, cy = SCREEN_WIDTH / 2.0, SCREEN_HEIGHT / 2.0
        pred_x = cx + (pred_x - cx) * self.speed_multiplier
        pred_y = cy + (pred_y - cy) * self.speed_multiplier

        pred_x = max(0.0, min(pred_x, float(SCREEN_WIDTH - 1)))
        pred_y = max(0.0, min(pred_y, float(SCREEN_HEIGHT - 1)))

        if self.invert_x:
            pred_x = (SCREEN_WIDTH - 1) - pred_x
        if self.invert_y:
            pred_y = (SCREEN_HEIGHT - 1) - pred_y

        if smooth:
            pred_x, pred_y = self._smooth(pred_x, pred_y)

        return pred_x, pred_y

    def _smooth(self, x, y):
        """One-Euro filter + กันสั่นระดับพิกเซล"""
        t = time.perf_counter()
        sx = self._fx(x, t)
        sy = self._fy(y, t)

        if self._last_out is not None:
            lx, ly = self._last_out
            if (sx - lx) ** 2 + (sy - ly) ** 2 < _STILL_RADIUS_PX ** 2:
                return lx, ly

        self._last_out = (sx, sy)
        return sx, sy

    def reset_smoothing(self):
        """ล้าง smoothing history (เช่น หลังหน้าหาย หรือเริ่มใหม่)"""
        self._fx.reset()
        self._fy.reset()
        self._last_out = None

    def is_loaded(self):
        """ตรวจสอบว่าโหลดโมเดลแล้วหรือยัง"""
        return self.model is not None
