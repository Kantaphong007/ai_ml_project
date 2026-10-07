"""
EYE ORDER COME AI — Cursor Predictor (Regression Model)
ทำนายพิกัดเมาส์ (Screen_X, Screen_Y) จาก 7 features ปลายจมูกและหัว:
  [nose_x, nose_y, pitch, yaw, roll, nose_offset_x, nose_offset_y]
"""
import os
import time
from collections import deque
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
from core.feature_extractor import CURSOR_FEATURE_NAMES, CURSOR_FEATURE_VERSION
from utils.one_euro_filter import OneEuroFilter

# One-Euro: beta คือความไวต่อความเร็ว (หน่วย 1/px) — เคลื่อนเร็วกรองน้อยลงเพื่อไม่ให้หน่วง
# beta สูงเกิน → noise ของโมเดล (~100–150 px/s) ถูกนับเป็น "การเคลื่อนที่" ตัวกรองเลยกรองน้อยลงเอง
# (เดิม 0.012 ทำให้ปรับ Smoothing แทบไม่มีผล) — ค่านี้วัดจาก noise จริง: นิ่งขึ้น ~25%, หน่วงตอนหันเร็ว ~35 px
_BETA = 0.003
_D_CUTOFF = 0.5
# Sticky pointer: ความเร็วต่ำกว่านี้ (px/s) ต่อเนื่อง _SETTLE_S วินาที = หยุดแล้ว → ล็อกตำแหน่ง
_SETTLE_SPEED_PX_S = 90.0
_SETTLE_S = 0.15
_SPEED_WINDOW_S = 0.25
_BREAKOUT_AVG_S = 0.15
_BREAKOUT_FRAMES = 2          # ต้องหลุด deadzone ติดกันกี่เฟรมถึงเริ่มตาม (กัน noise ชั่วขณะ)
# ล็อกแน่นขึ้นเมื่อหยุดนาน: หลังนิ่ง _LOCK_DELAY_S วินาที deadzone ค่อยๆ ขยายถึง ×_LOCK_MAX ใน _LOCK_RAMP_S
# (จำลองการส่ายหัวตามธรรมชาติ ±15/25 px: เคอร์เซอร์ขยับเอง ~53 → ~2 ครั้ง/นาที, ขยับตั้งใจยังตามใน ~0.3 วิ)
_LOCK_MAX = 2.5
_LOCK_DELAY_S = 0.5
_LOCK_RAMP_S = 1.0
# วัดจาก noise ของโมเดลจริง (~12/18 px ต่อเฟรม): 24 px → หลุดล็อกเองราว 1 ครั้ง/นาที, เริ่มตามหัวใน ~0.1 วิ
DEFAULT_DEADZONE_PX = 24


class CursorPredictor:
    """ทำนายตำแหน่งเคอร์เซอร์จากฟีเจอร์ท่าทางศีรษะและปลายจมูก

    ขั้นตอน: โมเดล regression → ขยายรอบกึ่งกลางจอ (speed) → clamp ขอบจอ → One-Euro filter
             → sticky pointer (deadzone)

    Sticky pointer: หัวนิ่ง = เคอร์เซอร์ล็อกนิ่งสนิท (noise ไม่ทำให้สั่น)
                    ขยับเกิน deadzone = ตามทันทีไม่หน่วง → หยุดแล้วล็อกที่ตำแหน่งใหม่
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_idx = None            # คอลัมน์ที่โมเดลใช้ (จาก feature selection ตอนเทรน)
        self._smoothing_level = MOUSE_SMOOTHING_FRAMES  # 1 = ไวสุด, 15 = นิ่งสุด
        self.speed_multiplier = CURSOR_SPEED_MULTIPLIER
        self.invert_x = INVERT_CURSOR_X
        self.invert_y = INVERT_CURSOR_Y
        self._fx = OneEuroFilter(beta=_BETA, d_cutoff=_D_CUTOFF)
        self._fy = OneEuroFilter(beta=_BETA, d_cutoff=_D_CUTOFF)
        self.deadzone_px = DEFAULT_DEADZONE_PX
        self._axis_scale = (1.0, 1.0)      # รูปวงรีของ deadzone ตาม noise แต่ละแกนที่โมเดลวัดได้ตอนเทรน
        self._reset_sticky()
        self._apply_smoothing_level()

    def load(self, model_path=None):
        """โหลดโมเดลจากไฟล์"""
        if model_path is None:
            model_path = CURSOR_MODEL_PATH

        self.model = None
        if not os.path.exists(model_path):
            print("  ℹ️ ยังไม่มีโมเดล cursor (โหมด relative ใช้ได้เลย — โหมด hybrid/absolute "
                  "ต้อง Calibrate + Train Cursor)")
            return False

        data = joblib.load(model_path)
        if data.get("feature_version") != CURSOR_FEATURE_VERSION:
            print("  ⚠️ โมเดล cursor เป็นฟีเจอร์รุ่นเก่า (ก่อนใช้ head pose matrix) — "
                  "กด Calibrate + Train Cursor ใหม่เพื่อใช้โหมด hybrid/absolute")
            return False
        self.model = data["model"]
        self.scaler = data.get("scaler", None)
        cols = data["feature_cols"]
        self.feature_idx = [CURSOR_FEATURE_NAMES.index(c) for c in cols]
        noise = data.get("noise_px")
        if noise and min(noise) > 0:
            # แกนที่ noise มากกว่าได้ deadzone กว้างกว่า (เช่น แนวตั้งมักสั่นกว่าเพราะก้ม/เงยได้น้อย)
            m = (noise[0] + noise[1]) / 2.0
            self._axis_scale = tuple(min(1.6, max(0.6, n / m)) for n in noise)
            print(f"  ✅ noise ของโมเดล: x {noise[0]:.1f}px, y {noise[1]:.1f}px → "
                  f"deadzone รูปวงรี ×{self._axis_scale[0]:.2f}/×{self._axis_scale[1]:.2f}")
        else:
            self._axis_scale = (1.0, 1.0)
        print(f"  ✅ โหลดโมเดล cursor: {data.get('name', 'unknown')} (ฟีเจอร์: {', '.join(cols)})")
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

    def set_deadzone(self, px):
        """ตั้งรัศมีล็อกเคอร์เซอร์ตอนนิ่ง (px) — 0 = ปิด, มาก = นิ่งมากแต่ต้องขยับหัวมากขึ้นก่อนเคอร์เซอร์ตาม"""
        self.deadzone_px = max(0.0, float(px))

    def _apply_smoothing_level(self):
        # level 1 → 3.0 Hz, 5 → 1.07 Hz, 10 → 0.59 Hz, 15 → 0.41 Hz
        cutoff = 3.0 / (1.0 + 0.45 * (self._smoothing_level - 1))
        self._fx.min_cutoff = cutoff
        self._fy.min_cutoff = cutoff

    def predict_raw(self, features):
        """ทำนายพิกัดจากโมเดลโดยตรง (ยังไม่ขยาย/กรอง) — คืน (x, y) หรือ None"""
        if self.model is None:
            return None

        X = np.asarray(features, dtype=np.float64)
        if self.feature_idx is not None and len(X) != len(self.feature_idx):
            X = X[self.feature_idx]
        X = X.reshape(1, -1)
        if self.scaler is not None:
            X = self.scaler.transform(X)

        prediction = self.model.predict(X)[0]
        return float(prediction[0]), float(prediction[1])

    def predict(self, features, smooth=True, t=None):
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
            pred_x, pred_y = self._smooth(pred_x, pred_y, t)

        return pred_x, pred_y

    def _smooth(self, x, y, t=None):
        """One-Euro filter → sticky pointer"""
        t = time.perf_counter() if t is None else t
        sx = self._fx(x, t)
        sy = self._fy(y, t)
        return self._sticky(sx, sy, t)

    def _reset_sticky(self):
        self._anchor = None          # ตำแหน่งที่ล็อกไว้ตอนนิ่ง
        self._moving = False
        self._slow_since = None
        self._out_count = 0
        self._still_since = None     # เริ่มนิ่ง (ล็อก) ตั้งแต่เมื่อไหร่ → ยิ่งนาน deadzone ยิ่งกว้าง
        self._recent = deque()       # (t, x, y) ล่าสุด ~0.25 วิ — ใช้เฉลี่ยกัน noise

    def _sticky(self, x, y, t):
        rec = self._recent
        rec.append((t, x, y))
        while rec and t - rec[0][0] > _SPEED_WINDOW_S:
            rec.popleft()
        if self._anchor is None or self.deadzone_px <= 0:
            self._anchor = (x, y)
            self._still_since = t
            return x, y

        # ค่าเฉลี่ยช่วงสั้นๆ: noise เฟรมเดียวกระโดดเกิน deadzone ไม่ทำให้หลุดล็อก
        near = [(px, py) for pt, px, py in rec if t - pt <= _BREAKOUT_AVG_S]
        mx = sum(p[0] for p in near) / len(near)
        my = sum(p[1] for p in near) / len(near)

        if not self._moving:
            ax, ay = self._anchor
            held = t - (self._still_since if self._still_since is not None else t)
            lock = 1.0 + (_LOCK_MAX - 1.0) * min(1.0, max(0.0, (held - _LOCK_DELAY_S) / _LOCK_RAMP_S))
            rx = self.deadzone_px * self._axis_scale[0] * lock
            ry = self.deadzone_px * self._axis_scale[1] * lock
            if ((mx - ax) / rx) ** 2 + ((my - ay) / ry) ** 2 <= 1.0:
                self._out_count = 0
                return self._anchor            # ยังอยู่ใน deadzone → นิ่งสนิท
            self._out_count += 1
            if self._out_count < _BREAKOUT_FRAMES:
                return self._anchor
            self._moving = True
            self._out_count = 0
            self._slow_since = None

        # กำลังเคลื่อน: ตามค่าที่กรองแล้วตรงๆ; ความเร็ววัดจากระยะที่ไปได้ในหน้าต่าง ~0.25 วิ
        t0, x0, y0 = rec[0]
        speed = ((x - x0) ** 2 + (y - y0) ** 2) ** 0.5 / (t - t0) if t > t0 else 0.0
        if speed < _SETTLE_SPEED_PX_S and t - t0 >= _SPEED_WINDOW_S * 0.8:
            if self._slow_since is None:
                self._slow_since = t
            elif t - self._slow_since >= _SETTLE_S:
                self._moving = False
                self._still_since = t
                self._anchor = (mx, my)        # ล็อกที่ค่าเฉลี่ย (นิ่งกว่าค่าเฟรมเดียว)
                return self._anchor
        else:
            self._slow_since = None
        self._anchor = (x, y)
        return x, y

    def reset_smoothing(self):
        """ล้าง smoothing history (เช่น หลังหน้าหาย หรือเริ่มใหม่)"""
        self._fx.reset()
        self._fy.reset()
        self._reset_sticky()

    def is_loaded(self):
        """ตรวจสอบว่าโหลดโมเดลแล้วหรือยัง"""
        return self.model is not None
