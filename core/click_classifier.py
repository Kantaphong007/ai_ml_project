"""
EYE ORDER COME AI — Click Classifier (Classification Model)
จำแนกท่าทางตาจาก "episode" (ช่วงหลับตาที่ GestureDetector ตรวจพบ) เป็น 5 คลาส:
  Class 0: No Action (กะพริบปกติ / noise)  → เพิกเฉย
  Class 1: Left Wink                       → คลิกซ้าย
  Class 2: Right Wink                      → คลิกขวา
  Class 3: Double Blink                    → ดับเบิลคลิก
  Class 4: Extended Wink                   → Drag mode
ฟีเจอร์ 19 ตัวต่อ episode (ดู core/episode_features.py) — เทรนด้วย training/train_click_model.py
"""
import os
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import CLICK_MODEL_PATH
from core.episode_features import EYE_CLASS_NAMES, FEATURE_NAMES, FEATURE_VERSION


class ClickClassifier:
    """ห่อโมเดล sklearn ให้ GestureDetector เรียก predict(features) → (class_id, confidence)

    Example:
        >>> clf = ClickClassifier()
        >>> if clf.load() and clf.use_live:
        ...     detector.set_classifier(clf)
    """

    def __init__(self):
        self.model = None
        self.name = None
        self.use_live = False          # False = ผลประเมินแพ้ rule-based → pipeline ใช้ rule แทน
        self.info = {}
        self.class_names = EYE_CLASS_NAMES

    def load(self, model_path=None, verbose=True):
        """โหลดโมเดล — คืน False ถ้าไม่มีไฟล์ หรือเป็นโมเดลรูปแบบเก่า (sliding window 90 ฟีเจอร์)"""
        model_path = model_path or CLICK_MODEL_PATH
        self.model = None
        if not os.path.exists(model_path):
            if verbose:
                print(f"  ℹ️ ยังไม่มีโมเดล click ({model_path}) — ใช้ rule-based")
            return False

        data = joblib.load(model_path)
        if data.get("feature_version") != FEATURE_VERSION or data.get("feature_names") != FEATURE_NAMES:
            if verbose:
                print("  ⚠️ โมเดล click เป็นรูปแบบเก่า — ใช้ rule-based "
                      "(Record Signals แล้วกด Train Click ใหม่)")
            return False

        self.model = data["model"]
        self.name = data.get("name", "unknown")
        self.use_live = bool(data.get("use_live", True))
        self.info = data
        if verbose:
            mode = "ใช้สั่งคลิกจริง" if self.use_live else "ไม่ใช้ (แพ้ rule-based ตอนประเมิน)"
            print(f"  ✅ โหลดโมเดล click: {self.name} — {mode}")
        return True

    def predict(self, features):
        """
        Args:
            features: list/array ตามลำดับ FEATURE_NAMES

        Returns:
            tuple: (class_id, confidence)
        """
        X = np.asarray(features, dtype=np.float64).reshape(1, -1)
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            i = int(np.argmax(proba))
            return int(self.model.classes_[i]), float(proba[i])
        return int(self.model.predict(X)[0]), 1.0

    def segmentation_mismatch(self, gesture_params):
        """พารามิเตอร์ที่ใช้หา episode ตอนเทรน ≠ ตอนนี้ (จูนใหม่หลังเทรน) → ควรเทรนใหม่"""
        seg = self.info.get("segmentation", {})
        return [k for k, v in seg.items()
                if abs(float(getattr(gesture_params, k, v)) - float(v)) > 1e-6]

    def get_class_name(self, class_id):
        return self.class_names.get(class_id, f"Unknown ({class_id})")

    def is_loaded(self):
        return self.model is not None
