"""
EYE ORDER COME AI — Click Classifier (Classification Model)
จำแนกคำสั่งคลิก 6 คลาส จาก sliding window 30 เฟรม (90 features):
  Class 0: Normal Blink     → เพิกเฉย
  Class 1: Left Wink        → คลิกซ้าย
  Class 2: Right Wink       → คลิกขวา
  Class 3: Double Blink     → ดับเบิลคลิก
  Class 4: Extended Wink    → Drag mode
  Class 5: Smile + Head Nod → Scroll mode
"""
import os
import numpy as np
import joblib

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import CLICK_MODEL_PATH, CLASS_NAMES


class ClickClassifier:
    """จำแนกคำสั่งคลิกจาก time-series features ใบหน้า
    
    รับ input เป็น flatten vector ขนาด 90 
    (30 เฟรม × 3 features: EAR_L, EAR_R, ΔSmile)
    
    Example:
        >>> classifier = ClickClassifier()
        >>> classifier.load()
        >>> vector = sliding_queue.flatten()  # shape (90,)
        >>> cls, confidence = classifier.predict(vector)
        >>> print(f"Class: {cls}, Confidence: {confidence:.2f}")
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.class_names = CLASS_NAMES
    
    def load(self, model_path=None):
        """โหลดโมเดลจากไฟล์
        
        Args:
            model_path: path ไปยัง .pkl file (None = ใช้ default)
        
        Returns:
            bool: สำเร็จหรือไม่
        """
        if model_path is None:
            model_path = CLICK_MODEL_PATH
        
        if not os.path.exists(model_path):
            print(f"  ❌ ไม่พบโมเดล: {model_path}")
            return False
        
        data = joblib.load(model_path)
        self.model = data["model"]
        self.scaler = data.get("scaler", None)
        print(f"  ✅ โหลดโมเดล click: {data.get('name', 'unknown')}")
        return True
    
    def predict(self, features_flat):
        """ทำนายคำสั่งคลิก
        
        Args:
            features_flat: np.ndarray shape (90,) — flatten sliding window
        
        Returns:
            tuple: (class_id, confidence)
                   class_id: int 0-5
                   confidence: float 0-1 (probability ของ class ที่ทำนาย)
            None: ถ้ายังไม่โหลดโมเดล
        """
        if self.model is None:
            return None
        
        X = features_flat.reshape(1, -1)
        
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        # ทำนาย class
        predicted_class = int(self.model.predict(X)[0])
        
        # confidence (probability)
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X)[0]
            confidence = float(proba[predicted_class])
        else:
            confidence = 1.0  # SVM อาจไม่มี probability
        
        return predicted_class, confidence
    
    def predict_with_proba(self, features_flat):
        """ทำนายพร้อม probability ของทุก class
        
        Args:
            features_flat: np.ndarray shape (90,)
        
        Returns:
            tuple: (class_id, probabilities_dict)
        """
        if self.model is None:
            return None
        
        X = features_flat.reshape(1, -1)
        
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        predicted_class = int(self.model.predict(X)[0])
        
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X)[0]
            proba_dict = {
                int(cls): float(p) 
                for cls, p in zip(self.model.classes_, proba)
            }
        else:
            proba_dict = {predicted_class: 1.0}
        
        return predicted_class, proba_dict
    
    def get_class_name(self, class_id):
        """ดึงชื่อ class
        
        Args:
            class_id: int 0-5
        
        Returns:
            str: ชื่อ class
        """
        return self.class_names.get(class_id, f"Unknown ({class_id})")
    
    def is_loaded(self):
        """ตรวจสอบว่าโหลดโมเดลแล้วหรือยัง"""
        return self.model is not None
