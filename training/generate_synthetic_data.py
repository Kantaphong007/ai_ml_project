"""
Synthetic Data Generator & Full Pipeline Verification
สร้างข้อมูลตัวอย่างจำลอง และทดสอบการเทรนโมเดล Cursor & Click + ประเมินผล
"""
import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    CALIBRATION_DATA_PATH, GESTURE_DATA_PATH, BASELINE_DATA_PATH,
    SCREEN_WIDTH, SCREEN_HEIGHT, SLIDING_WINDOW_SIZE
)
from core.feature_extractor import FeatureExtractor


def generate_synthetic_data():
    """สร้างข้อมูลตัวอย่างจำลองสำหรับทดสอบระบบการเทรนโมเดล"""
    print("  🎲 กำลังสร้างข้อมูลตัวอย่างจำลอง (Synthetic Data)...")
    
    # 1. Baseline
    import json
    baseline_data = {
        "baseline_mouth_width": 0.125,
        "baseline_ear_l": 0.30,
        "baseline_ear_r": 0.30,
        "num_frames": 90
    }
    os.makedirs(os.path.dirname(BASELINE_DATA_PATH), exist_ok=True)
    with open(BASELINE_DATA_PATH, 'w') as f:
        json.dump(baseline_data, f, indent=2)
    print(f"     ✅ Baseline data: {BASELINE_DATA_PATH}")
    
    # 2. Calibration Data (Cursor Regression)
    np.random.seed(42)
    n_samples = 405
    feature_names = FeatureExtractor.get_cursor_feature_names()
    
    # 7 features: [nose_x, nose_y, pitch, yaw, roll, nose_offset_x, nose_offset_y]
    nose_x = np.random.uniform(0.2, 0.8, n_samples)
    nose_y = np.random.uniform(0.2, 0.8, n_samples)
    yaw = (nose_x - 0.5) * 60.0 + np.random.normal(0, 2, n_samples)
    pitch = (nose_y - 0.5) * 40.0 + np.random.normal(0, 2, n_samples)
    roll = np.random.uniform(-10, 10, n_samples)
    nose_offset_x = (nose_x - 0.5) * 0.1
    nose_offset_y = (nose_y - 0.5) * 0.1
    
    X_cursor = np.column_stack([nose_x, nose_y, pitch, yaw, roll, nose_offset_x, nose_offset_y])
    
    # Direct realistic mapping across full screen width (0 to 1920)
    screen_x = np.interp(nose_x, [0.25, 0.75], [0, SCREEN_WIDTH]) + np.random.normal(0, 10, n_samples)
    screen_y = np.interp(nose_y, [0.25, 0.75], [0, SCREEN_HEIGHT]) + np.random.normal(0, 10, n_samples)
    screen_x = np.clip(screen_x, 0, SCREEN_WIDTH)
    screen_y = np.clip(screen_y, 0, SCREEN_HEIGHT)
    
    df_cursor = pd.DataFrame(X_cursor, columns=feature_names)
    df_cursor["screen_x"] = screen_x
    df_cursor["screen_y"] = screen_y
    
    os.makedirs(os.path.dirname(CALIBRATION_DATA_PATH), exist_ok=True)
    df_cursor.to_csv(CALIBRATION_DATA_PATH, index=False)
    print(f"     ✅ Calibration dataset ({n_samples} samples): {CALIBRATION_DATA_PATH}")
    
    # 3. Gesture Data (Click Classification)
    # 6 classes, 20 samples per class
    gesture_features = FeatureExtractor.get_click_sliding_feature_names(SLIDING_WINDOW_SIZE)
    rows = []
    
    for cls in range(6):
        for _ in range(25):
            window = []
            for f in range(SLIDING_WINDOW_SIZE):
                if cls == 0:  # Normal Blink
                    ear_l = 0.05 if 12 <= f <= 16 else 0.30 + np.random.normal(0, 0.01)
                    ear_r = 0.05 if 12 <= f <= 16 else 0.30 + np.random.normal(0, 0.01)
                    smile = np.random.normal(0, 0.02)
                elif cls == 1:  # Left Wink
                    ear_l = 0.08 if 8 <= f <= 22 else 0.30 + np.random.normal(0, 0.01)
                    ear_r = 0.30 + np.random.normal(0, 0.01)
                    smile = np.random.normal(0, 0.02)
                elif cls == 2:  # Right Wink
                    ear_l = 0.30 + np.random.normal(0, 0.01)
                    ear_r = 0.08 if 8 <= f <= 22 else 0.30 + np.random.normal(0, 0.01)
                    smile = np.random.normal(0, 0.02)
                elif cls == 3:  # Double Blink
                    ear_l = 0.05 if (5 <= f <= 8 or 18 <= f <= 22) else 0.30 + np.random.normal(0, 0.01)
                    ear_r = 0.05 if (5 <= f <= 8 or 18 <= f <= 22) else 0.30 + np.random.normal(0, 0.01)
                    smile = np.random.normal(0, 0.02)
                elif cls == 4:  # Extended Wink
                    ear_l = 0.05 if 2 <= f <= 28 else 0.30 + np.random.normal(0, 0.01)
                    ear_r = 0.30 + np.random.normal(0, 0.01)
                    smile = np.random.normal(0, 0.02)
                else:  # Smile + Nod
                    ear_l = 0.30 + np.random.normal(0, 0.01)
                    ear_r = 0.30 + np.random.normal(0, 0.01)
                    smile = 0.35 + np.random.normal(0, 0.03)
                
                window.extend([ear_l, ear_r, smile])
            window.append(cls)
            rows.append(window)
            
    df_gesture = pd.DataFrame(rows, columns=gesture_features + ["class"])
    os.makedirs(os.path.dirname(GESTURE_DATA_PATH), exist_ok=True)
    df_gesture.to_csv(GESTURE_DATA_PATH, index=False)
    print(f"     ✅ Gesture dataset ({len(rows)} samples): {GESTURE_DATA_PATH}")


if __name__ == "__main__":
    generate_synthetic_data()
