"""
EYE ORDER COME AI — Global Settings & Constants
กำหนดค่าคงที่สำหรับระบบควบคุมเคอร์เซอร์ด้วยสายตา
"""
import os
import pyautogui

# ──────────────────────────────────────────────
# Camera
# ──────────────────────────────────────────────
CAMERA_INDEX = 0
FPS = 30
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MIRROR_CAMERA = True   # กลับภาพซ้าย-ขวา (Mirror mode)

# ──────────────────────────────────────────────
# Screen — auto-detect
# ──────────────────────────────────────────────
try:
    SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
except Exception:
    SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080

# ──────────────────────────────────────────────
# Calibration (9-Point)
# ──────────────────────────────────────────────
CALIBRATION_POINTS = 9
CALIBRATION_DELAY_SEC = 1.0       # หน่วงก่อนเริ่มบันทึก (ให้ศีรษะเคลื่อนมาถึงจุดและนิ่งก่อน)
CALIBRATION_DURATION_SEC = 1.5    # บันทึกต่อจุด
FRAMES_PER_POINT = 45             # 30 FPS × 1.5s
CALIBRATION_MARGIN = 0.05         # 5% ขอบจอ

# ตำแหน่ง 9 จุด (normalized 0-1)
NINE_POINTS = [
    (0.05, 0.05),   # top-left
    (0.5, 0.05),    # top-center
    (0.95, 0.05),   # top-right
    (0.05, 0.5),    # mid-left
    (0.5, 0.5),     # center
    (0.95, 0.5),    # mid-right
    (0.05, 0.95),   # bottom-left
    (0.5, 0.95),    # bottom-center
    (0.95, 0.95),   # bottom-right
]

# ──────────────────────────────────────────────
# Feature Extraction — Eye Aspect Ratio
# ──────────────────────────────────────────────
EAR_THRESHOLD_OPEN = 0.25        # ค่า EAR เมื่อลืมตา
SLIDING_WINDOW_SIZE = 30         # 30 เฟรม = 1 วินาที
SMILE_THRESHOLD = 0.25           # ΔSmile +25%

# ──────────────────────────────────────────────
# Gesture Timing (frames @ 30 FPS)
# ──────────────────────────────────────────────
NORMAL_BLINK_MAX_FRAMES = 6      # ≤200ms
WINK_MIN_FRAMES = 10             # 350ms
WINK_MAX_FRAMES = 18             # 600ms
DOUBLE_BLINK_WINDOW_FRAMES = 15  # 500ms
EXTENDED_WINK_MIN_FRAMES = 30    # 1000ms
SMILE_HOLD_MIN_FRAMES = 6       # 200ms

# ──────────────────────────────────────────────
# Mouse Control
# ──────────────────────────────────────────────
MOUSE_SMOOTHING_FRAMES = 5       # moving average
CURSOR_SPEED_MULTIPLIER = 1.0
SCROLL_SPEED = 3                 # lines per scroll tick
INVERT_CURSOR_X = False          # กลับทิศเมาส์ซ้าย-ขวา (False = ตามตำแหน่ง calibration จริง)
INVERT_CURSOR_Y = False          # กลับทิศเมาส์บน-ล่าง (False = ตามตำแหน่ง calibration จริง)

# ──────────────────────────────────────────────
# On-Screen Keyboard
# ──────────────────────────────────────────────
KEYBOARD_KEY_SIZE = 60           # px per key
KEYBOARD_OPACITY = 0.85
KEYBOARD_DWELL_TIME_MS = 800     # ms to trigger key press
KEYBOARD_FONT_SIZE = 16

# ──────────────────────────────────────────────
# Gesture Collection
# ──────────────────────────────────────────────
BASELINE_DURATION_SEC = 3.0      # บันทึกค่าฐาน 3 วินาที
GESTURE_REPS_PER_CLASS = 25      # จำนวนรอบต่อคลาส
NUM_GESTURE_CLASSES = 6

# Class labels
CLASS_NAMES = {
    0: "Normal Blink",
    1: "Left Wink",
    2: "Right Wink",
    3: "Double Blink",
    4: "Extended Wink (Drag)",
    5: "Smile + Head Nod (Scroll)",
}

# Feature Names
CURSOR_FEATURE_NAMES = [
    "nose_x", "nose_y",
    "pitch", "yaw", "roll",
    "nose_offset_x", "nose_offset_y"
]

# ──────────────────────────────────────────────
# Model Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CURSOR_MODEL_PATH = os.path.join(MODELS_DIR, "cursor_model.pkl")
CLICK_MODEL_PATH = os.path.join(MODELS_DIR, "click_model.pkl")
CALIBRATION_DATA_PATH = os.path.join(DATA_DIR, "calibration_data.csv")
GESTURE_DATA_PATH = os.path.join(DATA_DIR, "gesture_data.csv")
BASELINE_DATA_PATH = os.path.join(DATA_DIR, "baseline.json")
USER_SETTINGS_PATH = os.path.join(DATA_DIR, "user_settings.json")

# สร้างโฟลเดอร์ถ้ายังไม่มี
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

import json

def get_default_settings():
    return {
        "smoothing": MOUSE_SMOOTHING_FRAMES,
        "speed": CURSOR_SPEED_MULTIPLIER,
        "camera": CAMERA_INDEX,
        "mirror": MIRROR_CAMERA,
        "invert_x": INVERT_CURSOR_X,
        "invert_y": INVERT_CURSOR_Y,
        "dwell_time": KEYBOARD_DWELL_TIME_MS,
        "keyboard_opacity": int(KEYBOARD_OPACITY * 100),
        "show_overlay": True,
    }

def load_user_settings():
    defaults = get_default_settings()
    if os.path.exists(USER_SETTINGS_PATH):
        try:
            with open(USER_SETTINGS_PATH, 'r', encoding='utf-8') as f:
                saved = json.load(f)
            defaults.update(saved)
        except Exception:
            pass
    return defaults

def save_user_settings(settings_dict):
    os.makedirs(os.path.dirname(USER_SETTINGS_PATH), exist_ok=True)
    try:
        with open(USER_SETTINGS_PATH, 'w', encoding='utf-8') as f:
            json.dump(settings_dict, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving settings: {e}")
        return False

# ──────────────────────────────────────────────
# MediaPipe Face Mesh Landmark Indices
# ──────────────────────────────────────────────
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
LEFT_IRIS_IDX = [468, 469, 470, 471, 472]
RIGHT_IRIS_IDX = [473, 474, 475, 476, 477]
MOUTH_LEFT_IDX = 61
MOUTH_RIGHT_IDX = 291
HEAD_POSE_IDX = [1, 152, 33, 263, 61, 291]

# PyAutoGUI: ปิด failsafe เพราะเคอร์เซอร์ที่คุมด้วยศีรษะไปถึงมุมจอได้ปกติ
# (failsafe จะ raise exception ทุกครั้งที่คลิก/พิมพ์ตอนเคอร์เซอร์อยู่มุมจอ)
# การหยุดฉุกเฉินใช้ปุ่ม Pause/Break แทน (ดู core/pipeline.py)
try:
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0
except Exception:
    pass
