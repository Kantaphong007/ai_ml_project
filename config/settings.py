"""
EYE ORDER COME AI — Global Settings & Constants
ค่าคงที่ของระบบ + การตั้งค่าของผู้ใช้ (data/user_settings.json)

พารามิเตอร์ที่จูนจากข้อมูล (การตรวจจับท่าทาง / scroll) อยู่ที่ config/tuning.py
"""
import json
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

# PyAutoGUI: ปิด failsafe เพราะเคอร์เซอร์ที่คุมด้วยศีรษะไปถึงมุมจอได้ปกติ
# (failsafe จะ raise exception ทุกครั้งที่คลิก/พิมพ์ตอนเคอร์เซอร์อยู่มุมจอ)
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

# ──────────────────────────────────────────────
# Calibration (cursor regression dataset)
# ──────────────────────────────────────────────
CALIBRATION_DELAY_SEC = 1.0       # หน่วงก่อนเริ่มบันทึก (ให้ศีรษะเคลื่อนมาถึงจุดและนิ่งก่อน)
CALIBRATION_DURATION_SEC = 1.5    # บันทึกต่อจุด

# ตำแหน่งจุด calibration (normalized 0-1) — 9 จุดหลัก + 4 จุดภายในแต่ละควอดรันต์
NINE_POINTS = [
    (0.05, 0.05), (0.5, 0.05), (0.95, 0.05),
    (0.05, 0.5), (0.5, 0.5), (0.95, 0.5),
    (0.05, 0.95), (0.5, 0.95), (0.95, 0.95),
    (0.275, 0.275), (0.725, 0.275), (0.275, 0.725), (0.725, 0.725),
]

BASELINE_DURATION_SEC = 3.0       # บันทึกค่าฐาน (หน้าปกติ) 3 วินาที

# ──────────────────────────────────────────────
# Mouse Control (ค่าเริ่มต้น — ปรับได้ใน Settings)
# ──────────────────────────────────────────────
MOUSE_SMOOTHING_FRAMES = 5
CURSOR_SPEED_MULTIPLIER = 1.0
SCROLL_SPEED = 3                  # บรรทัดต่อ notch (macOS/Linux)
INVERT_CURSOR_X = False
INVERT_CURSOR_Y = False

# ──────────────────────────────────────────────
# On-Screen Keyboard
# ──────────────────────────────────────────────
KEYBOARD_KEY_SIZE = 60            # px per key
KEYBOARD_OPACITY = 0.85
KEYBOARD_DWELL_TIME_MS = 800      # ms to trigger key press
KEYBOARD_FONT_SIZE = 16

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CURSOR_MODEL_PATH = os.path.join(MODELS_DIR, "cursor_model.pkl")
CLICK_MODEL_PATH = os.path.join(MODELS_DIR, "click_model.pkl")
CALIBRATION_DATA_PATH = os.path.join(DATA_DIR, "calibration_data.csv")
BASELINE_DATA_PATH = os.path.join(DATA_DIR, "baseline.json")
USER_SETTINGS_PATH = os.path.join(DATA_DIR, "user_settings.json")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# ──────────────────────────────────────────────
# User settings
# ──────────────────────────────────────────────
def get_default_settings():
    return {
        "cursor_mode": "hybrid",  # hybrid / relative / absolute (ดู core/head_pointer.py)
        "smoothing": MOUSE_SMOOTHING_FRAMES,
        "speed": CURSOR_SPEED_MULTIPLIER,
        "stability": 24,          # รัศมีล็อกเคอร์เซอร์ตอนหัวนิ่ง (px) 0 = ปิด
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
                defaults.update(json.load(f))
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
