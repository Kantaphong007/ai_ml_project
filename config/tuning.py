"""
EYE ORDER COME AI — Tunable parameters (ค่าที่จูนด้วยข้อมูลจริง)

พารามิเตอร์ทั้งหมดของการตรวจจับท่าทาง (ตา/ยิ้ม) และโหมด scroll อยู่ที่นี่ที่เดียว
ค่า default ด้านล่างเป็นค่าเริ่มต้นที่ใช้งานได้ — ค่าที่จูนแล้วจะถูกเขียนลง
data/gesture_tuning.json โดย training/tune_gestures.py และโหลดทับตอนเริ่มระบบ

ขั้นตอนจูน:
  1. python calibration/signal_recorder.py   → บันทึกสัญญาณดิบพร้อม label (data/gesture_signals.csv)
  2. python training/tune_gestures.py        → replay + ค้นหาค่าที่ดีที่สุด → data/gesture_tuning.json

ไฟล์นี้ต้องไม่ import อะไรหนัก (pyautogui/cv2) เพื่อให้ test/สคริปต์จูนใช้ได้โดยไม่ต้องมีกล้อง
"""
import json
import os
from dataclasses import dataclass, asdict, fields

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TUNING_PATH = os.path.join(_PROJECT_ROOT, "data", "gesture_tuning.json")


@dataclass
class GestureParams:
    """พารามิเตอร์ของ GestureDetector (ratio = EAR ÷ EAR ตอนลืมตาของผู้ใช้)"""
    # ── ระดับการปิดตา ──
    close_ratio: float = 0.60         # ต่ำกว่านี้ = เริ่มหลับตา (เริ่ม episode)
    open_ratio: float = 0.75          # สองตาสูงกว่านี้ = ลืมตาแล้ว (จบ episode)
    freeze_ratio: float = 0.80        # ต่ำกว่านี้ = เริ่มหลับตา → ตรึงเคอร์เซอร์
    wink_asym: float = 0.22           # |ratio_L - ratio_R| เกินนี้ในเฟรมนั้น = ตาข้างเดียวปิด
    wink_side_frac: float = 0.60      # สัดส่วนเฟรมใน episode ที่ต้องเป็นข้างเดียวกันถึงนับเป็นขยิบ
    blink_depth: float = 0.55         # กะพริบที่นับรวมเป็นดับเบิลคลิก: ตาที่เปิดกว้างกว่าต้องลงถึงระดับนี้

    # ── ยิ้ม (ΔSmile = สัดส่วนปากกว้างขึ้นจากหน้าปกติ) ──
    smile_on: float = 0.20
    smile_off: float = 0.12
    smile_freeze: float = 0.10

    # ── เวลา (วินาที) ──
    wink_min_s: float = 0.25
    wink_max_s: float = 1.10
    drag_hold_s: float = 1.20
    blink_max_s: float = 0.40
    double_blink_gap_s: float = 0.70  # จากลืมตาครั้งแรก → เริ่มหลับครั้งที่สอง
    smile_hold_s: float = 0.50
    open_confirm_s: float = 0.06      # ต้องลืมตาต่อเนื่องเท่านี้ถึงจบ episode (กันค่ากระตุกเฟรมเดียว)
    release_s: float = 0.12           # ลืมตา/หยุดยิ้มต่อเนื่องเท่านี้ถึงปล่อยการตรึงเคอร์เซอร์
    event_cooldown_s: float = 0.35
    half_open_abort_s: float = 0.40   # ค้างระหว่าง close–open นานเท่านี้ = ไม่ใช่การหลับตา → ยกเลิก
    episode_timeout_s: float = 4.0    # หลับตานานกว่านี้ = ไม่ใช่ท่าทาง → เรียนค่าตาเปิดใหม่

    # ── ค่าอ้างอิงตาเปิดแบบปรับตัว (ตามท่าศีรษะ/แสงที่เปลี่ยน) ──
    ref_window_s: float = 1.5
    ref_percentile: float = 0.60
    ref_min_samples: int = 8


@dataclass
class ScrollParams:
    """พารามิเตอร์โหมด scroll (หน่วยสัญญาณ = ระยะปลายจมูกจากกึ่งกลางหน้า ÷ ความสูงหน้า)"""
    deadzone: float = 0.020           # ก้ม/เงยน้อยกว่านี้จากจุดเริ่ม = ไม่เลื่อน
    full_speed: float = 0.065         # ก้ม/เงยถึงระดับนี้ = เลื่อนเร็วสุด
    max_notch_per_s: float = 14.0
    speed_gamma: float = 1.5          # >1 = คุมละเอียดช่วงต้น เร่งช่วงปลาย
    smoothing: float = 0.30           # EMA ของสัญญาณ (0–1, มาก = ไว)
    settle_s: float = 0.30            # เก็บค่าหลังเลิกยิ้มนานเท่านี้ก่อนตั้งจุดอ้างอิง
    anchor_follow: float = 0.01       # จุดอ้างอิงค่อยๆ ตามท่านั่งขณะอยู่ใน deadzone (ต่อเฟรม)
    idle_exit_s: float = 10.0         # อยู่ใน deadzone นานเท่านี้ → ออกโหมด scroll เอง (0 = ปิด)
    invert: bool = False              # True = ก้ม → เลื่อนขึ้น


def _apply(obj, values):
    names = {f.name for f in fields(obj)}
    for k, v in (values or {}).items():
        if k in names:
            setattr(obj, k, type(getattr(obj, k))(v))
    return obj


def load_tuning(path=TUNING_PATH):
    """โหลดค่าที่จูนแล้ว (ไม่มีไฟล์ = ใช้ค่า default)

    Returns:
        tuple: (GestureParams, ScrollParams)
    """
    gp, sp = GestureParams(), ScrollParams()
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            _apply(gp, data.get("gesture"))
            _apply(sp, data.get("scroll"))
        except Exception as e:
            print(f"  ⚠️ อ่าน {path} ไม่ได้ ({e}) — ใช้ค่า default")
    return gp, sp


def save_tuning(gesture_params, scroll_params, path=TUNING_PATH, meta=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {"gesture": asdict(gesture_params), "scroll": asdict(scroll_params)}
    if meta:
        data["meta"] = meta
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path
