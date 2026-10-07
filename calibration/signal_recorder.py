"""
EYE ORDER COME AI — Signal Recorder (เก็บข้อมูลสำหรับจูนการตรวจจับท่าทาง)

บันทึกสัญญาณดิบ "ทุกเฟรม" ต่อเนื่องทั้ง session พร้อม label ว่าตอนนั้นสั่งให้ทำท่าอะไร
แล้วใช้ training/tune_gestures.py replay ผ่าน GestureDetector ตัวจริงเพื่อหาค่าที่ดีที่สุด

ต่างจาก gesture_collection.py (เก็บหน้าต่าง 30 เฟรมสำหรับโมเดล ML):
  ไฟล์นี้เก็บสัญญาณต่อเนื่องจริง รวมช่วงพักระหว่างท่า → วัดได้ทั้ง "จับถูกไหม" และ "จับผิดตอนไม่ได้ทำท่าไหม"

ผลลัพธ์: data/gesture_signals.csv (ต่อท้ายทุกครั้ง, แยก session ด้วยคอลัมน์ session)
คอลัมน์: session, trial, label, phase(gap/go), t, ear_l, ear_r, delta_smile, mouth_ratio, pitch
  ear_l / ear_r เป็นตาซ้าย/ขวา "ของผู้ใช้จริง" (สลับตาม mirror แล้ว) = ค่าที่ป้อนเข้า GestureDetector

Usage:
    python calibration/signal_recorder.py               # ทำครบทุกท่า
    python calibration/signal_recorder.py --reps 4      # ท่าละ 4 ครั้ง (เร็วขึ้น)
    python calibration/signal_recorder.py --only wink_left wink_right
"""
import argparse
import json
import os
import random
import sys
import time

import cv2
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.camera import CameraStream
from core.face_mesh import FaceMeshDetector
from core.feature_extractor import FeatureExtractor, EAR_VERSION
from config.settings import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, FPS,
    BASELINE_DATA_PATH, DATA_DIR, load_user_settings,
)

SIGNALS_PATH = os.path.join(DATA_DIR, "gesture_signals.csv")
GAP_S = 1.6   # ช่วงพักก่อนแต่ละท่า (ใช้เป็นช่วงวัด false positive + ค่านิ่งก่อนก้ม/เงย)

# label → (คำสั่งบนจอ, คำอธิบายไทยในคอนโซล, ความยาวหน้าต่าง go วินาที, จำนวนครั้ง default)
PROTOCOL = {
    "idle":         ("Use the computer normally: move head to all screen corners, blink naturally",
                     "ใช้งานปกติ: หันหัวไปทุกมุมจอ กะพริบตาตามธรรมชาติ (ห้ามทำท่าคลิก)", 20.0, 1),
    "blink":        ("Blink ONCE naturally", "กะพริบตาตามปกติ 1 ครั้ง", 2.0, 8),
    "wink_left":    ("Wink LEFT eye quickly (~0.2-0.4s)", "ขยิบตาซ้ายสั้นๆ ~0.2–0.4 วิ แล้วลืมตา", 2.5, 8),
    "wink_right":   ("Wink RIGHT eye quickly (~0.2-0.4s)", "ขยิบตาขวาสั้นๆ ~0.2–0.4 วิ แล้วลืมตา", 2.5, 8),
    "double_blink": ("Blink firmly TWICE quickly", "กะพริบตาแน่นๆ 2 ครั้งติดกัน", 2.5, 8),
    "drag":         ("Close ONE eye and hold ~1.5s, then open", "หลับตาข้างเดียวค้าง ~1.5 วิ แล้วลืมตา", 3.2, 5),
    "smile":        ("Smile wide, hold ~1s, then relax", "ยิ้มกว้างค้าง ~1 วิ แล้วหยุดยิ้ม", 3.0, 6),
    "nod_down":     ("Tilt head DOWN (as if scrolling down), hold 1s, come back",
                     "ก้มหน้า (เหมือนจะเลื่อนลง) ค้าง 1 วิ แล้วกลับท่าปกติ", 3.0, 5),
    "nod_up":       ("Tilt head UP (as if scrolling up), hold 1s, come back",
                     "เงยหน้า (เหมือนจะเลื่อนขึ้น) ค้าง 1 วิ แล้วกลับท่าปกติ", 3.0, 5),
}


class SignalRecorder:
    def __init__(self):
        settings = load_user_settings()
        self.mirror = bool(settings.get("mirror", True))
        self.camera = CameraStream(settings.get("camera", CAMERA_INDEX),
                                   FRAME_WIDTH, FRAME_HEIGHT, FPS, mirror=self.mirror)
        self.detector = FaceMeshDetector()
        self.extractor = FeatureExtractor(self.detector, (FRAME_WIDTH, FRAME_HEIGHT))
        self.rows = []
        self.session = time.strftime("%Y%m%d-%H%M%S")
        self._t0 = None
        self._last_id = -1

    def _load_baseline(self):
        if os.path.exists(BASELINE_DATA_PATH):
            with open(BASELINE_DATA_PATH, "r") as f:
                b = json.load(f)
            if b.get("ear_version") == EAR_VERSION:
                self.extractor.set_baseline_mouth_ratio(b.get("baseline_mouth_ratio"),
                                                        b.get("baseline_pitch"))
                return
        print("  ⚠️ ไม่มีค่าฐานเวอร์ชันปัจจุบัน — ค่าปากจะเรียนจากกล้องเอง (แนะนำกด Baseline ก่อน)")

    # ──────────────────────────────────────────────
    def _step(self, trial, label, phase, banner, sub, progress, color):
        """อ่าน 1 เฟรม → บันทึก (ถ้าเจอหน้า) → แสดงผล; คืน False ถ้ากด ESC"""
        frame_id, frame = self.camera.read_with_id()
        if frame is None or frame_id == self._last_id:
            if cv2.waitKey(1) & 0xFF == 27:
                return False
            time.sleep(0.003)
            return True
        self._last_id = frame_id
        t = time.perf_counter() - self._t0

        lm = self.detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        face = lm is not None
        if face:
            feat = self.extractor.extract_click_features(lm)
            pitch = self.extractor.extract_head_pitch_signal(lm)
            if feat is not None and pitch is not None:
                ear_l, ear_r, ds = (float(v) for v in feat)
                if not self.mirror:
                    ear_l, ear_r = ear_r, ear_l
                self.rows.append({
                    "session": self.session, "trial": trial, "label": label, "phase": phase,
                    "t": round(t, 4), "ear_l": ear_l, "ear_r": ear_r, "delta_smile": ds,
                    "mouth_ratio": self.extractor.compute_mouth_ratio(lm) or 0.0, "pitch": pitch,
                })

        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 95), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.putText(frame, banner, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        cv2.putText(frame, sub, (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
        cv2.putText(frame, "Face OK" if face else "NO FACE", (12, 86),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0) if face else (0, 0, 255), 1)
        cv2.rectangle(frame, (20, h - 30), (20 + int((w - 40) * min(progress, 1.0)), h - 15), color, -1)
        cv2.imshow("Signal Recorder", frame)
        return (cv2.waitKey(1) & 0xFF) != 27

    def _segment(self, trial, label, phase, dur, banner, sub, color):
        start = time.perf_counter()
        while (el := time.perf_counter() - start) < dur:
            if not self._step(trial, label, phase, banner, sub, el / dur, color):
                return False
        return True

    def run(self, reps=None, only=None):
        labels = [k for k in PROTOCOL if not only or k in only]
        plan = [k for k in labels if k != "idle" for _ in range(reps or PROTOCOL[k][3])]
        random.shuffle(plan)
        if "idle" in labels:
            plan = ["idle"] + plan

        print("=" * 60)
        print("   SIGNAL RECORDER — เก็บข้อมูลจูนการตรวจจับท่าทาง")
        print(f"   {len(plan)} ท่า (~{sum(PROTOCOL[k][2] + GAP_S for k in plan) / 60:.1f} นาที)")
        print("   ทำท่า 'เฉพาะตอนแถบเป็นสีเขียว (GO)' ช่วงพักให้นั่งปกติ มองจอ")
        print("   SPACE = เริ่ม, ESC = หยุด (ข้อมูลที่เก็บแล้วยังถูกบันทึก)")
        print("=" * 60)

        self._load_baseline()
        self.camera.start()
        self.extractor.set_frame_size(*self.camera.get_frame_size())
        try:
            while True:
                frame = self.camera.read()
                if frame is not None:
                    cv2.putText(frame, "Press SPACE to start, ESC to quit", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.imshow("Signal Recorder", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    break
                if key == 27:
                    return None

            self._t0 = time.perf_counter()
            for i, label in enumerate(plan, 1):
                text, thai, go_s, _ = PROTOCOL[label]
                print(f"  [{i}/{len(plan)}] {label}: {thai}")
                if not self._segment(i, label, "gap", GAP_S, f"[{i}/{len(plan)}] NEXT: {label}",
                                     text, (0, 200, 255)):
                    break
                if not self._segment(i, label, "go", go_s, f"GO! {label}", text, (0, 220, 0)):
                    break
        finally:
            self.camera.stop()
            cv2.destroyAllWindows()
        return self._save()

    def _save(self):
        if not self.rows:
            print("  ⚠️ ไม่มีข้อมูล")
            return None
        df = pd.DataFrame(self.rows)
        header = not os.path.exists(SIGNALS_PATH)
        df.to_csv(SIGNALS_PATH, mode="a", header=header, index=False)
        n_trials = df[df.phase == "go"].groupby("trial").ngroups
        fps = len(df) / max(df.t.max() - df.t.min(), 1e-6)
        print(f"\n  ✅ บันทึก {len(df)} เฟรม / {n_trials} ท่า (~{fps:.0f} FPS) → {SIGNALS_PATH}")
        print("  ขั้นต่อไป: python training/tune_gestures.py")
        return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="บันทึกสัญญาณตา/ปาก/ก้มเงย สำหรับจูนค่า")
    ap.add_argument("--reps", type=int, default=None, help="จำนวนครั้งต่อท่า (default ตาม PROTOCOL)")
    ap.add_argument("--only", nargs="+", choices=list(PROTOCOL), default=None)
    args = ap.parse_args()
    SignalRecorder().run(reps=args.reps, only=args.only)
