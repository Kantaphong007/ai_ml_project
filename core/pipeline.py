"""
EYE ORDER COME AI — Real-time Processing Pipeline
Main processing loop ที่รวม camera → face mesh → feature extraction
→ cursor prediction → gesture detection → mouse control
"""
import time
import threading
from collections import deque
import numpy as np
import cv2
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.camera import CameraStream
from core.face_mesh import FaceMeshDetector
from core.feature_extractor import FeatureExtractor
from core.cursor_predictor import CursorPredictor
from core.click_classifier import ClickClassifier
from core.gesture_detector import GestureDetector
from core.mouse_controller import MouseController
from config.settings import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, FPS, MIRROR_CAMERA,
    BASELINE_DATA_PATH, SCREEN_WIDTH, SCREEN_HEIGHT
)

# ตำแหน่งเคอร์เซอร์ที่ "ก่อนเริ่มทำท่า" — ย้อนกลับกี่เฟรมตอนเริ่มตรึง
# (ตายังปิดไม่ถึง threshold ตอนที่หัว/ใบหน้าเริ่มขยับไปแล้ว 2–3 เฟรม)
_REWIND_FRAMES = 3
# เฟรมที่ไม่เจอหน้าติดกันเกินนี้ (วินาที) ให้ล้างตัวกรอง กันเคอร์เซอร์พุ่งจากค่าเก่า
_FACE_LOST_RESET_S = 0.5
# โหมด scroll: ระยะ (พิกเซลจอ) ที่ต้องก้ม/เงยจากจุดเริ่มก่อนเลื่อน และความเร็ว
_SCROLL_DEADZONE_PX = 80
_SCROLL_FULL_SPEED_PX = 400
_SCROLL_MAX_NOTCH_PER_S = 12
_VK_PAUSE = 0x13


class Pipeline:
    """Real-time pipeline สำหรับควบคุมเมาส์ด้วยท่าทางศีรษะและใบหน้า

    Camera → Face Mesh → Feature Extractor →
    [Cursor Predictor + Gesture Detector] → Mouse Controller

    การคลิกแม่นตำแหน่งเพราะ:
      ทันทีที่เริ่มหลับตา/ยิ้ม ระบบ "ตรึง" เคอร์เซอร์ที่ตำแหน่งก่อนทำท่า
      แล้วคลิกตรงจุดนั้น (ไม่ใช่ตำแหน่งที่ใบหน้าเบี้ยวไปตอนขยิบตา)

    Example:
        >>> pipe = Pipeline()
        >>> if pipe.initialize():
        ...     pipe.start()
        ...     pipe.stop()
    """

    def __init__(self):
        self.camera = CameraStream(CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, FPS, mirror=MIRROR_CAMERA)
        self.detector = FaceMeshDetector()
        self.extractor = FeatureExtractor(self.detector)
        self.cursor_predictor = CursorPredictor()
        # click_classifier (ML) ยังโหลดไว้ใช้ประเมิน/รายงาน แต่ไม่อยู่ในเส้นทางสั่งคลิกสด
        # เพราะข้อมูลเทรนเดิมเป็นข้อมูลสังเคราะห์ ไม่ตรงกับสัญญาณตาจริง
        self.click_classifier = ClickClassifier()
        self.gesture_detector = GestureDetector()
        self.mouse_controller = MouseController()

        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._paused = False
        self._pause_key_down = False

        # สถานะปัจจุบัน (สำหรับ UI overlay)
        self._status = {
            "fps": 0,
            "face_detected": False,
            "ear_l": 0.0,
            "ear_r": 0.0,
            "delta_smile": 0.0,
            "cursor_x": 0,
            "cursor_y": 0,
            "last_action": "",
            "last_action_time": 0,
            "mode": "normal",  # normal, drag, scroll
            "confidence": 0.0,
            "locked": False,   # True = เคอร์เซอร์ถูกตรึงระหว่างทำท่าทาง
            "paused": False,
        }

        # Callback สำหรับ UI
        self.on_status_update = None    # callback(status_dict)
        self.on_frame_update = None     # callback(frame)

    # ──────────────────────────────────────────────
    # Setup
    # ──────────────────────────────────────────────
    def initialize(self):
        """เตรียมระบบ: โหลดโมเดล cursor + ค่าฐานของตา/ปาก

        Returns:
            bool: สำเร็จหรือไม่ (ต้องมีโมเดล cursor)
        """
        print("  🔄 กำลังเตรียมระบบ...")

        ref_l = ref_r = 0.30
        if os.path.exists(BASELINE_DATA_PATH):
            with open(BASELINE_DATA_PATH, 'r') as f:
                baseline = json.load(f)
            self.extractor.set_baseline_mouth_width(baseline["baseline_mouth_width"])
            ref_l = baseline.get("baseline_ear_l", ref_l)
            ref_r = baseline.get("baseline_ear_r", ref_r)
            print(f"  ✅ โหลดค่าฐาน: mouth={baseline['baseline_mouth_width']:.4f} "
                  f"EAR_L={ref_l:.3f} EAR_R={ref_r:.3f}")
        else:
            print("  ⚠️ ไม่พบค่าฐาน — ใช้ค่าเริ่มต้น (ยิ้มเพื่อ Scroll อาจไม่แม่นยำ กรุณากด Baseline)")
        self.gesture_detector.set_reference(ref_l, ref_r)

        cursor_ok = self.cursor_predictor.load()
        if not cursor_ok:
            print("  ❌ ไม่สามารถโหลดโมเดล cursor กรุณา Calibrate + Train ก่อน")
            return False

        # ML click model: ไม่บังคับ (เส้นทางคลิกสดใช้ GestureDetector)
        self.click_classifier.load()
        return True

    def start(self):
        """เริ่ม pipeline"""
        if self._running:
            return

        self._reset_runtime_state()
        self.camera.start()
        self.mouse_controller.enable()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("  ▶ Pipeline เริ่มทำงาน (กดปุ่ม Pause/Break เพื่อหยุด/เริ่มชั่วคราว)")

    def stop(self):
        """หยุด pipeline"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        self.mouse_controller.disable()
        self.camera.stop()
        print("  ⏸ Pipeline หยุดทำงาน")

    def _reset_runtime_state(self):
        self._pos_history = deque(maxlen=_REWIND_FRAMES + 1)
        self._locked_pos = None
        self._scroll_anchor_y = None
        self._scroll_accum = 0.0
        self._last_scroll_t = None
        self._last_face_t = time.perf_counter()
        self._last_frame_id = -1
        self._status["mode"] = "normal"
        self._status["locked"] = False
        self.gesture_detector.reset()
        self.cursor_predictor.reset_smoothing()
        self.extractor.reset_cursor_history()

    # ──────────────────────────────────────────────
    # Main loop
    # ──────────────────────────────────────────────
    def _loop(self):
        """Main processing loop"""
        frame_count = 0
        fps_start = time.time()
        swap_eyes = not self.camera.mirror   # ไม่ mirror → EAR_L คือตาขวาจริงของผู้ใช้

        while self._running:
            self._poll_pause_key()

            frame_id, frame = self.camera.read_with_id()
            if frame is None or frame_id == self._last_frame_id:
                time.sleep(0.003)
                continue
            self._last_frame_id = frame_id
            now = time.perf_counter()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = self.detector.detect(rgb)

            with self._lock:
                if landmarks is None:
                    self._on_face_lost(now)
                else:
                    self._status["face_detected"] = True
                    self._last_face_t = now
                    self._process_landmarks(landmarks, now, swap_eyes)

            if self.on_frame_update:
                self.on_frame_update(frame)

            frame_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                self._status["fps"] = frame_count / elapsed
                frame_count = 0
                fps_start = time.time()

            if self.on_status_update:
                self.on_status_update(self._status.copy())

    def _on_face_lost(self, now):
        self._status["face_detected"] = False
        if now - self._last_face_t > _FACE_LOST_RESET_S:
            # หน้าหายนาน: ล้างทุกอย่าง และปล่อยเมาส์ที่ค้างอยู่ กันค้างโหมด drag
            self.gesture_detector.reset()
            self.cursor_predictor.reset_smoothing()
            self.extractor.reset_cursor_history()
            self._pos_history.clear()
            self._locked_pos = None
            self._status["locked"] = False
            if self.mouse_controller.is_dragging:
                self.mouse_controller.mouse_up()
                self._status["mode"] = "normal"
                self._status["last_action"] = "Drag End (face lost)"

    def _process_landmarks(self, landmarks, now, swap_eyes):
        fw, fh = self.camera.get_frame_size()

        # ── Gesture (ตา/ปาก) ก่อน เพื่อรู้ว่าต้องตรึงเคอร์เซอร์เฟรมนี้หรือไม่ ──
        click_features = self.extractor.extract_click_features(landmarks)
        events = []
        if click_features is not None:
            ear_l, ear_r, delta_smile = (float(v) for v in click_features)
            self._status["ear_l"], self._status["ear_r"] = ear_l, ear_r
            self._status["delta_smile"] = delta_smile
            if swap_eyes:
                ear_l, ear_r = ear_r, ear_l
            events = self.gesture_detector.update(now, ear_l, ear_r, delta_smile)
        freeze = self.gesture_detector.freeze_cursor

        # ── Cursor ──
        cursor_features = self.extractor.extract_cursor_features(landmarks, fw, fh)
        pos = None
        if cursor_features is not None and self.cursor_predictor.is_loaded():
            pos = self.cursor_predictor.predict(cursor_features)

        in_scroll = self._status["mode"] == "scroll"

        if pos is not None:
            if freeze and self._locked_pos is None:
                # เพิ่งเริ่มทำท่า → ย้อนกลับไปตำแหน่งก่อนใบหน้าเริ่มเปลี่ยน แล้วล็อกไว้
                self._locked_pos = self._pos_history[0] if self._pos_history else pos
                if not in_scroll and not self._paused:
                    self.mouse_controller.move_to(*self._locked_pos)
            elif not freeze and self._locked_pos is not None:
                self._locked_pos = None
                # ให้ตัวกรองเริ่มจากตำแหน่งใหม่ ไม่ลากเคอร์เซอร์จากค่าที่เพี้ยนตอนทำท่า
                self.cursor_predictor.reset_smoothing()
                self._pos_history.clear()

            if self._locked_pos is None:
                self._pos_history.append(pos)
                if not in_scroll and not self._paused:
                    self.mouse_controller.move_to(*pos)
                    self._status["cursor_x"], self._status["cursor_y"] = int(pos[0]), int(pos[1])
            else:
                self._status["cursor_x"], self._status["cursor_y"] = (int(self._locked_pos[0]),
                                                                      int(self._locked_pos[1]))

            if in_scroll:
                self._update_scroll(now, pos)

        self._status["locked"] = self._locked_pos is not None

        # ── Execute events ──
        for ev in events:
            self._execute_action(ev, now)

    # ──────────────────────────────────────────────
    # Actions
    # ──────────────────────────────────────────────
    def _click_pos(self):
        """ตำแหน่งที่ควรคลิก: จุดที่ล็อกไว้ก่อนทำท่า (ถ้าไม่มีใช้ตำแหน่งล่าสุด)"""
        if self._locked_pos is not None:
            return self._locked_pos
        if self._pos_history:
            return self._pos_history[-1]
        return None

    def _execute_action(self, event, now):
        """สั่งการตาม event จาก GestureDetector"""
        if self._paused:
            return

        mc = self.mouse_controller
        x, y = self._click_pos() or (None, None)

        if event == "left_click":
            mc.left_click(x, y)
            self._status["last_action"] = "Left Click"
        elif event == "right_click":
            mc.right_click(x, y)
            self._status["last_action"] = "Right Click"
        elif event == "double_click":
            mc.double_click(x, y)
            self._status["last_action"] = "Double Click"
        elif event == "drag_toggle":
            if mc.is_dragging:
                mc.mouse_up()
                self._status["last_action"] = "Drag End"
                self._status["mode"] = "normal"
            else:
                mc.mouse_down(x, y)
                self._status["last_action"] = "Drag Start"
                self._status["mode"] = "drag"
        elif event == "scroll_toggle":
            if self._status["mode"] == "scroll":
                self._status["mode"] = "normal"
                self._status["last_action"] = "Scroll End"
                mc.is_scrolling = False
                self._scroll_anchor_y = None
                self.cursor_predictor.reset_smoothing()
                self._pos_history.clear()
            else:
                if mc.is_dragging:
                    mc.mouse_up()
                self._status["mode"] = "scroll"
                self._status["last_action"] = "Scroll Mode"
                mc.is_scrolling = True
                self._scroll_anchor_y = None   # ตั้งจุดอ้างอิงจากเฟรมถัดไป
                self._scroll_accum = 0.0
                self._last_scroll_t = None
        else:
            return

        self._status["last_action_time"] = now

    def _update_scroll(self, now, pos):
        """โหมด scroll: ก้มหน้า = เลื่อนลง, เงยหน้า = เลื่อนขึ้น (ความเร็วตามระยะที่ก้ม/เงย)"""
        if self._locked_pos is not None or self._paused:
            self._last_scroll_t = None
            return
        if self._scroll_anchor_y is None:
            self._scroll_anchor_y = pos[1]
            self._last_scroll_t = now
            return

        dt = now - (self._last_scroll_t or now)
        self._last_scroll_t = now

        dy = pos[1] - self._scroll_anchor_y          # + = ก้ม (ตำแหน่งต่ำลงจอ)
        mag = abs(dy)
        if mag < _SCROLL_DEADZONE_PX:
            self._scroll_accum = 0.0
            return
        speed = min(1.0, (mag - _SCROLL_DEADZONE_PX) / (_SCROLL_FULL_SPEED_PX - _SCROLL_DEADZONE_PX))
        notches = speed * _SCROLL_MAX_NOTCH_PER_S * dt * (-1.0 if dy > 0 else 1.0)
        self._scroll_accum += notches
        whole = int(self._scroll_accum)
        if whole != 0:
            self._scroll_accum -= whole
            self.mouse_controller.scroll(whole)

    # ──────────────────────────────────────────────
    # Emergency pause (Pause/Break key)
    # ──────────────────────────────────────────────
    def _poll_pause_key(self):
        if sys.platform != 'win32':
            return
        import ctypes
        down = bool(ctypes.windll.user32.GetAsyncKeyState(_VK_PAUSE) & 0x8000)
        if down and not self._pause_key_down:
            self._paused = not self._paused
            self._status["paused"] = self._paused
            self._status["last_action"] = "Paused" if self._paused else "Resumed"
            if self._paused and self.mouse_controller.is_dragging:
                self.mouse_controller.mouse_up()
                self._status["mode"] = "normal"
            self.cursor_predictor.reset_smoothing()
        self._pause_key_down = down

    def get_status(self):
        """ดึงสถานะปัจจุบัน (thread-safe)"""
        with self._lock:
            return self._status.copy()

    @property
    def is_running(self):
        return self._running
