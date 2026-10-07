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
from core.feature_extractor import FeatureExtractor, EAR_VERSION
from core.cursor_predictor import CursorPredictor
from core.click_classifier import ClickClassifier
from core.gesture_detector import GestureDetector
from core.head_pointer import HeadPointer
from core.mouse_controller import MouseController, has_input_permission
from config.settings import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, FPS, MIRROR_CAMERA,
    BASELINE_DATA_PATH, SCREEN_WIDTH, SCREEN_HEIGHT
)
from config.tuning import load_tuning, TUNING_PATH

# ตำแหน่งเคอร์เซอร์ที่ "ก่อนเริ่มทำท่า" — ย้อนกลับกี่เฟรมตอนเริ่มตรึง
# (ตายังปิดไม่ถึง threshold ตอนที่หัว/ใบหน้าเริ่มขยับไปแล้ว 2–3 เฟรม)
_REWIND_FRAMES = 3
# เฟรมที่ไม่เจอหน้าติดกันเกินนี้ (วินาที) ให้ล้างตัวกรอง กันเคอร์เซอร์พุ่งจากค่าเก่า
_FACE_LOST_RESET_S = 0.5
_VK_PAUSE = 0x13
# event จากตาที่ไม่ทำงานในโหมด scroll (ก้ม/เงยทำให้ EAR เปลี่ยน เสี่ยงคลิกโดยไม่ตั้งใจ)
_EYE_EVENTS = ("left_click", "right_click", "double_click", "drag_start")
# โหมดคุมเคอร์เซอร์: hybrid = relative + ML ช่วยตอนหันเร็ว, relative = ไม่ใช้โมเดล, absolute = ML ล้วน
CURSOR_MODES = ("hybrid", "relative", "absolute")


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
        self.extractor = FeatureExtractor(self.detector, (FRAME_WIDTH, FRAME_HEIGHT))
        self.gesture_params, self.scroll_params = load_tuning()
        self.cursor_predictor = CursorPredictor()
        self.pointer = HeadPointer(SCREEN_WIDTH, SCREEN_HEIGHT)
        self.cursor_mode = "hybrid"
        self._pointer_hold = False
        self._baseline_neutral = None    # มุมหน้าตรงจาก Baseline (yaw, pitch)
        # โมเดล ML จำแนกท่าทางตา (hybrid: detector หา episode → โมเดลตัดสินท่า)
        # ไม่มีโมเดล / โมเดลแพ้ rule-based ตอนประเมิน → ใช้ rule-based แทนอัตโนมัติ
        self.click_classifier = ClickClassifier()
        self.gesture_detector = GestureDetector(params=self.gesture_params)
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
            "click_engine": "Rule-based",
            "locked": False,   # True = เคอร์เซอร์ถูกตรึงระหว่างทำท่าทาง
            "paused": False,
            # สำหรับดู/จูนค่า
            "gesture_state": "idle",
            "decision": "",          # เหตุผลการตัดสินท่าล่าสุด (เช่น ทำไมขยิบแล้วไม่คลิก)
            "ratio_l": 1.0,
            "ratio_r": 1.0,
            "scroll_offset": 0.0,   # ก้ม(+)/เงย(−) เทียบจุดอ้างอิงของโหมด scroll
            "cursor_mode": "hybrid",
            "head_speed": 0.0,      # ความเร็วหมุนหัว (องศา/วินาที) — โหมด relative/hybrid
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

        # ค่าที่จูนจากข้อมูล (โหลดใหม่ทุกครั้งที่เริ่ม → จูนแล้วกด Start ใหม่ได้เลย)
        self.gesture_params, self.scroll_params = load_tuning()
        self.gesture_detector.set_params(self.gesture_params)
        self.extractor.set_smile_pitch_comp(self.gesture_params.smile_pitch_comp)
        if os.path.exists(TUNING_PATH):
            print(f"  ✅ โหลดค่าที่จูนแล้ว: {TUNING_PATH}")

        baseline = None
        if os.path.exists(BASELINE_DATA_PATH):
            with open(BASELINE_DATA_PATH, 'r') as f:
                baseline = json.load(f)
        # มุมหน้าตรงจาก Baseline (ต้องอัดด้วย mirror เดียวกัน ไม่งั้นทิศซ้าย-ขวากลับด้าน)
        self._baseline_neutral = None
        if (baseline and baseline.get("head_yaw") is not None
                and baseline.get("mirror", True) == self.camera.mirror):
            self._baseline_neutral = (baseline["head_yaw"], baseline["head_pitch"])
            print(f"  ✅ มุมหน้าตรงจาก Baseline: yaw {baseline['head_yaw']:+.1f}° "
                  f"pitch {baseline['head_pitch']:+.1f}°")
        else:
            print("  ℹ️ Baseline ยังไม่มีมุมหน้าตรง — ระบบจะใช้ท่าช่วงแรกหลังกด Start แทน "
                  "(กด Baseline ใหม่จะแม่นกว่า)")

        if baseline and baseline.get("ear_version") == EAR_VERSION:
            ref_l, ref_r = baseline["baseline_ear_l"], baseline["baseline_ear_r"]
            if not self.camera.mirror:      # detector รับ EAR ตามตาจริงของผู้ใช้ (ดู _loop)
                ref_l, ref_r = ref_r, ref_l
            self.extractor.set_baseline_mouth_ratio(baseline.get("baseline_mouth_ratio"),
                                                    baseline.get("baseline_pitch"))
            self.gesture_detector.set_reference(ref_l, ref_r, trusted=True)
            print(f"  ✅ โหลดค่าฐาน: mouth_ratio={baseline.get('baseline_mouth_ratio', 0):.4f} "
                  f"EAR_L={ref_l:.3f} EAR_R={ref_r:.3f}")
        else:
            # ไม่มีค่าฐาน หรือเป็นค่าฐานเวอร์ชันเก่า (คนละสเกล) → เรียนจากกล้องเองใน ~0.5 วิแรก
            self.extractor.set_baseline_mouth_ratio(None)
            self.gesture_detector.set_reference(0.28, 0.28, trusted=False)
            print("  ⚠️ ไม่พบค่าฐานเวอร์ชันปัจจุบัน — ระบบจะเรียนค่าตา/ปากจากกล้องเอง "
                  "(แนะนำกด Baseline ใหม่ 1 ครั้ง)")

        cursor_ok = self.cursor_predictor.load()
        effective = self.cursor_mode
        if self.cursor_mode == "absolute" and not cursor_ok:
            print("  ❌ โหมด absolute ต้องมีโมเดล cursor — กด Calibrate + Train Cursor "
                  "(หรือเปลี่ยนเป็นโหมด relative ใน Settings)")
            return False
        if self.cursor_mode == "hybrid" and not cursor_ok:
            effective = "relative"
            print("  ℹ️ โหมด hybrid ยังไม่มีโมเดล cursor → ทำงานแบบ relative "
                  "(Calibrate + Train Cursor เพื่อเปิดตัวช่วย ML)")
        self._status["cursor_mode"] = effective
        print(f"  🖱️ โหมดเคอร์เซอร์: {effective}")

        # ML click model: ไม่บังคับ (ไม่มี → rule-based)
        self.gesture_detector.set_classifier(None)
        self._status["click_engine"] = "Rule-based"
        clf = self.click_classifier
        if clf.load() and clf.use_live:
            stale = clf.segmentation_mismatch(self.gesture_params)
            if stale:
                print(f"  ⚠️ จูนค่า {stale} หลังเทรนโมเดล click — ควรกด Train Click ใหม่")
            self.gesture_detector.set_classifier(clf)
            self._status["click_engine"] = f"ML: {clf.name}"
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
        if not has_input_permission():
            print("  ⚠️ macOS ยังไม่อนุญาต Accessibility → เมาส์จะไม่ขยับ/คลิกไม่ได้\n"
                  "     System Settings → Privacy & Security → Accessibility → เปิดให้แอปที่ใช้รัน แล้วเปิดแอปใหม่")

    def stop(self):
        """หยุด pipeline"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        self.mouse_controller.disable()
        self.camera.stop()
        print("  ⏸ Pipeline หยุดทำงาน")

    def set_cursor_mode(self, mode):
        self.cursor_mode = mode if mode in CURSOR_MODES else "hybrid"

    def _reset_runtime_state(self):
        # relative/hybrid เริ่มจากตำแหน่งเมาส์ปัจจุบัน
        try:
            self.pointer.set_position(*self.mouse_controller.get_current_position())
        except Exception:
            self.pointer.set_position(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)
        self.pointer.resync()
        if self._baseline_neutral is not None:
            self.pointer.set_neutral(*self._baseline_neutral)   # หน้าตรงตอน Baseline = กลางจอ
        else:
            self.pointer.reset_neutral()                         # ไม่มี → เรียนจากช่วงแรกหลัง Start
        self._pointer_hold = False
        self._pos_history = deque(maxlen=_REWIND_FRAMES + 1)
        self._locked_pos = None
        self._gesture_pos = None         # relative/hybrid: ตำแหน่งตอนเริ่มทำท่า (ไว้คลิก)
        self._scroll_pos = None
        self._reset_scroll_tracking()
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
            self.pointer.resync()
            self.extractor.reset_cursor_history()
            self._pos_history.clear()
            self._locked_pos = None
            self._status["locked"] = False
            self._reset_scroll_tracking()   # กลับมาแล้วตั้งจุดอ้างอิงใหม่ (ท่านั่งอาจเปลี่ยน)
            if self._status["mode"] == "scroll":     # detector ถูก reset → ยิ้มต้องเริ่มใหม่
                self._exit_scroll_mode("Scroll End (face lost)")
            if self.mouse_controller.is_dragging:
                self.mouse_controller.mouse_up()
                self._status["mode"] = "normal"
                self._status["last_action"] = "Drag End (face lost)"

    def _process_landmarks(self, landmarks, now, swap_eyes):
        fw, fh = self.camera.get_frame_size()
        self.extractor.set_frame_size(fw, fh)

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
        gd = self.gesture_detector
        self._status["gesture_state"] = gd.state
        self._status["decision"] = gd.last_decision
        self._status["ratio_l"], self._status["ratio_r"] = gd.ratio_l, gd.ratio_r
        freeze = gd.freeze_cursor

        # ── Cursor ──
        cursor_features = self.extractor.extract_cursor_features(landmarks)
        in_scroll = self._status["mode"] == "scroll"
        relative = self._status["cursor_mode"] != "absolute"
        # relative/hybrid: มุมหัวจาก matrix 3 มิติแทบไม่เปลี่ยนตอนหลับตา → ไม่ต้องหยุดเคอร์เซอร์ตอนขยิบ/กะพริบ
        pos = self._cursor_position(now, cursor_features,
                                    hold=in_scroll if relative else (freeze or in_scroll))

        if in_scroll:
            self._update_scroll(now, self.extractor.extract_head_pitch_signal(landmarks))

        if pos is not None and in_scroll:
            # โหมด scroll: เคอร์เซอร์อยู่นิ่งที่จุดเข้าโหมด (wheel event ลงหน้าต่างนั้น)
            self._pos_history.append(pos)
        elif pos is not None and relative:
            # ไม่หยุด/ไม่ย้อนเคอร์เซอร์ (เดิมย้อนไป 3 เฟรมก่อนทุกครั้งที่ตาเริ่มปิด รวมการกะพริบปกติ
            # → เห็นเคอร์เซอร์ "ยึกถอยหลัง" ระหว่างขยับ) — แค่จำตำแหน่งตอนเริ่มทำท่าไว้คลิกให้ตรง
            if freeze and self._gesture_pos is None:
                self._gesture_pos = self._pos_history[-1] if self._pos_history else pos
            elif not freeze:
                self._gesture_pos = None
            self._locked_pos = None
            self._pos_history.append(pos)
            if not self._paused:
                self.mouse_controller.move_to(*pos)
                self._status["cursor_x"], self._status["cursor_y"] = int(pos[0]), int(pos[1])
        elif pos is not None:
            if freeze and self._locked_pos is None:
                # เพิ่งเริ่มทำท่า → ย้อนกลับไปตำแหน่งก่อนใบหน้าเริ่มเปลี่ยน แล้วล็อกไว้
                self._locked_pos = self._pos_history[0] if self._pos_history else pos
                self.pointer.set_position(*self._locked_pos)   # relative: ไปต่อจากจุดที่ล็อก
                if not self._paused:
                    self.mouse_controller.move_to(*self._locked_pos)
            elif not freeze and self._locked_pos is not None:
                # ไม่ reset ตัวกรอง: เดิม reset แล้วเคอร์เซอร์ "กระตุก" ไปค่าดิบทุกครั้งที่กะพริบตา
                # ตัวกรองยังจำตำแหน่งก่อนทำท่า (ไม่ได้ป้อนค่าเพี้ยนระหว่างตรึง) จึงไหลต่อได้นุ่มนวล
                self._locked_pos = None
                self._pos_history.clear()

            if self._locked_pos is None:
                self._pos_history.append(pos)
                if not self._paused:
                    self.mouse_controller.move_to(*pos)
                    self._status["cursor_x"], self._status["cursor_y"] = int(pos[0]), int(pos[1])
            else:
                self._status["cursor_x"], self._status["cursor_y"] = (int(self._locked_pos[0]),
                                                                      int(self._locked_pos[1]))

        self._status["locked"] = self._locked_pos is not None or self._gesture_pos is not None

        # ── Execute events ──
        for ev in events:
            if self._status["mode"] == "scroll" and ev in _EYE_EVENTS:
                continue
            self._execute_action(ev, now)

    def _cursor_position(self, now, feats, hold):
        """ตำแหน่งเคอร์เซอร์ของเฟรมนี้ตามโหมด

        hold = กำลังทำท่า/อยู่ในโหมด scroll → ไม่ป้อนมุมหัวช่วงนี้ (ใบหน้าเบี้ยวตอนขยิบ/ยิ้ม)
        ปล่อยแล้ว relative เริ่มนับจากมุมใหม่ เคอร์เซอร์จึงไม่กระโดด
        """
        if feats is None:
            return None
        mode = self._status["cursor_mode"]
        if mode == "absolute":
            if not self.cursor_predictor.is_loaded():
                return None
            # ระหว่างทำท่าไม่ป้อนค่าที่เพี้ยนเข้าตัวกรอง → ปล่อยแล้วกรองต่อได้ลื่น
            return self.cursor_predictor.predict(feats, smooth=not hold)

        if hold:
            self._pointer_hold = True
            return self.pointer.pos
        if self._pointer_hold:
            self._pointer_hold = False
            self.pointer.resync()
        target = None
        if mode == "hybrid" and self.cursor_predictor.is_loaded():
            target = self.cursor_predictor.predict(feats, smooth=False)
        pos = self.pointer.update(now, feats[0], feats[1], target)
        self._status["head_speed"] = self.pointer.velocity
        return pos

    # ──────────────────────────────────────────────
    # Actions
    # ──────────────────────────────────────────────
    def _click_pos(self):
        """ตำแหน่งที่ควรคลิก: จุดก่อนเริ่มทำท่า (ถ้าไม่มีใช้ตำแหน่งล่าสุด)"""
        if self._gesture_pos is not None:
            return self._gesture_pos
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
        elif event == "drag_start":
            # หลับตาค้างครบเวลา → กดเมาส์ที่ตำแหน่งก่อนเริ่มหลับตา แล้วลากตามหัวจนกว่าจะลืมตา
            if not mc.is_dragging:
                mc.mouse_down(x, y)
            self._status["last_action"] = "Drag Start (ลืมตาเพื่อปล่อย)"
            self._status["mode"] = "drag"
        elif event == "drag_end":
            if not mc.is_dragging:
                return
            mc.mouse_up()
            self._status["last_action"] = "Drag End"
            self._status["mode"] = "normal"
        elif event == "scroll_end":
            if self._status["mode"] == "scroll":
                self._exit_scroll_mode("Scroll End")
        elif event == "scroll_start":
            if self._status["mode"] != "scroll":
                if mc.is_dragging:
                    mc.mouse_up()
                self._status["mode"] = "scroll"
                self._status["last_action"] = "Scroll (ยิ้มค้าง + ก้ม/เงย, หุบยิ้มเพื่อออก)"
                mc.is_scrolling = True
                # wheel event ไปที่หน้าต่างใต้เคอร์เซอร์ → ตรึงเคอร์เซอร์ไว้ตลอดโหมด scroll
                self._scroll_pos = (x, y) if x is not None else None
                self._reset_scroll_tracking()
                self._scroll_active_t = now
        else:
            return

        self._status["last_action_time"] = now

    # ──────────────────────────────────────────────
    # Scroll mode
    # ──────────────────────────────────────────────
    def _reset_scroll_tracking(self):
        """ล้างจุดอ้างอิง — จะตั้งใหม่หลังสัญญาณนิ่ง settle_s"""
        self._scroll_anchor = None
        self._scroll_sig = None
        self._scroll_settle = []
        self._scroll_settle_t = None
        self._scroll_accum = 0.0
        self._last_scroll_t = None
        self._scroll_active_t = time.perf_counter()
        self._status["scroll_offset"] = 0.0

    def _exit_scroll_mode(self, label):
        self._status["mode"] = "normal"
        self._status["last_action"] = label
        self.mouse_controller.is_scrolling = False
        self._reset_scroll_tracking()
        # absolute: ยังยิ้มค้างอยู่ตอนสลับโหมด → ตรึงเคอร์เซอร์ไว้ที่เดิมจนเลิกยิ้ม ไม่ให้กระโดด
        # (relative/hybrid ไปต่อจากจุดเดิมเองอยู่แล้ว)
        if self._scroll_pos is not None and self._status["cursor_mode"] == "absolute":
            self._locked_pos = self._scroll_pos
        self._scroll_pos = None
        self.cursor_predictor.reset_smoothing()
        self._pos_history.clear()

    def _update_scroll(self, now, pitch):
        """โหมด scroll (ระหว่างยิ้มค้าง): ก้มหน้า = เลื่อนลง, เงยหน้า = เลื่อนขึ้น
        ความเร็วตามระยะที่ก้ม/เงยจากท่าตอนเริ่มยิ้ม

        ใช้สัญญาณก้ม/เงยจากใบหน้าโดยตรง (ไม่ผ่านโมเดลเคอร์เซอร์ที่ถูก clamp ที่ขอบจอ)
        และไม่หยุดเพราะ EAR เปลี่ยน (ก้มหน้าทำให้ตาดูหรี่ลงเป็นปกติ)
        """
        sp = self.scroll_params
        if pitch is None:
            return
        self._scroll_sig = pitch if self._scroll_sig is None else (
            self._scroll_sig + sp.smoothing * (pitch - self._scroll_sig))
        sig = self._scroll_sig

        if self._paused:
            # หยุดชั่วคราว → ไม่เลื่อน และตั้งจุดอ้างอิงใหม่เมื่อกลับมา
            self._scroll_anchor = None
            self._scroll_settle = []
            self._scroll_settle_t = None
            self._last_scroll_t = None
            self._scroll_active_t = now
            return

        if self._scroll_anchor is None:
            if self._scroll_settle_t is None:
                self._scroll_settle_t = now
            self._scroll_settle.append(sig)
            if now - self._scroll_settle_t >= sp.settle_s:
                self._scroll_anchor = float(np.median(self._scroll_settle))
                self._scroll_settle = []
                self._last_scroll_t = now
                self._scroll_active_t = now
            return

        dt = min(now - (self._last_scroll_t or now), 0.1)
        self._last_scroll_t = now

        d = sig - self._scroll_anchor                 # + = ก้ม
        self._status["scroll_offset"] = d
        mag = abs(d)
        if mag < sp.deadzone:
            self._scroll_accum = 0.0
            # ขณะนิ่ง ค่อยๆ ให้จุดอ้างอิงตามท่านั่งที่เปลี่ยนไป
            self._scroll_anchor += sp.anchor_follow * d
            if sp.idle_exit_s > 0 and now - self._scroll_active_t >= sp.idle_exit_s:
                self._exit_scroll_mode("Scroll End (idle)")
            return

        self._scroll_active_t = now
        span = max(sp.full_speed - sp.deadzone, 1e-6)
        speed = min(1.0, (mag - sp.deadzone) / span) ** sp.speed_gamma
        direction = -1.0 if d > 0 else 1.0           # ก้ม → wheel ลบ = เลื่อนลง
        if sp.invert:
            direction = -direction
        self._scroll_accum += speed * sp.max_notch_per_s * dt * direction
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
            try:   # relative: กลับมาแล้วเริ่มจากตำแหน่งเมาส์จริง (อาจถูกขยับด้วยมือระหว่างพัก)
                self.pointer.set_position(*self.mouse_controller.get_current_position())
            except Exception:
                pass
            self.pointer.resync()
        self._pause_key_down = down

    def get_status(self):
        """ดึงสถานะปัจจุบัน (thread-safe)"""
        with self._lock:
            return self._status.copy()

    @property
    def is_running(self):
        return self._running
