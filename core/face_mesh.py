"""
EYE ORDER COME AI — Face Landmarker (MediaPipe Tasks)
ตรวจจับใบหน้า → 478 landmarks + "ท่าศีรษะ" (yaw / pitch / roll) จาก facial transformation matrix

ทำไมใช้ transformation matrix แทน solvePnP 6 จุดแบบเดิม:
  MediaPipe fit โมเดลใบหน้า 3 มิติทั้งหน้า (468 จุด) เข้ากับภาพ → มุมศีรษะนิ่งกว่ามาก
  solvePnP เดิมใช้แค่ 6 จุด + โมเดลหน้าทั่วไป ค่า pitch กระโดด 88°–174° จนใช้ไม่ได้

ทิศของมุม (ตรวจกับภาพจริงแล้ว — กลับภาพซ้ายขวาแล้ว yaw กลับเครื่องหมาย ตรงกับตำแหน่งปลายจมูก):
  head_yaw   + = จมูกชี้ไปทางขวาของภาพ   (mirror เปิด = ผู้ใช้หันขวา → เคอร์เซอร์ไปขวา)
  head_pitch + = จมูกชี้ลงล่างของภาพ      (ก้มหน้า → เคอร์เซอร์ลง)
  head_roll  + = เอียงศีรษะตามเข็มนาฬิกาในภาพ
"""
import math
import os
import time
import warnings

# ปิดข้อความเตือนที่ไม่เกี่ยวกับการทำงาน (protobuf deprecation / log ของ MediaPipe-TFLite)
warnings.filterwarnings("ignore", message=".*GetPrototype.*")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "face_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")


def ensure_model(path=MODEL_PATH):
    """ดาวน์โหลดโมเดล face_landmarker.task (~4 MB) ถ้ายังไม่มี"""
    if not os.path.exists(path):
        import urllib.request
        print(f"  ⬇️ ดาวน์โหลดโมเดล Face Landmarker → {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


def pose_from_matrix(matrix):
    """4×4 facial transformation matrix → (yaw, pitch, roll) องศา ตามทิศที่อธิบายไว้ด้านบน

    พิกัดของ matrix: แกน y ชี้ขึ้น, z ชี้เข้าหากล้อง (OpenGL) — แกน z ของโมเดลหน้า = ทิศที่จมูกชี้
    """
    R = np.asarray(matrix, dtype=np.float64)[:3, :3]
    fx, fy, fz = R[:, 2]          # ทิศที่ใบหน้าหันไป
    ux, uy, _ = R[:, 1]           # ทิศ "ขึ้น" ของใบหน้า
    yaw = math.degrees(math.atan2(fx, fz))
    pitch = math.degrees(math.atan2(-fy, math.hypot(fx, fz)))
    roll = math.degrees(math.atan2(ux, uy))
    return yaw, pitch, roll


class FaceMeshDetector:
    """ตรวจจับใบหน้าทีละเฟรม (โหมดวิดีโอ: ใช้ผลเฟรมก่อนช่วย tracking → นิ่งและเร็วกว่าโหมดภาพ)

    Example:
        >>> detector = FaceMeshDetector()
        >>> landmarks = detector.detect(rgb_frame)
        >>> if landmarks is not None:
        ...     yaw, pitch, roll = detector.head_pose
    """

    # มุมปาก (ใช้คำนวณ ΔSmile)
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291
    NOSE_TIP = 1

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=ensure_model()),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts = -1
        self.head_pose = None     # (yaw, pitch, roll) ของเฟรมล่าสุด หรือ None

    def detect(self, rgb_frame):
        """ตรวจจับใบหน้า

        Args:
            rgb_frame: ภาพ RGB (np.ndarray)

        Returns:
            list ของ (x, y, z) 478 จุด (x, y ∈ [0, 1]) หรือ None ถ้าไม่พบใบหน้า
            ท่าศีรษะของเฟรมนี้อยู่ใน self.head_pose
        """
        # โหมดวิดีโอต้องการ timestamp ที่เพิ่มขึ้นเสมอ
        ts = max(int(time.perf_counter() * 1000), self._last_ts + 1)
        self._last_ts = ts
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_frame))
        result = self._landmarker.detect_for_video(image, ts)

        if not result.face_landmarks:
            self.head_pose = None
            return None
        if result.facial_transformation_matrixes:
            self.head_pose = pose_from_matrix(result.facial_transformation_matrixes[0])
        else:
            self.head_pose = None
        return [(lm.x, lm.y, lm.z) for lm in result.face_landmarks[0]]

    def get_mouth_corners(self, landmarks):
        return landmarks[self.MOUTH_LEFT], landmarks[self.MOUTH_RIGHT]

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
