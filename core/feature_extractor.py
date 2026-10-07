"""
EYE ORDER COME AI — Feature Extractor
สกัดฟีเจอร์จาก Face Mesh landmarks สำหรับ 2 โมเดล:
  1. Cursor Regression: 7 features (Nose Tip 2D + Head Pose + Face Outline Offset)
  2. Click Classification: 3 features per frame (EAR_L, EAR_R, ΔSmile)
และสัญญาณก้ม/เงยสำหรับโหมด scroll

EAR / ความกว้างปาก คำนวณในพิกัด 3 มิติที่สเกลเป็นพิกเซลแล้ว (x·w, y·h, z·w)
  - พิกัด normalized เดิม x หาร w แต่ y หาร h → ระยะเบี้ยวตามสัดส่วนภาพและมุมเอียงหัว
  - ระยะ 3 มิติแทบไม่เปลี่ยนเมื่อก้ม/เงย/หันหน้า → EAR นิ่งขึ้นตอนคุมเคอร์เซอร์ด้วยศีรษะ
ΔSmile เทียบความกว้างปากกับระยะหางตาซ้าย–ขวา → ไม่เพี้ยนเมื่อนั่งใกล้/ไกลกล้อง
"""
from collections import deque

import numpy as np
from core.face_mesh import FaceMeshDetector
from utils.math_utils import (
    calculate_ear_multi,
    calculate_mouth_width,
    solve_head_pose,
)

# ตา: (หัวตา/หางตา, คู่เปลือกตาบน-ล่าง 3 คู่) — ใช้ 3 คู่แทน 2 ลด noise ของ EAR
_LEFT_EYE_CORNERS = (33, 133)
_LEFT_EYE_LIDS = ((160, 144), (159, 145), (158, 153))
_RIGHT_EYE_CORNERS = (362, 263)
_RIGHT_EYE_LIDS = ((385, 380), (386, 374), (387, 373))
_EYE_OUTER_L, _EYE_OUTER_R = 33, 263          # ใช้เป็นสเกลของใบหน้า
_FOREHEAD, _CHIN, _NOSE_TIP = 10, 152, 1

# EAR version ที่บันทึกใน baseline.json (ค่า EAR เวอร์ชันเก่าคนละสเกล ห้ามปนกัน)
EAR_VERSION = 2


class FeatureExtractor:
    """สกัดฟีเจอร์จาก face landmarks
    
    ผลิต 2 ชุดฟีเจอร์ต่อเฟรม:
    
    1. cursor_features (7 ตัว):
       [nose_x, nose_y, pitch, yaw, roll, nose_offset_x, nose_offset_y]
       (ปลายจมูก + การหันหัว เท่านั้น ไม่ใช้ตำแหน่งลูกตา/ทิศทางสายตา)
    
    2. click_features (3 ตัว):
       [EAR_L, EAR_R, ΔSmile]
    
    Example:
        >>> detector = FaceMeshDetector()
        >>> extractor = FeatureExtractor(detector)
        >>> landmarks = detector.detect(rgb_frame)
        >>> cursor_feat = extractor.extract_cursor_features(landmarks, 640, 480)
        >>> click_feat = extractor.extract_click_features(landmarks)
    """
    
    FEATURE_SMOOTH_FRAMES = 3
    # ค่าฐานปากปรับตัวช้าๆ: median ของ ~10 วินาทีล่าสุดที่ "ไม่ได้ยิ้ม"
    MOUTH_ADAPT_FRAMES = 300
    MOUTH_LEARN_MIN = 15
    MOUTH_ADAPT_MAX_DELTA = 0.10

    def __init__(self, face_mesh_detector: FaceMeshDetector, frame_size=(640, 480)):
        """
        Args:
            face_mesh_detector: instance ของ FaceMeshDetector
            frame_size: (w, h) ของเฟรม — ใช้แปลงพิกัด normalized เป็นพิกเซล
        """
        self.detector = face_mesh_detector
        self.frame_w, self.frame_h = frame_size
        self.baseline_mouth_ratio = None  # ความกว้างปาก ÷ ระยะหางตา ตอนหน้าปกติ
        self._mouth_buf = deque(maxlen=self.MOUTH_ADAPT_FRAMES)
        self._cursor_history = []         # Feature history buffer ป้องกันการแกว่ง

    def set_frame_size(self, w, h):
        if w and h:
            self.frame_w, self.frame_h = w, h

    def reset_cursor_history(self):
        """ล้าง history ของฟีเจอร์เคอร์เซอร์ (เรียกเมื่อหน้าหายไปนาน/เริ่มใหม่)"""
        self._cursor_history = []

    def set_baseline_mouth_ratio(self, ratio):
        """ตั้งค่าฐาน (ความกว้างปาก ÷ ระยะหางตา) จากหน้าปกติ — None = ให้เรียนเองจากกล้อง"""
        self.baseline_mouth_ratio = float(ratio) if ratio else None
        self._mouth_buf.clear()

    def _px(self, landmarks, idx):
        x, y, z = landmarks[idx][:3]
        return (x * self.frame_w, y * self.frame_h, z * self.frame_w)
    
    def extract_cursor_features(self, landmarks, frame_width, frame_height):
        """สกัดฟีเจอร์ 7 ตัวสำหรับ Nose Tip / Head Motion Cursor Regression
        
        Features:
          [nose_x, nose_y, pitch, yaw, roll, nose_offset_x, nose_offset_y]
        """
        if landmarks is None:
            return None
        
        try:
            # ── 1. Nose Tip 2D coordinates ──
            nose_tip = self.detector.get_nose_tip(landmarks)
            nose_x, nose_y = float(nose_tip[0]), float(nose_tip[1])
            
            # ── 2. Head pose angles ──
            head_pose_pts = self.detector.get_head_pose_points(landmarks)
            pitch, yaw, roll = solve_head_pose(
                head_pose_pts, frame_width, frame_height
            )
            
            # ── 3. Relative Nose Offset from Pure Face Outline Center (อิสระจากสายตา 100%) ──
            # โหนกแก้มซ้าย (#234) & ขวา (#454), หน้าผาก (#10) & คาง (#152)
            left_cheek = landmarks[234]
            right_cheek = landmarks[454]
            forehead = landmarks[10]
            chin = landmarks[152]
            
            face_center_x = (left_cheek[0] + right_cheek[0]) / 2.0
            face_center_y = (forehead[1] + chin[1]) / 2.0
            
            nose_offset_x = nose_x - face_center_x
            nose_offset_y = nose_y - face_center_y
            
            raw_feat = np.array([
                nose_x, nose_y,
                pitch, yaw, roll,
                nose_offset_x, nose_offset_y
            ], dtype=np.float64)
            
            # กรองเบาๆ ด้วย weighted moving average 3 เฟรม (หน่วงน้อยกว่า 1 เฟรม)
            # การกรองหลักทำที่ One-Euro filter ใน CursorPredictor เพียงชั้นเดียว
            # (เดิมกรองซ้อนหลายชั้นจนเคอร์เซอร์หน่วงและคุมยาก)
            self._cursor_history.append(raw_feat)
            if len(self._cursor_history) > self.FEATURE_SMOOTH_FRAMES:
                self._cursor_history = self._cursor_history[-self.FEATURE_SMOOTH_FRAMES:]

            weights = np.arange(1, len(self._cursor_history) + 1, dtype=np.float64)
            weights /= weights.sum()
            return np.average(self._cursor_history, axis=0, weights=weights)

        except Exception:
            return None
    
    def _ear(self, landmarks, corners, lids):
        px = lambda i: self._px(landmarks, i)
        return calculate_ear_multi(px(corners[0]), px(corners[1]),
                                   [(px(u), px(l)) for u, l in lids])

    def compute_mouth_ratio(self, landmarks):
        """ความกว้างปาก ÷ ระยะหางตาซ้าย–ขวา (ไม่ขึ้นกับระยะห่างจากกล้อง)"""
        if landmarks is None:
            return None
        try:
            mouth = calculate_mouth_width(self._px(landmarks, self.detector.MOUTH_LEFT),
                                          self._px(landmarks, self.detector.MOUTH_RIGHT))
            eyes = calculate_mouth_width(self._px(landmarks, _EYE_OUTER_L),
                                         self._px(landmarks, _EYE_OUTER_R))
            return mouth / eyes if eyes > 0 else None
        except Exception:
            return None

    def _delta_smile(self, ratio):
        """ΔSmile เทียบค่าฐาน พร้อมปรับค่าฐานช้าๆ จากช่วงที่ไม่ได้ยิ้ม"""
        ref = self.baseline_mouth_ratio
        delta = 0.0 if not ref else ratio / ref - 1.0
        if not ref or delta < self.MOUTH_ADAPT_MAX_DELTA:
            self._mouth_buf.append(ratio)
            if len(self._mouth_buf) >= self.MOUTH_LEARN_MIN:
                self.baseline_mouth_ratio = float(np.median(self._mouth_buf))
        return delta

    def extract_click_features(self, landmarks):
        """สกัดฟีเจอร์ 3 ตัวสำหรับ Click Classification (1 เฟรม)
        
        Args:
            landmarks: list ของ (x, y, z) จาก FaceMeshDetector.detect()
        
        Returns:
            np.ndarray: shape (3,) → [EAR_L, EAR_R, ΔSmile]
            None: ถ้าไม่สามารถสกัดได้
        """
        if landmarks is None:
            return None
        
        try:
            ear_l = self._ear(landmarks, _LEFT_EYE_CORNERS, _LEFT_EYE_LIDS)
            ear_r = self._ear(landmarks, _RIGHT_EYE_CORNERS, _RIGHT_EYE_LIDS)

            ratio = self.compute_mouth_ratio(landmarks)
            delta_smile = self._delta_smile(ratio) if ratio else 0.0

            return np.array([ear_l, ear_r, delta_smile], dtype=np.float64)
            
        except Exception:
            return None

    def extract_head_pitch_signal(self, landmarks):
        """สัญญาณก้ม/เงยสำหรับโหมด scroll (+ = ก้ม, − = เงย)

        = (ปลายจมูก − กึ่งกลางหน้าผาก/คาง) ÷ ความสูงหน้า ในแนวตั้ง
        ไม่ขึ้นกับโมเดลเคอร์เซอร์ / ขอบจอ / ระยะห่างจากกล้อง / การขยับตัวขึ้นลง
        (มุม pitch จาก solvePnP ใช้ไม่ได้ — สั่นและเพี้ยนตามมุมปากเวลายิ้ม)
        """
        if landmarks is None:
            return None
        try:
            fy = landmarks[_FOREHEAD][1]
            cy = landmarks[_CHIN][1]
            ny = landmarks[_NOSE_TIP][1]
            h = cy - fy
            if h <= 1e-6:
                return None
            return float((ny - (fy + cy) / 2.0) / h)
        except Exception:
            return None
    
    def extract_all_features(self, landmarks, frame_width, frame_height):
        """สกัดฟีเจอร์ทั้ง 2 ชุดพร้อมกัน
        
        Args:
            landmarks: list ของ (x, y, z)
            frame_width: ความกว้างเฟรม
            frame_height: ความสูงเฟรม
        
        Returns:
            tuple: (cursor_features, click_features)
                   cursor_features: np.ndarray shape (7,) หรือ None
                   click_features: np.ndarray shape (3,) หรือ None
        """
        cursor_feat = self.extract_cursor_features(
            landmarks, frame_width, frame_height
        )
        click_feat = self.extract_click_features(landmarks)
        return cursor_feat, click_feat
    
    @staticmethod
    def get_cursor_feature_names():
        """ชื่อ columns สำหรับ cursor features (Nose Tip & Head Pose)"""
        return [
            "nose_x", "nose_y",
            "pitch", "yaw", "roll",
            "nose_offset_x", "nose_offset_y"
        ]
    
    @staticmethod
    def get_click_feature_names():
        """ชื่อ columns สำหรับ click features (1 เฟรม)"""
        return ["EAR_L", "EAR_R", "delta_smile"]
    
    @staticmethod
    def get_click_sliding_feature_names(window_size=30):
        """ชื่อ columns สำหรับ click features (flatten จาก sliding window)"""
        base_names = ["EAR_L", "EAR_R", "delta_smile"]
        names = []
        for i in range(window_size):
            for name in base_names:
                names.append(f"{name}_f{i}")
        return names
