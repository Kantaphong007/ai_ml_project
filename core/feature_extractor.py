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

import numpy as np
from core.face_mesh import FaceMeshDetector
from core.smile import SmileEstimator
from utils.math_utils import calculate_ear_multi, calculate_mouth_width

# ตา: (หัวตา/หางตา, คู่เปลือกตาบน-ล่าง 3 คู่) — ใช้ 3 คู่แทน 2 ลด noise ของ EAR
_LEFT_EYE_CORNERS = (33, 133)
_LEFT_EYE_LIDS = ((160, 144), (159, 145), (158, 153))
_RIGHT_EYE_CORNERS = (362, 263)
_RIGHT_EYE_LIDS = ((385, 380), (386, 374), (387, 373))
_EYE_OUTER_L, _EYE_OUTER_R = 33, 263          # ใช้เป็นสเกลของใบหน้า
_FOREHEAD, _CHIN, _NOSE_TIP = 10, 152, 1

# EAR version ที่บันทึกใน baseline.json (ค่า EAR เวอร์ชันเก่าคนละสเกล ห้ามปนกัน)
EAR_VERSION = 2

# ฟีเจอร์ของโมเดลเคอร์เซอร์ (ลำดับคงที่ — ไฟล์ calibration / โมเดลอ้างอิงตามชื่อ)
# v5: มุมศีรษะจาก facial transformation matrix (เดิม solvePnP 6 จุด ใช้ไม่ได้)
CURSOR_FEATURE_VERSION = 5
CURSOR_FEATURE_NAMES = [
    "head_yaw", "head_pitch", "head_roll",
    "nose_offset_x", "nose_offset_y",
    "nose_x", "nose_y",
]


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

    def __init__(self, face_mesh_detector: FaceMeshDetector, frame_size=(640, 480)):
        """
        Args:
            face_mesh_detector: instance ของ FaceMeshDetector
            frame_size: (w, h) ของเฟรม — ใช้แปลงพิกัด normalized เป็นพิกเซล
        """
        self.detector = face_mesh_detector
        self.frame_w, self.frame_h = frame_size
        # ΔSmile ชดเชยการก้ม/เงย + ค่าฐานปรับตัวช้าๆ (ดู core/smile.py)
        self.smile = SmileEstimator()
        self._cursor_history = []         # Feature history buffer ป้องกันการแกว่ง

    def set_frame_size(self, w, h):
        if w and h:
            self.frame_w, self.frame_h = w, h

    def reset_cursor_history(self):
        """ล้าง history ของฟีเจอร์เคอร์เซอร์ (เรียกเมื่อหน้าหายไปนาน/เริ่มใหม่)"""
        self._cursor_history = []

    def set_baseline_mouth_ratio(self, ratio, pitch=None):
        """ตั้งค่าฐาน (ความกว้างปาก ÷ ระยะหางตา, สัญญาณก้มเงย) จากหน้าปกติ — None = เรียนเองจากกล้อง"""
        self.smile.set_baseline(ratio, pitch)

    def set_smile_pitch_comp(self, k):
        """ค่าชดเชยการก้ม/เงยของ ΔSmile (จูนจากข้อมูลโดย training/tune_gestures.py)"""
        self.smile.pitch_comp = float(k)

    def _px(self, landmarks, idx):
        x, y, z = landmarks[idx][:3]
        return (x * self.frame_w, y * self.frame_h, z * self.frame_w)
    
    def extract_cursor_features(self, landmarks, frame_width=None, frame_height=None):
        """สกัดฟีเจอร์ 7 ตัวสำหรับ Cursor Regression (ลำดับตาม CURSOR_FEATURE_NAMES)

        [head_yaw, head_pitch, head_roll, nose_offset_x, nose_offset_y, nose_x, nose_y]
          head_*      : มุมศีรษะจาก facial transformation matrix ของ MediaPipe (องศา)
          nose_offset : ปลายจมูกเทียบกึ่งกลางกรอบหน้า (การหมุนหัวล้วน ไม่ขึ้นกับท่านั่ง)
          nose_x/y    : ตำแหน่งปลายจมูกในภาพ (เปลี่ยนตามท่านั่ง — ให้ feature selection ตัดสิน)
        """
        if landmarks is None or self.detector.head_pose is None:
            return None
        try:
            yaw, pitch, roll = self.detector.head_pose
            nose = landmarks[self.detector.NOSE_TIP]
            # กึ่งกลางกรอบหน้า: โหนกแก้มซ้าย (#234) / ขวา (#454), หน้าผาก (#10) / คาง (#152)
            face_cx = (landmarks[234][0] + landmarks[454][0]) / 2.0
            face_cy = (landmarks[10][1] + landmarks[152][1]) / 2.0
            raw_feat = np.array([
                yaw, pitch, roll,
                nose[0] - face_cx, nose[1] - face_cy,
                nose[0], nose[1],
            ], dtype=np.float64)

            # weighted moving average 3 เฟรม (หน่วงน้อยกว่า 1 เฟรม) — กรองหลักอยู่ชั้นถัดไป
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
            pitch = self.extract_head_pitch_signal(landmarks)
            delta_smile = self.smile.update(ratio, pitch) if ratio else 0.0

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
        """ชื่อ columns สำหรับ cursor features"""
        return list(CURSOR_FEATURE_NAMES)
    
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
