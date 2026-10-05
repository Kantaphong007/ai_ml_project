"""
EYE ORDER COME AI — Feature Extractor
สกัดฟีเจอร์จาก Face Mesh landmarks สำหรับ 2 โมเดล:
  1. Cursor Regression: 7 features (Nose Tip 2D + Head Pose + Face Outline Offset)
  2. Click Classification: 3 features per frame (EAR_L, EAR_R, ΔSmile)
"""
import numpy as np
from core.face_mesh import FaceMeshDetector
from utils.math_utils import (
    calculate_ear,
    calculate_mouth_width,
    calculate_delta_smile,
    calculate_iris_ratio,
    solve_head_pose,
)


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

    def __init__(self, face_mesh_detector: FaceMeshDetector):
        """
        Args:
            face_mesh_detector: instance ของ FaceMeshDetector
        """
        self.detector = face_mesh_detector
        self.baseline_mouth_width = None  # ค่าฐานความกว้างปาก (จะตั้งค่าจาก calibration)
        self._cursor_history = []         # Feature history buffer ป้องกันการแกว่ง

    def reset_cursor_history(self):
        """ล้าง history ของฟีเจอร์เคอร์เซอร์ (เรียกเมื่อหน้าหายไปนาน/เริ่มใหม่)"""
        self._cursor_history = []

    def set_baseline_mouth_width(self, width):
        """ตั้งค่าฐานความกว้างปาก (จากหน้าปกติ)"""
        self.baseline_mouth_width = width
    
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
            # ── EAR ──
            left_eye = self.detector.get_left_eye(landmarks)
            right_eye = self.detector.get_right_eye(landmarks)
            
            ear_l = calculate_ear(left_eye)
            ear_r = calculate_ear(right_eye)
            
            # ── ΔSmile ──
            mouth_l, mouth_r = self.detector.get_mouth_corners(landmarks)
            current_mouth_width = calculate_mouth_width(mouth_l, mouth_r)
            
            if self.baseline_mouth_width is not None and self.baseline_mouth_width > 0:
                delta_smile = calculate_delta_smile(
                    current_mouth_width, self.baseline_mouth_width
                )
            else:
                delta_smile = 0.0
            
            return np.array([ear_l, ear_r, delta_smile], dtype=np.float64)
            
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
    
    def compute_baseline_mouth_width(self, landmarks):
        """คำนวณ mouth width จาก landmarks ปัจจุบัน (สำหรับ baseline)
        
        Args:
            landmarks: list ของ (x, y, z)
        
        Returns:
            float: mouth width หรือ None
        """
        if landmarks is None:
            return None
        try:
            mouth_l, mouth_r = self.detector.get_mouth_corners(landmarks)
            return calculate_mouth_width(mouth_l, mouth_r)
        except Exception:
            return None

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
