"""
EYE ORDER COME AI — MediaPipe Face Mesh Wrapper
ตรวจจับใบหน้าและสกัด 478 landmarks + iris landmarks
"""
import os
import warnings

# ปิดข้อความเตือนที่ไม่เกี่ยวกับการทำงาน (protobuf deprecation / log ของ MediaPipe-TFLite)
warnings.filterwarnings("ignore", message=".*GetPrototype.*")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import mediapipe as mp
import numpy as np


class FaceMeshDetector:
    """Wrapper สำหรับ MediaPipe Face Mesh
    
    ตรวจจับใบหน้าและ return พิกัด 478 จุด (+ 10 iris landmarks)
    พร้อมฟังก์ชันช่วยดึง landmarks เฉพาะส่วน
    
    Example:
        >>> detector = FaceMeshDetector()
        >>> rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        >>> landmarks = detector.detect(rgb_frame)
        >>> if landmarks is not None:
        ...     left_eye = detector.get_left_eye(landmarks)
    """
    
    def __init__(self, max_faces=1, refine_landmarks=True,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Args:
            max_faces: จำนวนใบหน้าสูงสุดที่ตรวจจับ
            refine_landmarks: ถ้า True จะตรวจจับ iris ด้วย (478+10 จุด)
            min_detection_confidence: confidence threshold สำหรับ detection
            min_tracking_confidence: confidence threshold สำหรับ tracking
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        
        # Landmark indices
        # Left eye: p1(inner corner), p2(upper-left), p3(upper-right),
        #           p4(outer corner), p5(lower-right), p6(lower-left)
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        
        # Iris center
        self.LEFT_IRIS_CENTER = 468
        self.RIGHT_IRIS_CENTER = 473
        
        # Left iris ring
        self.LEFT_IRIS = [468, 469, 470, 471, 472]
        self.RIGHT_IRIS = [473, 474, 475, 476, 477]
        
        # Mouth corners
        self.MOUTH_LEFT = 61
        self.MOUTH_RIGHT = 291
        
        # Head pose reference points (6 จุด)
        # nose tip, chin, left eye inner, right eye inner, left mouth, right mouth
        self.HEAD_POSE_POINTS = [1, 152, 33, 263, 61, 291]
        
        # Upper/Lower lid midpoints สำหรับ iris Y ratio
        self.LEFT_UPPER_LID = 159   # กลางเปลือกตาบนซ้าย
        self.LEFT_LOWER_LID = 145   # กลางเปลือกตาล่างซ้าย
        self.RIGHT_UPPER_LID = 386  # กลางเปลือกตาบนขวา
        self.RIGHT_LOWER_LID = 374  # กลางเปลือกตาล่างขวา
    
    def detect(self, rgb_frame):
        """ตรวจจับใบหน้าและ return landmarks
        
        Args:
            rgb_frame: ภาพ RGB (np.ndarray)
        
        Returns:
            list หรือ None: list ของ (x, y, z) สำหรับทุก landmark
                           x, y อยู่ในช่วง [0, 1] (normalized)
                           z เป็นความลึกสัมพัทธ์
                           None ถ้าไม่พบใบหน้า
        """
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
            face = results.multi_face_landmarks[0]
            landmarks = [
                (lm.x, lm.y, lm.z) for lm in face.landmark
            ]
            return landmarks
        
        return None
    
    def get_landmarks_by_indices(self, landmarks, indices):
        """ดึง landmarks ตาม indices ที่ต้องการ
        
        Args:
            landmarks: list ของ (x, y, z) ทั้งหมด
            indices: list ของ index ที่ต้องการ
        
        Returns:
            list: [(x, y, z), ...] ของ landmarks ที่ต้องการ
        """
        return [landmarks[i] for i in indices]
    
    def get_left_eye(self, landmarks):
        """ดึง 6 จุดของตาซ้าย"""
        return self.get_landmarks_by_indices(landmarks, self.LEFT_EYE)
    
    def get_right_eye(self, landmarks):
        """ดึง 6 จุดของตาขวา"""
        return self.get_landmarks_by_indices(landmarks, self.RIGHT_EYE)
    
    def get_left_iris_center(self, landmarks):
        """ดึงจุดศูนย์กลาง iris ซ้าย"""
        return landmarks[self.LEFT_IRIS_CENTER]
    
    def get_right_iris_center(self, landmarks):
        """ดึงจุดศูนย์กลาง iris ขวา"""
        return landmarks[self.RIGHT_IRIS_CENTER]
    
    def get_mouth_corners(self, landmarks):
        """ดึงมุมปากซ้าย-ขวา
        
        Returns:
            tuple: (left_corner, right_corner) ใน (x, y, z)
        """
        return landmarks[self.MOUTH_LEFT], landmarks[self.MOUTH_RIGHT]
    
    def get_head_pose_points(self, landmarks):
        """ดึง 6 จุดสำหรับ Head Pose estimation"""
        return self.get_landmarks_by_indices(landmarks, self.HEAD_POSE_POINTS)
    
    def get_nose_tip(self, landmarks):
        """ดึงพิกัด (x, y, z) ของปลายจมูก (Landmark 1)"""
        return landmarks[1]
    
    def get_iris_ratio_points(self, landmarks, side="left"):
        """ดึงจุดที่จำเป็นสำหรับคำนวณ iris ratio
        
        Args:
            landmarks: all landmarks
            side: "left" หรือ "right"
        
        Returns:
            dict: {
                "iris_center": (x,y,z),
                "inner_corner": (x,y,z),
                "outer_corner": (x,y,z),
                "upper_lid": (x,y,z),
                "lower_lid": (x,y,z),
            }
        """
        if side == "left":
            return {
                "iris_center": landmarks[self.LEFT_IRIS_CENTER],
                "inner_corner": landmarks[self.LEFT_EYE[0]],  # 33
                "outer_corner": landmarks[self.LEFT_EYE[3]],  # 133
                "upper_lid": landmarks[self.LEFT_UPPER_LID],
                "lower_lid": landmarks[self.LEFT_LOWER_LID],
            }
        else:
            return {
                "iris_center": landmarks[self.RIGHT_IRIS_CENTER],
                "inner_corner": landmarks[self.RIGHT_EYE[0]],  # 362
                "outer_corner": landmarks[self.RIGHT_EYE[3]],  # 263
                "upper_lid": landmarks[self.RIGHT_UPPER_LID],
                "lower_lid": landmarks[self.RIGHT_LOWER_LID],
            }
    
    def close(self):
        """ปิด Face Mesh"""
        self.face_mesh.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
