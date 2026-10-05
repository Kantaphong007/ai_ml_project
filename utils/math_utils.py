"""
EYE ORDER COME AI — Math Utilities
ฟังก์ชันคำนวณ EAR, ΔSmile, Iris Position, Head Pose
"""
import numpy as np
import cv2


def euclidean_distance(p1, p2):
    """คำนวณระยะห่าง Euclidean ระหว่าง 2 จุด
    
    Args:
        p1: จุดที่ 1 (x, y) หรือ (x, y, z)
        p2: จุดที่ 2 (x, y) หรือ (x, y, z)
    
    Returns:
        float: ระยะห่าง
    """
    p1 = np.array(p1, dtype=np.float64)
    p2 = np.array(p2, dtype=np.float64)
    return np.linalg.norm(p1 - p2)


def calculate_ear(eye_landmarks):
    """คำนวณ Eye Aspect Ratio (EAR)
    
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    
    เมื่อ:
        p1 = หัวตา (inner corner)
        p2 = เปลือกตาบน-ซ้าย
        p3 = เปลือกตาบน-ขวา
        p4 = หางตา (outer corner)
        p5 = เปลือกตาล่าง-ขวา
        p6 = เปลือกตาล่าง-ซ้าย
    
    Args:
        eye_landmarks: list ของ 6 จุด [(x,y), ...] ตามลำดับ p1-p6
    
    Returns:
        float: ค่า EAR (สูง ~0.25-0.35 เมื่อลืมตา, ต่ำ ~0 เมื่อหลับตา)
    """
    p1, p2, p3, p4, p5, p6 = eye_landmarks
    
    # ระยะห่างแนวตั้ง
    vertical_1 = euclidean_distance(p2, p6)
    vertical_2 = euclidean_distance(p3, p5)
    
    # ระยะห่างแนวนอน
    horizontal = euclidean_distance(p1, p4)
    
    if horizontal == 0:
        return 0.0
    
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear


def calculate_mouth_width(left_corner, right_corner):
    """คำนวณความกว้างปาก
    
    Args:
        left_corner: พิกัดมุมปากซ้าย (x, y)
        right_corner: พิกัดมุมปากขวา (x, y)
    
    Returns:
        float: ความกว้างปาก (ระยะ Euclidean)
    """
    return euclidean_distance(left_corner, right_corner)


def calculate_delta_smile(current_width, baseline_width):
    """คำนวณ Relative Smile Ratio (ΔSmile)
    
    ΔSmile = (current - baseline) / baseline
    
    Args:
        current_width: ความกว้างปากปัจจุบัน
        baseline_width: ความกว้างปากค่าฐาน (หน้าปกติ)
    
    Returns:
        float: เปอร์เซ็นต์การขยายตัว (0.25 = ขยาย 25%)
    """
    if baseline_width <= 0:
        return 0.0
    return (current_width - baseline_width) / baseline_width


def calculate_iris_ratio(iris_center, inner_corner, outer_corner,
                         upper_lid=None, lower_lid=None):
    """คำนวณ Normalized Iris Position
    
    ratio_x: ตำแหน่ง iris เทียบแนวนอน (0=หัวตา, 1=หางตา)
    ratio_y: ตำแหน่ง iris เทียบแนวตั้ง (0=เปลือกตาบน, 1=เปลือกตาล่าง)
    
    Args:
        iris_center: จุดศูนย์กลาง iris (x, y)
        inner_corner: หัวตา (x, y)
        outer_corner: หางตา (x, y)
        upper_lid: เปลือกตาบน (x, y) — optional, ใช้สำหรับ ratio_y
        lower_lid: เปลือกตาล่าง (x, y) — optional
    
    Returns:
        tuple: (ratio_x, ratio_y)
    """
    iris = np.array(iris_center, dtype=np.float64)
    inner = np.array(inner_corner, dtype=np.float64)
    outer = np.array(outer_corner, dtype=np.float64)
    
    # X ratio: ตำแหน่งแนวนอนเทียบกับความกว้างตา
    eye_width = euclidean_distance(inner, outer)
    if eye_width == 0:
        ratio_x = 0.5
    else:
        # ฉายจุด iris ลงบนเส้นหัวตา-หางตา
        eye_vec = outer - inner
        iris_vec = iris[:2] - inner[:2]
        ratio_x = np.dot(iris_vec, eye_vec[:2]) / (eye_width ** 2)
        ratio_x = np.clip(ratio_x, 0.0, 1.0)
    
    # Y ratio: ตำแหน่งแนวตั้งเทียบกับความสูงตา
    if upper_lid is not None and lower_lid is not None:
        upper = np.array(upper_lid, dtype=np.float64)
        lower = np.array(lower_lid, dtype=np.float64)
        eye_height = euclidean_distance(upper, lower)
        if eye_height == 0:
            ratio_y = 0.5
        else:
            ratio_y = euclidean_distance(upper[:2], iris[:2]) / eye_height
            ratio_y = np.clip(ratio_y, 0.0, 1.0)
    else:
        ratio_y = 0.5
    
    return ratio_x, ratio_y


def solve_head_pose(face_landmarks_6, frame_width, frame_height):
    """คำนวณ Head Pose Angles ผ่าน solvePnP
    
    ใช้ 6 จุดอ้างอิงบนใบหน้า:
        0: ปลายจมูก (nose tip)
        1: คาง (chin)
        2: หัวตาซ้าย (left eye inner corner)
        3: หัวตาขวา (right eye inner corner)
        4: มุมปากซ้าย (left mouth corner)
        5: มุมปากขวา (right mouth corner)
    
    Args:
        face_landmarks_6: list ของ 6 จุด [(x, y, z), ...] ใน normalized coords
        frame_width: ความกว้างเฟรม (pixels)
        frame_height: ความสูงเฟรม (pixels)
    
    Returns:
        tuple: (pitch, yaw, roll) ในหน่วยองศา
            - pitch: ก้ม(-) / เงย(+)
            - yaw: หันซ้าย(-) / หันขวา(+)
            - roll: เอียงซ้าย(-) / เอียงขวา(+)
    """
    # 3D model points ของใบหน้ามาตรฐาน (Generic face model)
    model_points = np.array([
        (0.0, 0.0, 0.0),           # Nose tip
        (0.0, -330.0, -65.0),      # Chin
        (-225.0, 170.0, -135.0),   # Left eye inner corner
        (225.0, 170.0, -135.0),    # Right eye inner corner
        (-150.0, -150.0, -125.0),  # Left mouth corner
        (150.0, -150.0, -125.0),   # Right mouth corner
    ], dtype=np.float64)
    
    # แปลง normalized landmarks → pixel coords
    image_points = np.array([
        (lm[0] * frame_width, lm[1] * frame_height)
        for lm in face_landmarks_6
    ], dtype=np.float64)
    
    # Camera matrix (ประมาณจาก frame size)
    focal_length = frame_width
    center = (frame_width / 2, frame_height / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # ไม่มี lens distortion
    dist_coeffs = np.zeros((4, 1))
    
    # Solve PnP
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        return 0.0, 0.0, 0.0
    
    # แปลง rotation vector → rotation matrix → Euler angles
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    
    # ดึง Euler angles
    # ใช้ decomposition จาก rotation matrix
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)

    if sy > 1e-6:
        pitch = np.degrees(np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]))
        yaw = np.degrees(np.arctan2(-rotation_matrix[2, 0], sy))
        roll = np.degrees(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))
    else:
        pitch = np.degrees(np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1]))
        yaw = np.degrees(np.arctan2(-rotation_matrix[2, 0], sy))
        roll = 0.0

    # model_points ใช้แกน Y ขึ้น แต่ภาพใช้ Y ลง → มุม pitch/roll ได้ค่าใกล้ ±180°
    # และ "วน" ข้ามขอบ (+179 ↔ -179) ทำให้ฟีเจอร์กระโดดและเคอร์เซอร์เพี้ยน
    # จึงพับกลับเข้าช่วง [-90, 90] ซึ่งเป็นช่วงที่ศีรษะหมุนได้จริง
    return fold_head_angle(pitch), float(yaw), fold_head_angle(roll)


def fold_head_angle(angle_deg):
    """พับมุม (องศา) ให้อยู่ในช่วง [-90, 90] — ใช้ได้ทั้งค่าเดี่ยวและ array

    ตัวอย่าง: 171 → -9, -172 → 8, 12 → 12 (ค่าที่อยู่ในช่วงแล้วไม่เปลี่ยน)
    """
    a = (np.asarray(angle_deg, dtype=np.float64) + 180.0) % 360.0 - 180.0
    a = np.where(a > 90.0, a - 180.0, a)
    a = np.where(a < -90.0, a + 180.0, a)
    return float(a) if a.ndim == 0 else a
