"""
EYE ORDER COME AI — Math Utilities
ฟังก์ชันคำนวณระยะ / Eye Aspect Ratio / ความกว้างปาก
(มุมศีรษะได้จาก facial transformation matrix ใน core/face_mesh.py)
"""
import numpy as np


def euclidean_distance(p1, p2):
    """ระยะ Euclidean ระหว่าง 2 จุด (2 หรือ 3 มิติ)"""
    return float(np.linalg.norm(np.asarray(p1, dtype=np.float64) - np.asarray(p2, dtype=np.float64)))


def calculate_ear_multi(corner_a, corner_b, lid_pairs):
    """Eye Aspect Ratio แบบหลายคู่เปลือกตา (ลด noise) — ใช้พิกัดที่สเกลเป็นพิกเซลแล้ว

    EAR = mean(||upper_i − lower_i||) / ||corner_a − corner_b||

    Args:
        corner_a, corner_b: หัวตา/หางตา (x, y[, z])
        lid_pairs: list ของ (upper, lower) เปลือกตาบน-ล่างที่ตรงกัน

    Returns:
        float: ลืมตา ~0.25–0.35, หลับตา ~0.05
    """
    horizontal = euclidean_distance(corner_a, corner_b)
    if horizontal == 0:
        return 0.0
    vertical = sum(euclidean_distance(u, l) for u, l in lid_pairs) / len(lid_pairs)
    return vertical / horizontal


def calculate_mouth_width(left_corner, right_corner):
    """ความกว้างปาก (ระยะระหว่างมุมปาก) — ใช้คำนวณระยะอื่นแบบเดียวกันได้"""
    return euclidean_distance(left_corner, right_corner)
