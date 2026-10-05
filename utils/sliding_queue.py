"""
EYE ORDER COME AI — Sliding Queue
คิวเลื่อนสำหรับเก็บ time-series features ย้อนหลัง N เฟรม
ใช้สำหรับโมเดล Click Classification
"""
from collections import deque
import numpy as np


class SlidingQueue:
    """คิวเลื่อนขนาดคงที่สำหรับเก็บ feature vectors ย้อนหลัง
    
    เก็บค่า [EAR_L, EAR_R, ΔSmile] ย้อนหลัง window_size เฟรม
    เมื่อ queue เต็มแล้ว สามารถ flatten เป็น vector 1 มิติ
    สำหรับส่งเข้าโมเดล classification ได้
    
    Example:
        >>> sq = SlidingQueue(window_size=30, num_features=3)
        >>> sq.push([0.30, 0.31, 0.02])  # frame 1: [EAR_L, EAR_R, ΔSmile]
        >>> sq.push([0.29, 0.30, 0.01])  # frame 2
        >>> # ... push 28 more frames ...
        >>> if sq.is_full():
        ...     vector = sq.flatten()  # shape: (90,)
    """
    
    def __init__(self, window_size=30, num_features=3):
        """
        Args:
            window_size: จำนวนเฟรมที่เก็บ (default: 30 = 1 วินาทีที่ 30 FPS)
            num_features: จำนวน features ต่อเฟรม (default: 3 = EAR_L, EAR_R, ΔSmile)
        """
        self.window_size = window_size
        self.num_features = num_features
        self.queue = deque(maxlen=window_size)
    
    def push(self, frame_features):
        """เพิ่มข้อมูล 1 เฟรมเข้า queue
        
        Args:
            frame_features: list หรือ array ขนาด num_features
                           เช่น [EAR_L, EAR_R, ΔSmile]
        """
        if len(frame_features) != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, got {len(frame_features)}"
            )
        self.queue.append(list(frame_features))
    
    def is_full(self):
        """ตรวจสอบว่า queue เต็มแล้วหรือยัง
        
        Returns:
            bool: True ถ้ามีข้อมูลครบ window_size เฟรม
        """
        return len(self.queue) == self.window_size
    
    def flatten(self):
        """แปลง queue → vector 1 มิติ สำหรับส่งเข้าโมเดล
        
        ลำดับ: [frame0_feat0, frame0_feat1, ..., frame0_featN,
                frame1_feat0, frame1_feat1, ..., frameN_featN]
        
        Returns:
            np.ndarray: shape (window_size * num_features,)
        
        Raises:
            ValueError: ถ้า queue ยังไม่เต็ม
        """
        if not self.is_full():
            raise ValueError(
                f"Queue not full yet: {len(self.queue)}/{self.window_size}"
            )
        return np.array(self.queue, dtype=np.float64).flatten()
    
    def to_array(self):
        """แปลง queue → 2D array (ไม่ต้อง flatten)
        
        Returns:
            np.ndarray: shape (current_length, num_features)
        """
        return np.array(list(self.queue), dtype=np.float64)
    
    def clear(self):
        """ล้าง queue"""
        self.queue.clear()
    
    def __len__(self):
        return len(self.queue)
    
    def __repr__(self):
        return (
            f"SlidingQueue(window_size={self.window_size}, "
            f"num_features={self.num_features}, "
            f"current_length={len(self.queue)})"
        )
