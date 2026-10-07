"""
EYE ORDER COME AI — Smile Estimator (ΔSmile ที่ชดเชยการก้ม/เงย)

ปัญหา (วัดจากข้อมูลจริง): อัตราส่วน "ความกว้างปาก ÷ ระยะหางตา" เปลี่ยนตามการก้ม/เงย
  เงยหน้า → ค่าเพิ่ม +0.15…+0.25 (เท่ากับการยิ้ม!)   ก้มหน้า → ค่าลด −0.2…−0.3
  ผลเดิม: เงยหน้าเพื่อเลื่อนขึ้น = ระบบคิดว่ายิ้ม → หยุด scroll (เลื่อนขึ้นไม่ได้)
          และถ้ายิ้มค้างแล้วก้ม → ค่ายิ้มตก → หลุดโหมด scroll

แก้: ΔSmile = ratio / ratio_ref − 1 + k × (pitch − pitch_ref)
  k ได้จาก regression บนเฟรมที่ไม่ได้ยิ้ม (training/tune_gestures.py, ข้อมูลจริงได้ ≈ 0.8)
  ratio_ref / pitch_ref = median ของ ~10 วินาทีล่าสุดที่ไม่ได้ยิ้ม (ปรับตามท่านั่งเอง)

pure python — ใช้ทั้งตอนใช้งานจริง (FeatureExtractor) และตอน replay ข้อมูล (training/signal_data.py)
"""
from collections import deque
from statistics import median

DEFAULT_PITCH_COMP = 0.8


class SmileEstimator:
    ADAPT_FRAMES = 300          # ~10 วินาทีที่ 30 FPS
    LEARN_MIN = 15
    ADAPT_MAX_DELTA = 0.10      # เฟรมที่ยิ้มน้อยกว่านี้ใช้ปรับค่าฐาน

    def __init__(self, pitch_comp=DEFAULT_PITCH_COMP):
        self.pitch_comp = float(pitch_comp)
        self.ratio_ref = None
        self.pitch_ref = None
        self._buf = deque(maxlen=self.ADAPT_FRAMES)

    def set_baseline(self, ratio=None, pitch=None):
        """ค่าฐานจาก baseline.json (None = เรียนเองจากกล้อง)"""
        self.ratio_ref = float(ratio) if ratio else None
        self.pitch_ref = float(pitch) if pitch is not None else None
        self._buf.clear()

    def update(self, ratio, pitch=None):
        """ป้อน 1 เฟรม → ΔSmile ที่ชดเชยการก้ม/เงยแล้ว"""
        if pitch is None:
            pitch = self.pitch_ref if self.pitch_ref is not None else 0.0
        delta = 0.0
        if self.ratio_ref:
            p_ref = self.pitch_ref if self.pitch_ref is not None else pitch
            delta = ratio / self.ratio_ref - 1.0 + self.pitch_comp * (pitch - p_ref)
        if not self.ratio_ref or delta < self.ADAPT_MAX_DELTA:
            self._buf.append((ratio, pitch))
            if len(self._buf) >= self.LEARN_MIN:
                self.ratio_ref = median(r for r, _ in self._buf)
                self.pitch_ref = median(p for _, p in self._buf)
        return delta
