"""
EYE ORDER COME AI — One-Euro Filter
ตัวกรองแบบปรับตามความเร็ว (Casiez et al., CHI 2012) เหมาะกับการควบคุมเคอร์เซอร์

  - เคลื่อนช้า/นิ่ง → cutoff ต่ำ → กรองแรง เคอร์เซอร์นิ่ง ไม่สั่น
  - เคลื่อนเร็ว     → cutoff สูง → กรองน้อย เคอร์เซอร์ตามทันแทบไม่หน่วง

ใช้แทนการกรองซ้อนหลายชั้น + deadzone เดิม ซึ่งทำให้เคอร์เซอร์หน่วงและคุมยาก
"""
import math


class OneEuroFilter:
    """กรองค่า 1 มิติ (สร้าง 1 ตัวต่อแกน)"""

    def __init__(self, min_cutoff=1.0, beta=0.01, d_cutoff=1.0):
        """
        Args:
            min_cutoff: cutoff (Hz) ตอนนิ่ง — ยิ่งต่ำยิ่งนิ่ง แต่หน่วงขึ้น
            beta: ตัวคูณความเร็ว — ยิ่งสูงยิ่งตามทันตอนเคลื่อนเร็ว
            d_cutoff: cutoff ของตัวประมาณความเร็ว
        """
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.reset()

    def reset(self):
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        """กรองค่า x ที่เวลา t (วินาที)"""
        if self._x_prev is None:
            self._x_prev = x
            self._t_prev = t
            return x

        # dt ต้องไม่เป็น 0 และไม่ใหญ่เกินจริง (กรณีเฟรมค้าง)
        dt = min(max(t - self._t_prev, 1e-3), 0.25)
        self._t_prev = t

        dx = (x - self._x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = self._dx_prev + a_d * (dx - self._dx_prev)

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = self._x_prev + a * (x - self._x_prev)

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat
