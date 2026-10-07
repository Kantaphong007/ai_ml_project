"""
EYE ORDER COME AI — Head Pointer (โหมด Relative / Hybrid)

Relative: หมุนหัวเท่าไหร่ เคอร์เซอร์ขยับตามเท่านั้น (เหมือนเมาส์ / Apple Head Pointer)
  - pointer acceleration: หมุนช้า = gain ต่ำ (เล็งจุดเล็กได้แม่น) · หมุนเร็ว = gain สูง (ข้ามจอได้ไว)
  - ล็อกตอนนิ่ง: การขยับเล็กๆ (noise / ส่ายหัวตามธรรมชาติ) ไม่ทำให้เคอร์เซอร์ขยับ
  - ชนขอบจอแล้วหันต่อ = ติดขอบ

กันการ "หลุดจากทิศหน้า" (หน้าตรงแต่เคอร์เซอร์ไปค้างขอบ) ด้วย Gain modulation:
  relative สะสมความคลาดได้ (หันเลยขอบ / หันออกเร็วกลับช้า) — แก้โดย "ไม่เคยดึงหรือขยับเคอร์เซอร์เอง"
  แต่ปรับความเร็วของการขยับที่ผู้ใช้ทำอยู่แล้ว ทีละแกน:
    หันเข้าหาตำแหน่งที่หน้าหันไป → เร็วขึ้น (สูงสุด ×(1+BOOST_MAX))
    หันออกห่างจากตำแหน่งนั้น      → ช้าลง (ต่ำสุด ×(1−DAMP_MAX))
    ห่างไม่เกิน MOD_BAND_PX        → ไม่ปรับเลย (เล็งละเอียดแบบ relative ล้วน)
  → เคอร์เซอร์ขยับ "ทิศเดียวกับหัวเสมอ" ไม่มีการยึกกลับ ไม่ขยับเองตอนหัวนิ่ง แต่ความคลาดค่อยๆ หายเอง

"ตำแหน่งที่หน้าหันไป":
  Hybrid + มีโมเดล ML (Calibrate + Train Cursor) → ใช้ค่าที่โมเดล regression ทำนาย
  ไม่มีโมเดล → แมปเชิงเส้นจากมุมหน้าตรง (Baseline) + ช่วงที่ผู้ใช้หันจริง (เรียนระหว่างใช้งาน)

อินพุตเป็นมุมศีรษะ (องศา) จาก facial transformation matrix: yaw + = ขวา, pitch + = ก้ม
"""
import math

from utils.one_euro_filter import OneEuroFilter

# ── Pointer acceleration (px ต่อองศา) ──
GAIN_SLOW = 12.0          # หมุนช้ากว่า V_SLOW
GAIN_FAST = 85.0          # หมุนเร็วกว่า V_FAST
V_SLOW = 4.0              # องศา/วินาที
V_FAST = 45.0
ACCEL_EXP = 1.5           # รูปโค้งระหว่าง slow → fast (>1 = ช่วงต้นเพิ่มช้า คุมละเอียดง่าย)
VERTICAL_BOOST = 1.25     # ก้ม/เงยได้น้อยกว่าหันซ้ายขวา → แนวตั้งขยายมากกว่าเล็กน้อย

# ── ล็อกตอนนิ่ง ──
LOCK_RATIO = 0.6          # รัศมีล็อก = Stability (px) × ค่านี้
PENDING_DECAY = 0.95      # การขยับที่ยังไม่พ้นรัศมีล็อกค่อยๆ หาย (ส่ายหัวไปมาจึงไม่สะสม)
V_STILL = 2.5             # องศา/วินาที — ช้ากว่านี้ต่อเนื่อง STILL_S = หยุดแล้ว → ล็อก
STILL_S = 0.15
# ระหว่างล็อก: หมุนช้ากว่า SWAY_V_LO ไม่นับเลย (ส่ายหัว/หายใจ ~1–2 องศา/วินาที) → นับเต็มที่ SWAY_V_HI
SWAY_V_LO = 1.0
SWAY_V_HI = 3.0
VEL_WINDOW_S = 0.10       # วัดความเร็วจากช่วงนี้ (กัน noise เฟรมเดียว)

# ── Gain modulation (กันหลุดจากทิศหน้า โดยไม่ยึก) ──
# (band, range): ห่างไม่เกิน band = ไม่ปรับ · ห่างเกิน band + range = ปรับเต็มที่
MOD_ML = (60.0, 120.0)    # มีโมเดล ML (คลาดเฉลี่ย ~80 px) → คุมแน่น
MOD_AUTO = (80.0, 150.0)  # แมปจากมุมหน้าตรง (หยาบกว่า) → ยอมคลาดกว้างกว่า
BOOST_PER_RANGE = 1.6     # หันเข้าหา → gain × (1 + ค่านี้ × ระยะเกิน band / range)
BOOST_MAX = 4.0           #   สูงสุด ×5 (ยิ่งตามหลังทิศหน้ามาก ยิ่งไล่ทันเร็ว)
DAMP_MAX = 1.0            # หันออกจนห่างเต็ม range → หยุดรอให้หน้าตามทัน (ไม่ถอยกลับ)
SUBSTEP_PX = 8.0          # แบ่งการขยับต่อเฟรมเป็นช่วงย่อย (หันเร็วเฟรมเดียวไปได้หลายร้อย px)
TARGET_SMOOTH = 0.4       # EMA ของตำแหน่งจากโมเดล ML (กรอง noise แต่ไม่หน่วงตามหัวไม่ทัน)

# ── ตำแหน่งที่หน้าหันไป เมื่อไม่มีโมเดล ML ──
NEUTRAL_INIT_S = 0.5      # ไม่มี Baseline: ค่ากลางของมุมช่วงแรกหลังเริ่ม = ท่าหน้าตรง
NEUTRAL_TAU_S = 30.0      # มุมหน้าตรงค่อยๆ ปรับตามท่านั่ง (time constant)
RANGE_INIT = (12.0, 8.0)  # หันจากหน้าตรงกี่องศาถึงขอบจอ (ซ้าย-ขวา, บน-ล่าง) — เรียนต่อระหว่างใช้
RANGE_MIN = (6.0, 4.0)
RANGE_MAX = (35.0, 25.0)
RANGE_GROW = 0.05         # หันไกลกว่าช่วงเดิม → ขยายเร็ว
RANGE_SHRINK_TAU_S = 120.0  # ไม่ได้หันไกลนานๆ → หดช้าๆ


class HeadPointer:
    def __init__(self, screen_w, screen_h):
        self.w, self.h = int(screen_w), int(screen_h)
        self.speed = 1.0
        self.lock_px = 24 * LOCK_RATIO
        self._fx = OneEuroFilter(min_cutoff=1.5, beta=0.3, d_cutoff=1.0)
        self._fy = OneEuroFilter(min_cutoff=1.5, beta=0.3, d_cutoff=1.0)
        self.pos = (self.w / 2.0, self.h / 2.0)
        self._neutral = None          # มุมหน้าตรง (Baseline หรือเรียนเอง) — ไม่ล้างตอน resync
        self._neutral_init = []
        self._range = list(RANGE_INIT)
        self.resync()

    # ── ตั้งค่า ──
    def set_speed(self, speed):
        """ตัวคูณ gain (Cursor Speed ใน Settings) — น้อย = ต้องหันหัวมากขึ้น"""
        self.speed = max(0.2, min(3.0, float(speed)))

    def set_stability(self, px):
        """รัศมีล็อกตอนนิ่ง (Stability ใน Settings) — 0 = ปิด"""
        self.lock_px = max(0.0, float(px)) * LOCK_RATIO

    def set_smoothing(self, level):
        """ระดับกรองมุมศีรษะ (Smoothing 1–15 ใน Settings) — มาก = นิ่งขึ้นแต่หน่วงขึ้น"""
        cutoff = 3.0 / (1.0 + 0.3 * (max(1, min(15, int(level))) - 1))
        self._fx.min_cutoff = self._fy.min_cutoff = cutoff

    def set_position(self, x, y):
        self.pos = (self._clamp(x, self.w), self._clamp(y, self.h))
        self._pending = (0.0, 0.0)

    def set_neutral(self, yaw, pitch):
        """มุมหน้าตรงจาก Baseline (นั่งหน้าตรงมองกลางจอ) — ระหว่างใช้งานยังปรับตามท่านั่งช้าๆ"""
        self._neutral = (float(yaw), float(pitch))
        self._neutral_init = []

    def reset_neutral(self):
        """ไม่มี Baseline: เรียนท่าหน้าตรงจากช่วงแรกหลังเริ่มใช้งานแทน"""
        self._neutral = None
        self._neutral_init = []

    def resync(self):
        """ลืมมุมก่อนหน้า (หลังโหมด scroll / หน้าหาย / หยุดชั่วคราว)
        → เฟรมถัดไปใช้เป็นจุดตั้งต้นใหม่ การขยับระหว่างนั้นจึงไม่ทำให้เคอร์เซอร์กระโดด"""
        self._fx.reset()
        self._fy.reset()
        self._prev = None
        self._hist = []
        self._pending = (0.0, 0.0)
        self._moving = False
        self._slow_since = None
        self._target = None
        self.velocity = 0.0
        self.expected = None          # ตำแหน่งที่หน้าหันไป (ไว้ดู/debug)

    # ── ทุกเฟรม ──
    def update(self, t, yaw, pitch, target=None):
        """
        Args:
            t: เวลา (วินาที)
            yaw, pitch: มุมศีรษะ (องศา)
            target: (x, y) ตำแหน่งที่โมเดล ML ทำนายว่าหน้าหันไป (โหมด hybrid) หรือ None

        Returns:
            (x, y) ตำแหน่งเคอร์เซอร์
        """
        a = self._fx(yaw, t)
        b = self._fy(pitch, t)
        self._learn_neutral(t, a, b)
        self._smooth_target(target)
        if self._prev is None:
            self._prev = (a, b)
            self._hist = [(t, a, b)]
            return self.pos
        da, db = a - self._prev[0], b - self._prev[1]
        self._prev = (a, b)

        # ความเร็วเชิงมุมจากหน้าต่างสั้นๆ
        self._hist.append((t, a, b))
        while len(self._hist) > 2 and t - self._hist[0][0] > VEL_WINDOW_S:
            self._hist.pop(0)
        t0, a0, b0 = self._hist[0]
        v = math.hypot(a - a0, b - b0) / (t - t0) if t > t0 else 0.0
        self.velocity = v

        # pointer acceleration
        k = min(1.0, max(0.0, (v - V_SLOW) / (V_FAST - V_SLOW))) ** ACCEL_EXP
        gain = (GAIN_SLOW + (GAIN_FAST - GAIN_SLOW) * k) * self.speed
        dx, dy = gain * da, gain * db * VERTICAL_BOOST

        x, y = self.pos
        if not self._moving:
            # การขยับช้ามาก (ส่ายตามธรรมชาติ) ไม่สะสม — ต้องหมุน "ตั้งใจ" ถึงจะหลุดล็อก
            sway = min(1.0, max(0.0, (v - SWAY_V_LO) / (SWAY_V_HI - SWAY_V_LO)))
            px, py = self._pending
            px, py = px * PENDING_DECAY + dx * sway, py * PENDING_DECAY + dy * sway
            if self.lock_px <= 0 or math.hypot(px, py) > self.lock_px:
                self._moving = True
                self._slow_since = None
                x, y = x + px, y + py
                px = py = 0.0
            self._pending = (px, py)
        else:
            if v >= V_STILL:
                self._learn_range(a, b)
                dx, dy = self._modulate(dx, dy, a, b)
            x, y = x + dx, y + dy
            if v < V_STILL:
                if self._slow_since is None:
                    self._slow_since = t
                elif t - self._slow_since >= STILL_S:
                    self._moving = False
                    self._pending = (0.0, 0.0)
            else:
                self._slow_since = None

        self.pos = (self._clamp(x, self.w), self._clamp(y, self.h))
        return self.pos

    # ── กันหลุดจากทิศหน้า ──
    def _modulate(self, dx, dy, a, b):
        """ปรับความเร็วทีละแกนตามว่ากำลังหันเข้าหา/ออกจากตำแหน่งที่หน้าหันไป

        คืนระยะที่ขยับจริง — ทิศเดียวกับ (dx, dy) เสมอ (หรือ 0) ไม่มีทางสวนทิศหัว
        """
        expected = self._expected_position(a, b)
        self.expected = expected
        if expected is None:
            return dx, dy
        band, rng = MOD_ML if self._target is not None else MOD_AUTO
        out = []
        for d, pos, exp in ((dx, self.pos[0], expected[0]), (dy, self.pos[1], expected[1])):
            n = max(1, int(math.ceil(abs(d) / SUBSTEP_PX)))
            step, moved = d / n, 0.0
            for _ in range(n):
                err = exp - (pos + moved)
                excess = max(0.0, (abs(err) - band) / rng)
                if (step > 0) == (err > 0):                       # หันเข้าหา → เร็วขึ้น
                    moved += step * (1.0 + min(BOOST_MAX, BOOST_PER_RANGE * excess))
                else:                                             # หันออก → ช้าลง/หยุดรอ
                    moved += step * (1.0 - DAMP_MAX * min(1.0, excess))
            out.append(moved)
        return out[0], out[1]

    def _smooth_target(self, target):
        if target is None:
            self._target = None
            return
        tx, ty = self._clamp(target[0], self.w), self._clamp(target[1], self.h)
        if self._target is None:
            self._target = (tx, ty)
        else:
            sx, sy = self._target
            self._target = (sx + TARGET_SMOOTH * (tx - sx), sy + TARGET_SMOOTH * (ty - sy))

    def _expected_position(self, a, b):
        """โมเดล ML (ถ้ามี) หรือแมปเชิงเส้นจากมุมหน้าตรง + ช่วงที่ผู้ใช้หันจริง"""
        if self._target is not None:
            return self._target
        if self._neutral is None:
            return None
        ex = self.w / 2.0 + (self.w / 2.0) / self._range[0] * (a - self._neutral[0])
        ey = self.h / 2.0 + (self.h / 2.0) / self._range[1] * (b - self._neutral[1])
        return self._clamp(ex, self.w), self._clamp(ey, self.h)

    def _learn_neutral(self, t, a, b):
        if self._neutral is None:
            self._neutral_init.append((t, a, b))
            if t - self._neutral_init[0][0] >= NEUTRAL_INIT_S:
                ys = sorted(s[1] for s in self._neutral_init)
                ps = sorted(s[2] for s in self._neutral_init)
                self._neutral = (ys[len(ys) // 2], ps[len(ps) // 2])
                self._neutral_init = []
            return
        dt = 1.0 / 30.0
        if len(self._hist) > 1:
            dt = max(1e-3, min(0.2, self._hist[-1][0] - self._hist[-2][0]))
        alpha = dt / NEUTRAL_TAU_S
        n0, n1 = self._neutral
        self._neutral = (n0 + alpha * (a - n0), n1 + alpha * (b - n1))

    def _learn_range(self, a, b):
        """ช่วงการหันหัวของผู้ใช้: หันไกลกว่าเดิม → ขยายเร็ว, ไม่ได้หันไกลนานๆ → หดช้าๆ"""
        if self._neutral is None:
            return
        shrink = 1.0 - (1.0 / 30.0) / RANGE_SHRINK_TAU_S
        for i, dev in enumerate((abs(a - self._neutral[0]), abs(b - self._neutral[1]))):
            r = self._range[i] * shrink
            if dev > r:
                r += RANGE_GROW * (dev - r)
            self._range[i] = min(RANGE_MAX[i], max(RANGE_MIN[i], r))

    @staticmethod
    def _clamp(v, n):
        return max(0.0, min(float(n - 1), float(v)))
