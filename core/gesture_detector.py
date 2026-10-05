"""
EYE ORDER COME AI — Gesture Detector (Event-based state machine)
ตรวจจับท่าทางคลิกจากสัญญาณ EAR / ΔSmile ตามเวลาจริง (วินาที ไม่ผูกกับ FPS)

ท่าทาง → คำสั่ง:
  ขยิบตาซ้าย 0.30–1.10 วิ แล้วลืมตา   → left_click
  ขยิบตาขวา 0.30–1.10 วิ แล้วลืมตา   → right_click
  กะพริบตา 2 ครั้งติดกัน               → double_click
  หลับตาข้างเดียวค้าง ≥ 1.2 วิ         → drag_toggle (กดค้าง/ปล่อย)
  ยิ้มกว้างค้าง ≥ 0.5 วิ (ลืมตาทั้งสอง) → scroll_toggle
  กะพริบตาปกติ (2 ตาปิดพร้อมกัน)       → เพิกเฉย

ความแม่นยำมาจาก 3 อย่าง:
  1. ใช้ EAR "เทียบกับตาเปิดของผู้ใช้" (adaptive baseline) ไม่ใช่ค่าคงที่
  2. hysteresis (ปิด < 0.62, เปิด > 0.78) กันค่าสั่นรอบ threshold
  3. ตัดสินด้วยระยะเวลาจริง: กะพริบปกติ (~0.1–0.15 วิ) ไม่มีวันถึง 0.30 วิ
     และถ้าอีกตาปิดตามมาภายหลังจะถือเป็นกะพริบ ไม่ใช่ขยิบ
"""

# ── Thresholds (สัดส่วนเทียบกับ EAR ตอนลืมตา) ──
CLOSE_RATIO = 0.62        # ต่ำกว่านี้ = ตาปิด
OPEN_RATIO = 0.78         # สูงกว่านี้ = ตาเปิด
FREEZE_RATIO = 0.80       # ต่ำกว่านี้ = เริ่มหลับตา → ตรึงเคอร์เซอร์
SMILE_ON = 0.20           # ΔSmile เริ่มนับว่ายิ้ม
SMILE_OFF = 0.12          # ต่ำกว่านี้ถึงจะพร้อมยิ้มรอบใหม่
SMILE_FREEZE = 0.10       # ยิ้มเกินนี้ → มุมปากขยับทำให้ head pose เพี้ยน → ตรึงเคอร์เซอร์

# ── Timing (วินาที) ──
WINK_MIN_S = 0.30
WINK_MAX_S = 1.10
DRAG_HOLD_S = 1.20
BLINK_MAX_S = 0.40        # ปิดทั้งสองตานานกว่านี้ไม่ใช่กะพริบ (เช่น หลับตาพัก)
DOUBLE_BLINK_GAP_S = 0.80
SMILE_HOLD_S = 0.50
RELEASE_S = 0.12          # ต้องลืมตา/หยุดยิ้มต่อเนื่องเท่านี้ถึงปล่อยการตรึงเคอร์เซอร์
EVENT_COOLDOWN_S = 0.35

# ── Adaptive baseline ──
REF_ALPHA = 0.02
REF_MIN, REF_MAX = 0.15, 0.45


class GestureDetector:
    """Feed ค่า EAR/ΔSmile ทีละเฟรมด้วย update() แล้วรับ list ของ event กลับมา"""

    def __init__(self, ref_ear_l=0.30, ref_ear_r=0.30):
        self._init_ref_l = self._clamp_ref(ref_ear_l)
        self._init_ref_r = self._clamp_ref(ref_ear_r)
        self.reset()

    @staticmethod
    def _clamp_ref(v):
        return max(REF_MIN, min(REF_MAX, float(v)))

    def set_reference(self, ear_l, ear_r):
        self._init_ref_l = self._clamp_ref(ear_l)
        self._init_ref_r = self._clamp_ref(ear_r)
        self.ref_l, self.ref_r = self._init_ref_l, self._init_ref_r

    def reset(self):
        """ล้างสถานะ (เรียกเมื่อหน้าหายหรือเริ่มใหม่) — ไม่ล้าง adaptive baseline"""
        self.ref_l = getattr(self, "ref_l", self._init_ref_l)
        self.ref_r = getattr(self, "ref_r", self._init_ref_r)
        self._closed_l = False
        self._closed_r = False
        self._both_start = None
        self._blink_times = []
        self._wink_side = None
        self._wink_start = 0.0
        self._drag_fired = False
        self._wink_blocked = False      # เพิ่งปิดสองตา แล้วตาหนึ่งเปิดก่อน ≠ ขยิบ
        self._smile_start = None
        self._smile_armed = True
        self._calm_since = None
        self._last_event_t = -1e9
        self._frozen = False
        self.ratio_l = 1.0
        self.ratio_r = 1.0

    @property
    def freeze_cursor(self):
        """True ระหว่างที่กำลังทำท่าทาง → pipeline ควรตรึงเคอร์เซอร์ไว้ที่เดิม"""
        return self._frozen

    @property
    def state(self):
        if self._wink_side:
            return f"wink_{self._wink_side}"
        if self._both_start is not None:
            return "blink"
        if self._smile_start is not None:
            return "smile"
        return "idle"

    def update(self, t, ear_l, ear_r, delta_smile):
        """ประมวลผล 1 เฟรม

        Args:
            t: เวลา (วินาที, monotonic)
            ear_l, ear_r: EAR ของตาซ้าย/ขวา (ตามผู้ใช้จริง — สลับก่อนส่งเข้ามาถ้าไม่ mirror)
            delta_smile: สัดส่วนปากกว้างขึ้น (0.2 = 20%)

        Returns:
            list[str]: event ที่เกิดในเฟรมนี้ (ส่วนใหญ่ว่าง)
        """
        events = []

        # ── อัปเดตค่าอ้างอิงตาเปิดแบบค่อยๆ ปรับ (เฉพาะตอนที่ตาเปิดอยู่จริง) ──
        rl = ear_l / self.ref_l
        rr = ear_r / self.ref_r
        if rl > 0.85 and rr > 0.85 and not self._closed_l and not self._closed_r:
            self.ref_l = self._clamp_ref(self.ref_l + REF_ALPHA * (ear_l - self.ref_l))
            self.ref_r = self._clamp_ref(self.ref_r + REF_ALPHA * (ear_r - self.ref_r))
            rl = ear_l / self.ref_l
            rr = ear_r / self.ref_r
        self.ratio_l, self.ratio_r = rl, rr

        # ── ตาปิด/เปิดแบบ hysteresis ──
        self._closed_l = (rl < OPEN_RATIO) if self._closed_l else (rl < CLOSE_RATIO)
        self._closed_r = (rr < OPEN_RATIO) if self._closed_r else (rr < CLOSE_RATIO)
        cl, cr = self._closed_l, self._closed_r

        # ── ตรึงเคอร์เซอร์ ──
        busy = (min(rl, rr) < FREEZE_RATIO) or cl or cr or (delta_smile > SMILE_FREEZE)
        if busy:
            self._frozen = True
            self._calm_since = None
        elif self._frozen:
            if self._calm_since is None:
                self._calm_since = t
            elif t - self._calm_since >= RELEASE_S:
                self._frozen = False

        # ── กะพริบ / ขยิบ ──
        if cl and cr:
            # สองตาปิดพร้อมกัน = กะพริบ → ยกเลิก wink ที่อาจเริ่มเพราะตาหนึ่งปิดก่อนเสี้ยววินาที
            self._wink_side = None
            self._drag_fired = False
            if self._both_start is None:
                self._both_start = t
        else:
            if self._both_start is not None:
                dur = t - self._both_start
                self._both_start = None
                if dur <= BLINK_MAX_S:
                    events += self._register_blink(t)
                if cl or cr:
                    # ตาหนึ่งยังปิดค้างหลังกะพริบ → ไม่นับเป็นขยิบ จนกว่าจะลืมตาครบ
                    self._wink_blocked = True

            if not cl and not cr:
                self._wink_blocked = False

            side = None
            if cl and not cr:
                side = "L"
            elif cr and not cl:
                side = "R"

            if side != self._wink_side:
                # ขยิบเดิมจบลง (ลืมตา) → ตัดสินจากระยะเวลา
                if self._wink_side and not self._drag_fired and side is None:
                    held = t - self._wink_start
                    if WINK_MIN_S <= held <= WINK_MAX_S:
                        events.append("left_click" if self._wink_side == "L" else "right_click")
                self._drag_fired = False
                self._wink_side = side if (side and not self._wink_blocked) else None
                self._wink_start = t

            if self._wink_side and not self._drag_fired:
                if t - self._wink_start >= DRAG_HOLD_S:
                    events.append("drag_toggle")
                    self._drag_fired = True

        # ── ยิ้ม (ต้องลืมตาทั้งสองข้าง ไม่งั้นแยกไม่ออกจากขยิบ) ──
        if delta_smile >= SMILE_ON and not cl and not cr:
            if self._smile_start is None and self._smile_armed:
                self._smile_start = t
            elif self._smile_start is not None and t - self._smile_start >= SMILE_HOLD_S:
                events.append("scroll_toggle")
                self._smile_start = None
                self._smile_armed = False
        else:
            self._smile_start = None
            if delta_smile < SMILE_OFF:
                self._smile_armed = True

        # ── cooldown: ปล่อยแค่ event แรก กันสั่งซ้ำ ──
        if events:
            if t - self._last_event_t < EVENT_COOLDOWN_S:
                return []
            self._last_event_t = t
            events = events[:1]
            self._blink_times.clear()
        return events

    def _register_blink(self, t):
        self._blink_times = [bt for bt in self._blink_times if t - bt <= DOUBLE_BLINK_GAP_S]
        self._blink_times.append(t)
        if len(self._blink_times) >= 2:
            self._blink_times.clear()
            return ["double_click"]
        return []
