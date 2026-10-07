"""
EYE ORDER COME AI — Gesture Detector (Event-based state machine)
ตรวจจับท่าทางคลิกจากสัญญาณ EAR / ΔSmile ตามเวลาจริง (วินาที ไม่ผูกกับ FPS)

ท่าทาง → คำสั่ง:
  ขยิบตาซ้าย 0.25–1.10 วิ แล้วลืมตา   → left_click
  ขยิบตาขวา 0.25–1.10 วิ แล้วลืมตา   → right_click
  กะพริบตา (แน่นๆ) 2 ครั้งติดกัน        → double_click
  หลับตาข้างเดียวค้าง ≥ 1.2 วิ         → drag_toggle (กดค้าง/ปล่อย)
  ยิ้มกว้างค้าง ≥ 0.5 วิ (ลืมตาทั้งสอง) → scroll_toggle
  กะพริบตาปกติ / หลับตาพัก             → เพิกเฉย

หลักการ (ทำไมแม่นกว่าการดูทีละเฟรม):
  1. ratio = EAR ÷ "EAR ตอนลืมตา" ที่เรียนต่อเนื่องจากช่วงลืมตาล่าสุด (median 1.5 วิ)
     → ก้ม/เงย/หันหน้า/แสงเปลี่ยนแล้ว EAR ลดลงทั้งสองตา ไม่ถูกนับเป็นหลับตา
  2. ตัดสินทั้ง "episode" (ตั้งแต่เริ่มหลับจนลืมตาครบ) แทนการดูทีละเฟรม
  3. ขยิบ vs กะพริบ ดูจาก "ความต่างของสองตา" ไม่ใช่ว่าอีกตาต้องเปิดเต็ม
     (MediaPipe มักลากตาที่เปิดอยู่ให้หรี่ตามเวลาขยิบ ทำให้ขยิบเดิมกลายเป็นกะพริบ)
  4. hysteresis + ต้องลืมตาต่อเนื่อง open_confirm_s ถึงจบ episode (กันค่ากระตุกเฟรมเดียว)

พารามิเตอร์ทั้งหมดอยู่ใน config/tuning.py (GestureParams) และจูนจากข้อมูลจริงได้
"""
import os
import sys
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.tuning import GestureParams

# ช่วง EAR ที่เป็นไปได้ (กันค่าอ้างอิงพังจาก landmark ที่เพี้ยนชั่วขณะ)
REF_MIN, REF_MAX = 0.05, 0.80


def _percentile(values, q):
    s = sorted(values)
    return s[int(round(q * (len(s) - 1)))]


class _Episode:
    """ช่วงที่ตาข้างใดข้างหนึ่งปิด — นับเฟรมว่าเป็นตาซ้ายปิด / ตาขวาปิด / ปิดทั้งคู่"""
    __slots__ = ("start", "open_since", "last_closed", "n", "n_l", "n_r", "min_hi", "drag_fired")

    def __init__(self, start):
        self.start = start
        self.open_since = None
        self.last_closed = start       # เฟรมล่าสุดที่ต่ำกว่า close_ratio จริง
        self.n = self.n_l = self.n_r = 0
        self.min_hi = float("inf")     # ตาที่เปิดกว้างกว่าลงต่ำสุดแค่ไหน (= ความลึกของการกะพริบ)
        self.drag_fired = False

    def add(self, rl, rr, p):
        self.n += 1
        if rr - rl >= p.wink_asym and rl < p.open_ratio:
            self.n_l += 1
        elif rl - rr >= p.wink_asym and rr < p.open_ratio:
            self.n_r += 1
        self.min_hi = min(self.min_hi, max(rl, rr))

    def side(self, frac):
        """'L' / 'R' ถ้าเป็นขยิบตาข้างเดียว, None ถ้าเป็นกะพริบ (สองตาพร้อมกัน)"""
        if self.n == 0:
            return None
        if self.n_l >= frac * self.n:
            return "L"
        if self.n_r >= frac * self.n:
            return "R"
        return None


class GestureDetector:
    """Feed ค่า EAR/ΔSmile ทีละเฟรมด้วย update() แล้วรับ list ของ event กลับมา"""

    def __init__(self, ref_ear_l=0.28, ref_ear_r=0.28, params=None):
        self.p = params or GestureParams()
        self._ref_buf = deque()
        self.set_reference(ref_ear_l, ref_ear_r, trusted=True)
        self.reset()

    @staticmethod
    def _clamp_ref(v):
        return max(REF_MIN, min(REF_MAX, float(v)))

    def set_params(self, params):
        self.p = params

    def set_reference(self, ear_l, ear_r, trusted=True):
        """ตั้งค่า EAR ตอนลืมตาเริ่มต้น

        Args:
            trusted: False = ค่านี้แค่ประมาณ (เช่น ไม่มี baseline) → ยังไม่ตรวจท่าทาง
                     จนกว่าจะเรียนค่าจริงจากกล้องได้ ~0.3 วิ
        """
        self.ref_l = self._clamp_ref(ear_l)
        self.ref_r = self._clamp_ref(ear_r)
        self._ref_buf.clear()
        self._ref_ready = bool(trusted)

    def reset(self):
        """ล้างสถานะท่าทาง (เรียกเมื่อหน้าหายหรือเริ่มใหม่) — ไม่ล้างค่าอ้างอิงตาเปิดที่เรียนไว้"""
        self._ep = None
        self._last_blink_end = None
        self._smile_start = None
        self._smile_armed = True
        self._calm_since = None
        self._last_event_t = -1e9
        self._frozen = False
        self.ratio_l = 1.0
        self.ratio_r = 1.0
        self.delta_smile = 0.0

    @property
    def freeze_cursor(self):
        """True ระหว่างที่กำลังทำท่าทาง → pipeline ควรตรึงเคอร์เซอร์ไว้ที่เดิม"""
        return self._frozen

    @property
    def smiling(self):
        """True ระหว่างยิ้ม (กำลังสลับโหมด scroll) — pipeline ใช้หยุดเลื่อนชั่วคราว"""
        return self._smile_start is not None or self.delta_smile > self.p.smile_freeze

    @property
    def state(self):
        if self._ep is not None:
            side = self._ep.side(self.p.wink_side_frac)
            return f"wink_{side}" if side else "blink"
        if self._smile_start is not None:
            return "smile"
        if not self._ref_ready:
            return "learning"
        return "idle"

    # ──────────────────────────────────────────────
    def update(self, t, ear_l, ear_r, delta_smile):
        """ประมวลผล 1 เฟรม

        Args:
            t: เวลา (วินาที, monotonic)
            ear_l, ear_r: EAR ของตาซ้าย/ขวา (ตามผู้ใช้จริง — สลับก่อนส่งเข้ามาถ้าไม่ mirror)
            delta_smile: สัดส่วนปากกว้างขึ้น (0.2 = 20%)

        Returns:
            list[str]: event ที่เกิดในเฟรมนี้ (ส่วนใหญ่ว่าง)
        """
        p = self.p
        events = []

        rl = ear_l / self.ref_l
        rr = ear_r / self.ref_r
        self.ratio_l, self.ratio_r = rl, rr
        self.delta_smile = delta_smile
        lo = min(rl, rr)

        # ── episode การหลับตา ──
        ep = self._ep
        if ep is None and self._ref_ready and lo < p.close_ratio:
            ep = self._ep = _Episode(t)
        if ep is not None:
            if lo >= p.open_ratio:
                if ep.open_since is None:
                    ep.open_since = t
                if t - ep.open_since >= p.open_confirm_s:
                    events += self._finish_episode(ep)
                    self._ep = ep = None
            else:
                ep.open_since = None
                ep.add(rl, rr, p)
                if lo < p.close_ratio:
                    ep.last_closed = t
                held = t - ep.start
                if (not ep.drag_fired and held >= p.drag_hold_s
                        and ep.side(p.wink_side_frac)):
                    ep.drag_fired = True
                    events.append("drag_toggle")
                if t - ep.last_closed > p.half_open_abort_s and not ep.drag_fired:
                    # ค้างครึ่งๆ (ไม่ปิดจริง ไม่เปิดเต็ม) = ท่าศีรษะเปลี่ยน ไม่ใช่ท่าทาง → ยกเลิก
                    self._ep = ep = None
                elif held > p.episode_timeout_s:
                    # หลับตานานเกินท่าทางใดๆ → น่าจะเป็นท่าศีรษะ/แสงเปลี่ยน เรียนค่าตาเปิดใหม่
                    self._ref_buf.clear()
                    self.ref_l = self._clamp_ref(ear_l)
                    self.ref_r = self._clamp_ref(ear_r)
                    self._ep = ep = None

        # ── เรียนค่า EAR ตาเปิด เฉพาะเฟรมที่ไม่ได้หลับตา ──
        if ep is None:
            self._update_reference(t, ear_l, ear_r)

        # ── ตรึงเคอร์เซอร์ ──
        busy = (lo < p.freeze_ratio and self._ref_ready) or ep is not None \
            or delta_smile > p.smile_freeze
        if busy:
            self._frozen = True
            self._calm_since = None
        elif self._frozen:
            if self._calm_since is None:
                self._calm_since = t
            elif t - self._calm_since >= p.release_s:
                self._frozen = False

        # ── ยิ้ม (ต้องลืมตาทั้งสองข้าง ไม่งั้นแยกไม่ออกจากขยิบ) ──
        if delta_smile >= p.smile_on and ep is None:
            if self._smile_start is None and self._smile_armed:
                self._smile_start = t
            elif self._smile_start is not None and t - self._smile_start >= p.smile_hold_s:
                events.append("scroll_toggle")
                self._smile_start = None
                self._smile_armed = False
        else:
            self._smile_start = None
            if delta_smile < p.smile_off:
                self._smile_armed = True

        # ── cooldown: ปล่อยแค่ event แรก กันสั่งซ้ำ ──
        if events:
            if t - self._last_event_t < p.event_cooldown_s:
                return []
            self._last_event_t = t
            events = events[:1]
        return events

    def _finish_episode(self, ep):
        """ลืมตาครบแล้ว → ตัดสินจากระยะเวลา + ข้างที่ปิด"""
        p = self.p
        end = ep.open_since
        dur = end - ep.start
        if ep.drag_fired:
            self._last_blink_end = None
            return []

        side = ep.side(p.wink_side_frac)
        if side:
            self._last_blink_end = None
            if p.wink_min_s <= dur <= p.wink_max_s:
                return ["left_click" if side == "L" else "right_click"]
            return []

        if dur <= p.blink_max_s and ep.min_hi <= p.blink_depth:
            if (self._last_blink_end is not None
                    and ep.start - self._last_blink_end <= p.double_blink_gap_s):
                self._last_blink_end = None
                return ["double_click"]
            self._last_blink_end = end
        else:
            self._last_blink_end = None
        return []

    def _update_reference(self, t, ear_l, ear_r):
        p = self.p
        buf = self._ref_buf
        buf.append((t, ear_l, ear_r))
        while buf and t - buf[0][0] > p.ref_window_s:
            buf.popleft()
        if len(buf) >= p.ref_min_samples:
            self.ref_l = self._clamp_ref(_percentile([b[1] for b in buf], p.ref_percentile))
            self.ref_r = self._clamp_ref(_percentile([b[2] for b in buf], p.ref_percentile))
            self._ref_ready = True
