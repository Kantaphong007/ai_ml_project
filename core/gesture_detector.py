"""
EYE ORDER COME AI — Gesture Detector (Event-based state machine)
ตรวจจับท่าทางคลิกจากสัญญาณ EAR / ΔSmile ตามเวลาจริง (วินาที ไม่ผูกกับ FPS)

ท่าทาง → คำสั่ง:
  ขยิบตาซ้าย 0.25–1.10 วิ แล้วลืมตา   → left_click
  ขยิบตาขวา 0.25–1.10 วิ แล้วลืมตา   → right_click
  กะพริบ/ขยิบเร็วๆ 2 ครั้งติดกัน         → double_click ตรวจได้ 2 แบบ:
    - ตาปิด 2 รอบในการหลับตาช่วงเดียว (ปิด → เปิดขึ้นครึ่งทาง → ปิด = รูปตัว W)
    - ตาปิดสั้นๆ แบบเดียวกัน 2 ครั้งติดกัน (ขยิบข้างเดิม / กะพริบแน่นสองตา) ห่างไม่เกิน double_blink_gap_s
  หลับตาข้างเดียวค้าง ≥ 1.2 วิ         → drag_start (กดเมาส์ค้าง) — ลากได้ขณะยังหลับตา
  ลืมตาข้างนั้น (ระหว่าง drag)           → drag_end (ปล่อยเมาส์)
  ยิ้มกว้าง (ลืมตาทั้งสอง) ค้างสั้นๆ      → scroll_start (อยู่ในโหมด scroll ตลอดที่ยังยิ้ม)
  หุบยิ้ม                               → scroll_end
  กะพริบตาปกติ / หลับตาพัก             → เพิกเฉย

หลักการ (ทำไมแม่นกว่าการดูทีละเฟรม):
  1. ratio = EAR ÷ "EAR ตอนลืมตา" ที่เรียนต่อเนื่องจากช่วงลืมตาล่าสุด (median 1.5 วิ)
     → ก้ม/เงย/หันหน้า/แสงเปลี่ยนแล้ว EAR ลดลงทั้งสองตา ไม่ถูกนับเป็นหลับตา
  2. ตัดสินทั้ง "episode" (ตั้งแต่เริ่มหลับจนลืมตาครบ) แทนการดูทีละเฟรม
  3. ขยิบ vs กะพริบ ดูจาก "ความต่างของสองตา" ไม่ใช่ว่าอีกตาต้องเปิดเต็ม
     (MediaPipe มักลากตาที่เปิดอยู่ให้หรี่ตามเวลาขยิบ ทำให้ขยิบเดิมกลายเป็นกะพริบ)
  4. hysteresis + ต้องลืมตาต่อเนื่อง open_confirm_s ถึงจบ episode (กันค่ากระตุกเฟรมเดียว)

โหมดตัดสินท่า:
  - ML (มีโมเดลจาก training/train_click_model.py): rule หาแค่ episode แล้วโมเดลจำแนกท่า
    จากฟีเจอร์ใน core/episode_features.py
  - Rule (ไม่มีโมเดล / fallback): ตัดสินด้วยระยะเวลา + ความต่างของสองตา

พารามิเตอร์ทั้งหมดอยู่ใน config/tuning.py (GestureParams) และจูนจากข้อมูลจริงได้
"""
import os
import sys
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.tuning import GestureParams
from core.episode_features import (
    CLASS_TO_EVENT, compute_features, episode_summary,
)

# ช่วง EAR ที่เป็นไปได้ (กันค่าอ้างอิงพังจาก landmark ที่เพี้ยนชั่วขณะ)
REF_MIN, REF_MAX = 0.05, 0.80


def _percentile(values, q):
    s = sorted(values)
    return s[int(round(q * (len(s) - 1)))]


class _Episode:
    """ช่วงที่ตาข้างใดข้างหนึ่งปิด — นับเฟรมว่าเป็นตาซ้ายปิด / ตาขวาปิด / ปิดทั้งคู่"""
    __slots__ = ("start", "open_since", "last_closed", "n", "n_l", "n_r", "min_hi",
                 "drag_fired", "hold_decided", "frames", "min_lo",
                 "dips", "_trough", "_peak", "_rising", "_lo_hist", "_streak")

    def __init__(self, start):
        self.start = start
        self.frames = []               # (t, ratio_l, ratio_r, delta_smile) สำหรับฟีเจอร์ของโมเดล
        self.hold_decided = False
        self.open_since = None
        self.last_closed = start       # เฟรมล่าสุดที่ต่ำกว่า close_ratio จริง
        self.n = self.n_l = self.n_r = 0
        self.min_hi = float("inf")     # ตาที่เปิดกว้างกว่าลงต่ำสุดแค่ไหน (= ความลึกของการกะพริบ)
        self.min_lo = float("inf")     # ตาที่ปิดมากกว่าลงลึกสุดแค่ไหน
        self.drag_fired = False
        self.dips = 0                  # จำนวนครั้งที่ตาปิดลง (ปิด → เปิดขึ้น ≥ dip_rise → ปิดใหม่ = 2)
        self._trough = None
        self._peak = None
        self._rising = False
        self._lo_hist = []
        self._streak = 0               # จำนวนเฟรมติดกันที่ผ่านเงื่อนไข (เปิดขึ้น / ปิดลงใหม่)

    def add(self, t, rl, rr, smile, p):
        self.frames.append((t, rl, rr, smile))
        self.n += 1
        if rr - rl >= p.wink_asym and rl < p.open_ratio:
            self.n_l += 1
        elif rl - rr >= p.wink_asym and rr < p.open_ratio:
            self.n_r += 1
        self.min_hi = min(self.min_hi, max(rl, rr))
        self.min_lo = min(self.min_lo, min(rl, rr))
        # นับรอบที่ตาปิด: ลงจุดต่ำ → ขึ้นอย่างน้อย dip_rise → ลงใหม่ต่ำกว่า close_ratio = อีกหนึ่งรอบ
        # ใช้ค่าเฉลี่ย 3 เฟรม กัน noise เฟรมเดียวถูกนับเป็นการเปิด-ปิดตา
        self._lo_hist = (self._lo_hist + [min(rl, rr)])[-3:]
        lo = sum(self._lo_hist) / len(self._lo_hist)
        # และต้องคงอยู่ 2 เฟรมติดกันทั้งตอนเปิดขึ้นและตอนปิดใหม่ (noise เฟรมเดียวไม่นับ)
        if self._trough is None:
            if lo < p.close_ratio:
                self._trough, self.dips = lo, 1
        elif not self._rising:
            if lo < self._trough:
                self._trough = lo
            reopened = lo >= self._trough + p.dip_rise and lo >= p.close_ratio - p.dip_reopen_margin
            self._streak = self._streak + 1 if reopened else 0
            if self._streak >= 2:
                self._rising, self._peak, self._streak = True, lo, 0
        else:
            self._peak = max(self._peak, lo)
            self._streak = self._streak + 1 if (lo <= self._peak - p.dip_rise
                                                and lo < p.close_ratio) else 0
            if self._streak >= 2:
                self.dips += 1
                self._rising, self._trough, self._streak = False, lo, 0

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
        self.classifier = None         # มี predict(features) → (class_id, confidence)
        self.episode_hook = None       # callback(kind, start, decision_t, features) — ใช้ตอนสร้าง dataset
        self._ref_buf = deque()
        self.set_reference(ref_ear_l, ref_ear_r, trusted=True)
        self.reset()

    @staticmethod
    def _clamp_ref(v):
        return max(REF_MIN, min(REF_MAX, float(v)))

    def set_params(self, params):
        self.p = params

    def set_classifier(self, classifier):
        """ตั้งโมเดลจำแนกท่า (None = ใช้ rule-based)"""
        self.classifier = classifier

    @property
    def uses_ml(self):
        return self.classifier is not None

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
        self._prev = None              # episode_summary ของ episode ล่าสุด (ฟีเจอร์ดับเบิลคลิก)
        self.last_decision = ""        # เหตุผลการตัดสิน episode ล่าสุด (แสดงบนจอ ไว้ดูว่าทำไมไม่คลิก)
        self._last_short = None        # (ชนิด L/R/B, เวลาจบ) ของการปิดตาสั้นครั้งล่าสุด — ตรวจดับเบิลคลิก
        self._smile_start = None
        self._scrolling = False        # กำลังยิ้มค้างอยู่ในโหมด scroll
        self._smile_low_since = None
        self._calm_since = None
        self._last_event_t = -1e9
        self._frozen = False
        self.ratio_l = 1.0
        self.ratio_r = 1.0
        self.delta_smile = 0.0

    @property
    def dragging(self):
        """True ระหว่างหลับตาค้างลาก (เมาส์ถูกกดอยู่)"""
        return self._ep is not None and self._ep.drag_fired

    @property
    def freeze_cursor(self):
        """True ระหว่างที่กำลังทำท่าทาง → pipeline ควรตรึงเคอร์เซอร์ไว้ที่เดิม"""
        return self._frozen

    @property
    def smiling(self):
        """True ระหว่างยิ้ม (กำลังสลับโหมด scroll) — pipeline ใช้หยุดเลื่อนชั่วคราว"""
        return (self._smile_start is not None or self._scrolling
                or self.delta_smile > self.p.smile_freeze)

    @property
    def scrolling(self):
        """True ระหว่างยิ้มค้างในโหมด scroll"""
        return self._scrolling

    @property
    def state(self):
        if self._ep is not None:
            if self._ep.drag_fired:
                return "drag"
            side = self._ep.side(self.p.wink_side_frac)
            return f"wink_{side}" if side else "blink"
        if self._scrolling:
            return "scroll (ยิ้มค้าง)"
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
                # ปล่อย drag ต้องลืมตานานกว่าปกติเล็กน้อย กัน landmark กระตุกแล้วหลุดกลางทาง
                confirm = p.drag_release_confirm_s if ep.drag_fired else p.open_confirm_s
                if t - ep.open_since >= confirm:
                    events += self._finish_episode(ep)
                    self._ep = ep = None
            else:
                ep.open_since = None
                ep.add(t, rl, rr, delta_smile, p)
                if lo < p.close_ratio:
                    ep.last_closed = t
                held = t - ep.start
                if not ep.hold_decided and held >= p.drag_hold_s:
                    ep.hold_decided = True
                    if self._decide_hold(ep, t):
                        ep.drag_fired = True
                        events.append("drag_start")
                if t - ep.last_closed > p.half_open_abort_s and not ep.hold_decided:
                    if ep.min_lo <= p.deep_close_ratio:
                        # หลับลึกจริงแล้วค่อยๆ ลืม (ไม่ถึง open_ratio) = ท่าทางที่จบแล้ว → ตัดสิน
                        # ณ จังหวะที่ตาเริ่มเปิดพ้น close_ratio (เดิมถูกทิ้ง → ขึ้น wink แต่ไม่คลิก)
                        ep.open_since = ep.last_closed + 1.0 / 30.0
                        events += self._finish_episode(ep)
                    else:
                        # ค้างครึ่งๆ ไม่เคยหลับลึก = ท่าศีรษะเปลี่ยน ไม่ใช่ท่าทาง → ยกเลิก
                        self.last_decision = "ยกเลิก: ตาหรี่ครึ่งๆ (ไม่ได้หลับจริง)"
                    self._ep = ep = None
                elif ep.drag_fired:
                    if held > p.drag_max_s:      # กันเมาส์ค้างถ้าตรวจไม่เจอการลืมตา
                        events.append("drag_end")
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
        if ep is not None and ep.drag_fired:
            busy = False          # กำลังลาก: เคอร์เซอร์ต้องตามหัว
        else:
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

        # ── ยิ้ม = โหมด scroll: ยิ้มค้าง smile_hold_s → เข้า, หุบยิ้ม smile_release_s → ออก ──
        if not self._scrolling:
            # เริ่มต้องลืมตาทั้งสองข้าง ไม่งั้นแยกไม่ออกจากขยิบ (แก้มยกตอนขยิบ)
            if delta_smile >= p.smile_on and ep is None:
                if self._smile_start is None:
                    self._smile_start = t
                elif t - self._smile_start >= p.smile_hold_s:
                    events.append("scroll_start")
                    self._scrolling = True
                    self._smile_start = None
                    self._smile_low_since = None
            else:
                self._smile_start = None
        elif delta_smile < p.smile_off:
            if self._smile_low_since is None:
                self._smile_low_since = t
            elif t - self._smile_low_since >= p.smile_release_s:
                events.append("scroll_end")
                self._scrolling = False
        else:
            self._smile_low_since = None

        # ── event ที่ต้องผ่านเสมอ (ไม่งั้นเมาส์ค้าง/โหมดไม่ตรงกับหน้า) ──
        # double_click ตามหลังคลิกเดี่ยวของขยิบครั้งแรกได้ใกล้กว่า cooldown → ต้องผ่านด้วย
        for must in ("drag_end", "scroll_end", "scroll_start", "double_click"):
            if must in events:
                self._last_event_t = t
                return [must]

        # ── cooldown: ปล่อยแค่ event แรก กันสั่งซ้ำ ──
        if events:
            if t - self._last_event_t < p.event_cooldown_s:
                return []
            self._last_event_t = t
            events = events[:1]
        return events

    # ──────────────────────────────────────────────
    # การตัดสินท่า
    # ──────────────────────────────────────────────
    def _features(self, ep, decision_t, is_hold):
        return compute_features(ep.frames, ep.start, decision_t, self._prev, is_hold)

    def _classify(self, ep, decision_t, is_hold):
        """เรียกโมเดล (ถ้ามี) + hook เก็บ dataset → (class_id, confidence) หรือ None (ไม่มีโมเดล)"""
        if self.classifier is None and self.episode_hook is None:
            return None
        feats = self._features(ep, decision_t, is_hold)
        if self.episode_hook is not None:
            self.episode_hook("hold" if is_hold else "end", ep.start, decision_t, feats)
        if self.classifier is None:
            return None
        return self.classifier.predict(feats)

    def _ml_confident(self, pred):
        return pred is not None and pred[1] >= self.p.ml_min_confidence

    def _decide_hold(self, ep, t):
        """หลับตาค้างถึง drag_hold_s → เป็น drag หรือไม่ (ML มั่นใจ → ใช้ ML, ไม่มั่นใจ → rule)"""
        pred = self._classify(ep, t, is_hold=True)
        if self._ml_confident(pred):
            ok = pred[0] == 4
            self.last_decision = f"ML: {'Drag' if ok else 'ไม่ใช่ drag'} ({pred[1]:.2f})"
            return ok
        ok = ep.side(self.p.wink_side_frac) is not None
        self.last_decision = ("rule: Drag" if ok else "rule: หลับสองตา (ไม่ใช่ drag)") + self._unsure(pred)
        return ok

    @staticmethod
    def _unsure(pred):
        return f" [ML ไม่มั่นใจ {pred[1]:.2f}]" if pred is not None else ""

    def _finish_episode(self, ep):
        """ลืมตาครบแล้ว → ปล่อย drag / ดับเบิลคลิก / ตัดสินท่าเดี่ยว (ML หรือ rule)"""
        dur = ep.open_since - ep.start
        if ep.drag_fired:
            events = ["drag_end"]
            self._last_short = None
        elif not ep.hold_decided and ep.dips >= 2:
            # ตาปิด 2 รอบในช่วงเดียว (เปิดไม่สุดระหว่างกลาง) — จากข้อมูลจริงพบเฉพาะท่าดับเบิลคลิก
            events = ["double_click"]
            self.last_decision = f"Double Click (ตาปิด {ep.dips} รอบใน {dur:.2f}s)"
            self._last_short = None
        else:
            events = self._judge_episode(ep)
            events = self._sequence_double(ep, dur, events)
        self._prev = episode_summary(ep.frames, ep.open_since)
        return events

    def _short_kind(self, ep, dur):
        """ชนิดของการปิดตาสั้นๆ: 'L'/'R' = ขยิบ, 'B' = กะพริบแน่นสองตา, None = ไม่นับ"""
        p = self.p
        if ep.hold_decided or dur > p.quick_wink_max_s:
            return None
        side = self._clear_wink_side(ep) or ep.side(p.wink_side_frac)
        if side:
            return side
        return "B" if ep.min_hi <= p.blink_depth else None

    def _sequence_double(self, ep, dur, events):
        """ปิดตาสั้นๆ แบบเดียวกัน 2 ครั้งติดกัน → ดับเบิลคลิก (ใช้ทั้งตอน ML และ rule ตัดสิน)

        ครั้งแรกเป็นคลิกเดี่ยวไปแล้ว (ขยิบ) หรือไม่ทำอะไร (กะพริบ) → ครั้งที่สองส่ง double_click
        """
        kind = self._short_kind(ep, dur)
        last, self._last_short = self._last_short, ((kind, ep.open_since) if kind else None)
        if (kind and last and last[0] == kind
                and ep.start - last[1] <= self.p.double_blink_gap_s):
            self._last_short = None
            what = {"L": "ขยิบซ้าย", "R": "ขยิบขวา", "B": "กะพริบ"}[kind]
            self.last_decision = f"Double Click ({what}เร็ว 2 ครั้ง)"
            return ["double_click"]
        return events

    def _judge_episode(self, ep):
        """ตัดสิน episode ที่จบแล้ว

        ML มั่นใจ (≥ ml_min_confidence) → ใช้คำตอบ ML
        ML ไม่มั่นใจ / ไม่มีโมเดล     → ใช้ rule (ระยะเวลา + ข้างที่ปิด) เป็นตัวสำรอง
        """
        end = ep.open_since
        dur = end - ep.start
        if ep.hold_decided:
            # ค้างเกิน drag_hold_s แล้ว: ตัดสินไปแล้วที่จุด hold (drag หรือไม่ทำอะไร)
            return []

        pred = self._classify(ep, end, is_hold=False)
        if self._ml_confident(pred):
            cls, conf = pred
            ev = CLASS_TO_EVENT.get(cls)
            if ev == "drag_start":
                ev = None
            names = {None: "ไม่ทำอะไร", "left_click": "Left Click", "right_click": "Right Click",
                     "double_click": "Double Click"}
            self.last_decision = f"ML: {names[ev]} ({conf:.2f}, {dur:.2f}s)"
            return [ev] if ev else []

        events = self._judge_rule(ep, dur, end)
        self.last_decision = f"rule: {self.last_decision}{self._unsure(pred)}"
        return events

    def _clear_wink_side(self, ep):
        """ตาหนึ่งหลับลึก + อีกตาเปิดเกือบปกติตลอด → 'L'/'R' (ขยิบชัดเจน) ไม่งั้น None"""
        p = self.p
        if ep.min_lo > p.deep_close_ratio or ep.min_hi < p.clear_wink_other:
            return None
        asym = sum(rr - rl for _, rl, rr, _ in ep.frames)
        return "L" if asym > 0 else "R"

    def _judge_rule(self, ep, dur, end):
        p = self.p
        clear = self._clear_wink_side(ep)
        side = clear or ep.side(p.wink_side_frac)
        if side:
            min_s = p.clear_wink_min_s if clear else p.wink_min_s
            if min_s <= dur <= p.wink_max_s:
                self.last_decision = f"{'Left' if side == 'L' else 'Right'} Click ({dur:.2f}s)"
                return ["left_click" if side == "L" else "right_click"]
            why = "สั้น" if dur < min_s else "นาน"
            self.last_decision = (f"ขยิบ{why}เกิน ({dur:.2f}s, ต้อง "
                                  f"{min_s:.2f}–{p.wink_max_s:.2f}s)")
            return []

        if dur <= p.blink_max_s and ep.min_hi <= p.blink_depth:
            self.last_decision = "กะพริบ 1 ครั้ง (รอครั้งที่ 2)"
        else:
            self.last_decision = f"กะพริบปกติ ({dur:.2f}s) ไม่ทำอะไร"
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
