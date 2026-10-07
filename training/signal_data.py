"""
EYE ORDER COME AI — Signal Data (โหลดข้อมูลจาก Record Signals + replay + ให้คะแนน)

ใช้ร่วมกันโดย:
  - training/tune_gestures.py     (จูนพารามิเตอร์ rule / scroll)
  - training/train_click_model.py (สร้าง dataset + ประเมินโมเดล ML ระดับ event)
  - training/evaluate.py

การให้คะแนนระดับ event: replay สัญญาณผ่าน GestureDetector ตัวจริง แล้วดูว่าแต่ละท่า
ได้ event ที่ถูกต้อง "ครั้งเดียว" หรือไม่ + นับ event ที่เกิดตอนไม่ได้ทำท่า (false positive/นาที)
"""
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.tuning import TUNING_PATH
from core.gesture_detector import GestureDetector
from core.smile import SmileEstimator

SIGNALS_PATH = os.path.join(os.path.dirname(TUNING_PATH), "gesture_signals.csv")

# ท่าที่สั่ง → ลำดับ event ที่ควรได้ ([] = ต้องไม่มี event)
EXPECTED = {
    "blink": [], "nod_down": [], "nod_up": [],
    "wink_left": ["left_click"], "wink_right": ["right_click"],
    "double_blink": ["double_click"], "drag": ["drag_start", "drag_end"],
    "smile": ["scroll_start", "scroll_end"],
}
# ลำดับอื่นที่ถือว่าถูก: ขยิบเร็ว 2 ครั้ง = คลิก + ดับเบิลคลิก (ผลต่อระบบคือดับเบิลคลิก)
ALSO_OK = {"double_blink": [["left_click", "double_click"], ["right_click", "double_click"]]}
GRACE_S = 0.6        # event หลังหมดหน้าต่าง GO ไม่เกินนี้ยังนับเป็นของท่านั้น (ลืมตา/เลิกยิ้มช้า)
FP_WEIGHT = 0.05     # หักคะแนน 0.05 ต่อการสั่งงานผิด 1 ครั้ง/นาที ตอนไม่ได้ทำท่า


# ──────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────
class Session:
    def __init__(self, sid, g, smile_comp=None):
        """
        Args:
            smile_comp: ค่าชดเชยการก้ม/เงยของ ΔSmile — ระบุแล้วจะคำนวณ delta_smile ใหม่
                        จาก mouth_ratio + pitch ที่อัดไว้ ด้วย SmileEstimator ตัวเดียวกับตอนใช้งานจริง
        """
        g = g.sort_values("t").copy()
        if smile_comp is not None and {"mouth_ratio", "pitch"} <= set(g.columns):
            est = SmileEstimator(smile_comp)
            g["delta_smile"] = [est.update(r, p) if r > 0 else 0.0
                                for r, p in zip(g["mouth_ratio"].values, g["pitch"].values)]
        self.sid = sid
        self.df = g
        self.frames = list(g[["t", "ear_l", "ear_r", "delta_smile"]].itertuples(index=False, name=None))
        go = g[g.phase == "go"].groupby("trial").agg(label=("label", "first"), t0=("t", "min"), t1=("t", "max"))
        self.trials = [(int(i), r.label, r.t0, r.t1) for i, r in go.iterrows()]
        gap = g[g.phase == "gap"]
        self.gap_s = float(gap.groupby("trial").t.agg(lambda s: s.max() - s.min()).sum())
        self.idle = [(t0, t1) for _, lb, t0, t1 in self.trials if lb == "idle"]
        quiet = gap if len(gap) > 10 else g
        self.ref_l = float(quiet.ear_l.quantile(0.6))
        self.ref_r = float(quiet.ear_r.quantile(0.6))


def load_sessions(path, smile_comp=None):
    if not os.path.exists(path):
        print(f"  ❌ ไม่พบ {path}\n     รัน python calibration/signal_recorder.py ก่อน")
        return []
    df = pd.read_csv(path)
    return [Session(sid, g, smile_comp) for sid, g in df.groupby("session", sort=False)]


def estimate_smile_pitch_comp(sessions):
    """ΔSmile เปลี่ยนตามการก้ม/เงยเท่าไหร่ — regression บนเฟรมที่ "ไม่ได้ยิ้ม"

    Returns:
        float หรือ None (ข้อมูลก้ม/เงยไม่พอ)
    """
    xs, ys = [], []
    for s in sessions:
        g = s.df
        if not {"mouth_ratio", "pitch"} <= set(g.columns):
            continue
        smile = np.zeros(len(g), bool)
        for _, label, t0, t1 in s.trials:
            if label == "smile":
                smile |= (g.t.values >= t0 - 0.3) & (g.t.values <= t1 + GRACE_S)
        n = g[~smile & (g.mouth_ratio.values > 0)]
        if len(n) < 100:
            continue
        xs.append((n.pitch - n.pitch.median()).values)
        ys.append((n.mouth_ratio / n.mouth_ratio.median() - 1).values)
    if not xs:
        return None
    x, y = np.concatenate(xs), np.concatenate(ys)
    if np.std(x) < 0.02:          # ไม่ได้ก้ม/เงยมากพอจะประมาณได้
        return None
    return float(np.clip(-np.sum(x * y) / np.sum(x * x), 0.0, 2.0))


# ──────────────────────────────────────────────
# Replay + scoring
# ──────────────────────────────────────────────
def replay(sess, gp, classifier=None, hook=None):
    """เล่นสัญญาณทั้ง session ผ่าน GestureDetector → [(t, event)]

    Args:
        classifier: โมเดลจำแนกท่า (None = rule-based)
        hook: episode_hook สำหรับเก็บฟีเจอร์ทุกจุดตัดสิน (ใช้สร้าง dataset)
    """
    det = GestureDetector(sess.ref_l, sess.ref_r, params=gp)
    det.set_classifier(classifier)
    det.episode_hook = hook
    events, prev = [], None
    for t, el, er, ds in sess.frames:
        if prev is not None and t - prev > 0.5:      # หน้าหาย → เหมือน pipeline
            det.reset()
        prev = t
        for e in det.update(t, el, er, ds):
            events.append((t, e))
    return events


def evaluate(sessions, gp, classifier=None):
    """คืน dict: score, balanced accuracy, fp/min, accuracy ต่อท่า, confusion

    Args:
        classifier: None = rule-based, โมเดลตัวเดียว, หรือ dict {session_id: โมเดล}
                    (ใช้ทดสอบแบบ leave-one-session-out)
    """
    per_label = defaultdict(lambda: [0, 0])
    confusion = defaultdict(Counter)
    fp, quiet_s = 0, 0.0
    for s in sessions:
        clf = classifier.get(s.sid) if isinstance(classifier, dict) else classifier
        events = replay(s, gp, clf)
        used = [False] * len(events)
        for _, label, t0, t1 in s.trials:
            got = []
            for i, (t, e) in enumerate(events):
                if not used[i] and t0 <= t <= t1 + GRACE_S:
                    used[i] = True
                    got.append(e)
            if label == "idle":
                fp += len(got)
                quiet_s += t1 - t0
                continue
            if label not in EXPECTED:
                continue
            ok = got == EXPECTED[label] or got in ALSO_OK.get(label, [])
            per_label[label][0] += ok
            per_label[label][1] += 1
            confusion[label]["none" if not got else "+".join(got)] += 1
        fp += used.count(False)
        quiet_s += s.gap_s
    acc = {k: c / n for k, (c, n) in per_label.items() if n}
    bal = float(np.mean(list(acc.values()))) if acc else 0.0
    fpm = fp / max(quiet_s / 60.0, 1e-6)
    return {"score": bal - FP_WEIGHT * fpm, "balanced_acc": bal, "fp_per_min": fpm,
            "acc": acc, "confusion": confusion}
