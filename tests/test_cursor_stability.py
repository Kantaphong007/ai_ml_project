"""ทดสอบความนิ่งของเคอร์เซอร์ (sticky pointer) และการติดขอบจอ (EdgeSafeRegressor)
รัน:  python -m pytest tests -q   หรือ   python tests/test_cursor_stability.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

from core.cursor_models import EdgeSafeRegressor
from core.cursor_predictor import CursorPredictor

DT = 1 / 30


def _run(pred, path):
    """path: list ของ (x, y) ดิบที่เข้าตัวกรองทีละเฟรม → list ผลลัพธ์"""
    return [pred._smooth(x, y, t=i * DT) for i, (x, y) in enumerate(path)]


def test_cursor_still_when_head_still_with_noise():
    rnd = random.Random(0)
    pred = CursorPredictor()
    out = _run(pred, [(700 + rnd.gauss(0, 12), 450 + rnd.gauss(0, 18)) for _ in range(150)])
    tail = out[30:]
    assert len(set(tail)) <= 3, len(set(tail))     # ล็อกนิ่ง (เดิมสั่นทุกเฟรม)


def test_cursor_follows_real_movement_and_settles():
    pred = CursorPredictor()
    path = [(400, 400)] * 30 + [(400 + 15 * i, 400) for i in range(40)] + [(1000, 400)] * 60
    out = _run(pred, path)
    assert out[69][0] > 800                          # ตามทันระหว่างเคลื่อน
    assert abs(out[-1][0] - 1000) < 15               # หยุดใกล้เป้า
    assert len(set(out[-20:])) == 1                  # แล้วล็อกนิ่ง


def test_natural_head_sway_does_not_move_cursor():
    import math
    pred = CursorPredictor()
    pred._axis_scale = (0.6, 1.42)
    rnd = random.Random(1)
    path = [(700 + 15 * math.sin(2 * math.pi * 0.25 * i / 30) + rnd.gauss(0, 8.8),
             450 + 25 * math.sin(2 * math.pi * 0.2 * i / 30) + rnd.gauss(0, 21.7)) for i in range(1800)]
    out = _run(pred, path)
    moves = sum(1 for a, b in zip(out[60:], out[61:]) if a != b)
    assert moves < 60, moves                          # เดิม (ไม่มีล็อกแน่นขึ้น) ~1,000+ เฟรมที่ขยับ


def test_deliberate_move_after_long_still_still_follows():
    pred = CursorPredictor()
    path = [(700, 450)] * 90 + [(700 + 6 * i, 450) for i in range(1, 21)] + [(820, 450)] * 30
    out = _run(pred, path)
    assert out[-1][0] > 790                           # นิ่ง 3 วิ (ล็อกแน่นสุด) แล้วขยับ 120px ยังตามได้


def test_deadzone_zero_disables_lock():
    pred = CursorPredictor()
    pred.set_deadzone(0)
    out = _run(pred, [(500 + (i % 2) * 6, 500) for i in range(60)])
    assert len(set(out[-10:])) > 1


def test_edge_safe_model_sticks_beyond_calibrated_range():
    rnd = np.random.default_rng(0)
    X = rnd.uniform(-1, 1, size=(400, 2))
    y = np.c_[700 + 700 * X[:, 0] + 30 * X[:, 0] ** 2, 450 + 450 * X[:, 1]]
    plain = MultiOutputRegressor(SVR(C=1000, epsilon=5)).fit(X, y)
    safe = EdgeSafeRegressor(MultiOutputRegressor(SVR(C=1000, epsilon=5))).fit(X, y)
    xs = np.linspace(0.9, 3.0, 12)
    probe = np.c_[xs, np.zeros_like(xs)]
    p_plain = plain.predict(probe)[:, 0]
    p_safe = safe.predict(probe)[:, 0]
    assert p_plain[-1] < p_plain[0]                 # SVR เดิม: หันเลยขอบ → ค่าวิ่งกลับเข้าจอ
    assert np.all(np.diff(p_safe) > 0)              # edge-safe: ยิ่งหันยิ่งออกไปทางเดิม (แล้วถูก clamp ติดขอบ)
    inside = rnd.uniform(-0.9, 0.9, size=(200, 2))
    y_in = np.c_[700 + 700 * inside[:, 0] + 30 * inside[:, 0] ** 2, 450 + 450 * inside[:, 1]]
    assert np.abs(safe.predict(inside) - y_in).mean() < 15   # ในช่วง calibrate ยังแม่น


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if fails else 0)
