"""ทดสอบ GestureDetector ด้วยสัญญาณ EAR จำลองตามเวลาจริง (30 FPS)
รัน:  python -m pytest tests -q   หรือ   python tests/test_gesture_detector.py
"""
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.gesture_detector import GestureDetector

OPEN = 0.30
CLOSED = 0.06
DT = 1.0 / 30.0


def run(segments, smile=None, noise=0.012, seed=1):
    """segments: [(duration_s, ear_l, ear_r), ...] → list ของ (event, t)"""
    rnd = random.Random(seed)
    det = GestureDetector(OPEN, OPEN)
    t, out = 0.0, []
    for dur, el, er in segments:
        for _ in range(int(round(dur / DT))):
            sm = smile(t) if smile else 0.0
            ev = det.update(t, el + rnd.gauss(0, noise), er + rnd.gauss(0, noise), sm)
            out += [(e, round(t, 2)) for e in ev]
            t += DT
    return out, det


def names(out):
    return [e for e, _ in out]


def test_idle_no_events():
    out, _ = run([(5.0, OPEN, OPEN)])
    assert out == []


def test_normal_blink_ignored():
    out, _ = run([(1, OPEN, OPEN), (0.13, CLOSED, CLOSED), (3, OPEN, OPEN)])
    assert out == []


def test_slow_both_eyes_closed_ignored():
    out, _ = run([(1, OPEN, OPEN), (1.5, CLOSED, CLOSED), (1, OPEN, OPEN)])
    assert out == []


def test_left_wink_click():
    out, _ = run([(1, OPEN, OPEN), (0.5, CLOSED, OPEN), (1, OPEN, OPEN)])
    assert names(out) == ["left_click"]


def test_right_wink_click():
    out, _ = run([(1, OPEN, OPEN), (0.5, OPEN, CLOSED), (1, OPEN, OPEN)])
    assert names(out) == ["right_click"]


def test_short_wink_ignored():
    out, _ = run([(1, OPEN, OPEN), (0.2, CLOSED, OPEN), (1, OPEN, OPEN)])
    assert out == []


def test_double_blink():
    out, _ = run([(1, OPEN, OPEN), (0.12, CLOSED, CLOSED), (0.2, OPEN, OPEN),
                  (0.12, CLOSED, CLOSED), (1, OPEN, OPEN)])
    assert names(out) == ["double_click"]


def test_two_blinks_far_apart_ignored():
    out, _ = run([(1, OPEN, OPEN), (0.12, CLOSED, CLOSED), (2.0, OPEN, OPEN),
                  (0.12, CLOSED, CLOSED), (1, OPEN, OPEN)])
    assert out == []


def test_blink_with_one_eye_lagging_is_not_wink():
    # ตาซ้ายปิดก่อน 0.1 วิ แล้วตาขวาตามมา, เปิดตาซ้ายก่อนตาขวา 0.1 วิ → กะพริบ ไม่ใช่ขยิบ
    out, _ = run([(1, OPEN, OPEN), (0.1, CLOSED, OPEN), (0.12, CLOSED, CLOSED),
                  (0.1, OPEN, CLOSED), (1, OPEN, OPEN)])
    assert out == []


def test_long_wink_is_drag_not_click():
    out, _ = run([(1, OPEN, OPEN), (1.6, CLOSED, OPEN), (1, OPEN, OPEN)])
    assert names(out) == ["drag_toggle"]


def test_drag_toggle_twice():
    out, _ = run([(1, OPEN, OPEN), (1.6, CLOSED, OPEN), (1, OPEN, OPEN),
                  (1.6, CLOSED, OPEN), (1, OPEN, OPEN)])
    assert names(out) == ["drag_toggle", "drag_toggle"]


def test_smile_scroll_toggle_once():
    out, _ = run([(3, OPEN, OPEN)], smile=lambda t: 0.3 if 0.5 < t < 2.5 else 0.0)
    assert names(out) == ["scroll_toggle"]


def test_brief_smile_ignored():
    out, _ = run([(3, OPEN, OPEN)], smile=lambda t: 0.3 if 0.5 < t < 0.8 else 0.0)
    assert out == []


def test_ear_noise_near_threshold_no_flicker():
    # EAR ลอยๆ ที่ ~0.65 ของค่าเปิด (เหนือ close threshold) ห้ามเกิด event
    out, _ = run([(4, OPEN * 0.68, OPEN * 0.68)], noise=0.01)
    assert out == []


def test_cursor_frozen_during_wink_and_released_after():
    rnd = random.Random(3)
    det = GestureDetector(OPEN, OPEN)
    t, frozen_during, frozen_end = 0.0, [], None
    for dur, el, er in [(1, OPEN, OPEN), (0.5, CLOSED, OPEN), (1, OPEN, OPEN)]:
        for _ in range(int(dur / DT)):
            det.update(t, el, er, 0.0)
            if el == CLOSED:
                frozen_during.append(det.freeze_cursor)
            t += DT
    frozen_end = det.freeze_cursor
    assert all(frozen_during[3:])      # ตรึงตั้งแต่ไม่กี่เฟรมแรกของการหลับตา
    assert frozen_end is False         # ปล่อยหลังลืมตาแล้ว


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
