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
    # ตากระตุก 2 เฟรม (0.07 วิ) = noise ไม่ใช่ขยิบ (ขยิบชัดต้อง ≥ clear_wink_min_s 0.10 วิ)
    out, _ = run([(1, OPEN, OPEN), (0.07, CLOSED, OPEN), (1, OPEN, OPEN)])
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
    assert names(out) == ["drag_start", "drag_end"]


def test_drag_released_when_eye_opens():
    out, _ = run([(1, OPEN, OPEN), (3.0, CLOSED, OPEN), (1, OPEN, OPEN)])
    starts = [t for e, t in out if e == "drag_start"]
    ends = [t for e, t in out if e == "drag_end"]
    assert names(out) == ["drag_start", "drag_end"]
    assert 2.1 <= starts[0] <= 2.4          # หลับครบ drag_hold_s (1.2 วิ) → กดค้าง
    assert 4.0 <= ends[0] <= 4.3            # ลืมตาที่ 4.0 วิ → ปล่อยทันที (ไม่ต้องหลับซ้ำ)


def test_long_drag_not_cut_by_episode_timeout():
    out, _ = run([(1, OPEN, OPEN), (8.0, CLOSED, OPEN), (1, OPEN, OPEN)])
    assert names(out) == ["drag_start", "drag_end"]
    assert [t for e, t in out if e == "drag_end"][0] >= 9.0


def test_cursor_follows_head_while_dragging():
    det = GestureDetector(OPEN, OPEN)
    t, frozen = 0.0, []
    for dur, el, er in [(1, OPEN, OPEN), (3.0, CLOSED, OPEN)]:
        for _ in range(int(dur / DT)):
            det.update(t, el, er, 0.0)
            if det.dragging:
                frozen.append(det.freeze_cursor)
            t += DT
    assert frozen and not any(frozen[6:])   # ไม่ตรึงเคอร์เซอร์ระหว่างลาก


def test_drag_end_not_blocked_by_cooldown():
    # ลืมตาเร็วมากหลัง drag_start (< event_cooldown_s) → ต้องปล่อยเมาส์เสมอ
    out, _ = run([(1, OPEN, OPEN), (1.25, CLOSED, OPEN), (1, OPEN, OPEN)], noise=0.0)
    assert names(out) == ["drag_start", "drag_end"]


def test_smile_holds_scroll_mode_until_smile_ends():
    out, _ = run([(3, OPEN, OPEN)], smile=lambda t: 0.3 if 0.5 < t < 2.5 else 0.0)
    assert names(out) == ["scroll_start", "scroll_end"]


def test_brief_smile_ignored():
    out, _ = run([(3, OPEN, OPEN)], smile=lambda t: 0.3 if 0.5 < t < 0.7 else 0.0)
    assert out == []


def test_scroll_ends_quickly_after_smile_ends():
    out, _ = run([(4, OPEN, OPEN)], smile=lambda t: 0.3 if 0.5 < t < 2.5 else 0.0)
    t_start = [t for e, t in out if e == "scroll_start"][0]
    t_end = [t for e, t in out if e == "scroll_end"][0]
    assert 0.75 <= t_start <= 0.95           # ยิ้มค้าง smile_hold_s (0.3 วิ) → เข้าโหมด
    assert 2.5 <= t_end <= 2.9               # หุบยิ้ม → ออกภายใน ~0.3 วิ


def test_ear_noise_near_threshold_no_flicker():
    # EAR ลอยๆ ที่ ~0.65 ของค่าเปิด (เหนือ close threshold) ห้ามเกิด event
    out, _ = run([(4, OPEN * 0.68, OPEN * 0.68)], noise=0.01)
    assert out == []


def test_cursor_frozen_during_wink_and_released_after():
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


def run_fn(fn, dur, det=None, t0=0.0, seed=2, noise=0.008):
    """fn(t) → (ear_l, ear_r, smile) ต่อเนื่องตามเวลา"""
    rnd = random.Random(seed)
    det = det or GestureDetector(OPEN, OPEN)
    t, out = t0, []
    while t < t0 + dur:
        el, er, sm = fn(t)
        out += [(e, round(t, 2)) for e in det.update(t, el + rnd.gauss(0, noise),
                                                     er + rnd.gauss(0, noise), sm)]
        t += DT
    return out, det


def test_wink_while_open_eye_is_dragged_half_closed():
    # MediaPipe มักลากตาที่เปิดอยู่ให้หรี่ตาม (เหลือ ~65%) ตอนขยิบ → ต้องยังเป็นขยิบซ้าย
    out, _ = run([(1, OPEN, OPEN), (0.5, CLOSED, OPEN * 0.62), (1, OPEN, OPEN)])
    assert names(out) == ["left_click"]


def test_right_wink_with_coupling():
    out, _ = run([(1, OPEN, OPEN), (0.5, OPEN * 0.6, CLOSED), (1, OPEN, OPEN)])
    assert names(out) == ["right_click"]


def _tilt(t, start=1.0, ramp=0.8, depth=0.62):
    """ก้มหน้า: EAR สองตาลดลงช้าๆ เหลือ depth ของค่าเปิด"""
    k = min(1.0, max(0.0, (t - start) / ramp))
    return OPEN * (1 - (1 - depth) * k)


def test_head_tilt_lowers_both_eyes_no_event_no_stuck_freeze():
    out, det = run_fn(lambda t: (_tilt(t), _tilt(t), 0.0), 5.0)
    assert out == []
    assert det.freeze_cursor is False     # ไม่ตรึงเคอร์เซอร์ค้างตอนก้มไปมองขอบล่างของจอ


def test_wink_still_works_after_head_tilt():
    def fn(t):
        e = _tilt(t)
        return (CLOSED if 4.0 <= t < 4.5 else e), e, 0.0
    out, _ = run_fn(fn, 6.0)
    assert names(out) == ["left_click"]


def test_learns_reference_without_baseline():
    # ไม่มี baseline: ค่าเริ่ม 0.30 แต่ตาผู้ใช้เปิดแค่ 0.18 → ห้ามคลิกมั่ว และต้องขยิบได้หลังเรียนค่า
    det = GestureDetector(0.30, 0.30)
    det.set_reference(0.30, 0.30, trusted=False)
    small = 0.18
    out, _ = run_fn(lambda t: ((0.03 if 2.0 <= t < 2.5 else small), small, 0.0), 4.0, det=det, noise=0.005)
    assert names(out) == ["left_click"]


def test_shallow_double_blink_ignored():
    # กะพริบตื้นๆ (ตาไม่ปิดจริง) สองครั้ง ≠ ดับเบิลคลิก
    half = OPEN * 0.58
    out, _ = run([(1, OPEN, OPEN), (0.12, half, half), (0.2, OPEN, OPEN),
                  (0.12, half, half), (1, OPEN, OPEN)], noise=0.004)
    assert out == []


def test_one_frame_dropout_does_not_split_wink():
    # landmark กระตุกเปิด 1 เฟรมกลางการขยิบ → ยังเป็นคลิกเดียว
    out, _ = run([(1, OPEN, OPEN), (0.25, CLOSED, OPEN), (DT, OPEN, OPEN),
                  (0.25, CLOSED, OPEN), (1, OPEN, OPEN)], noise=0.0)
    assert names(out) == ["left_click"]


def test_slow_reopen_after_deep_wink_still_clicks():
    # หลับลึกแล้วค่อยๆ ลืม ค้างที่ ~80% (ไม่ถึง open_ratio) → เดิมถูกยกเลิก ต้องคลิก
    out, det = run([(1, OPEN, OPEN), (0.5, CLOSED, OPEN), (1.0, OPEN * 0.70, OPEN), (1, OPEN, OPEN)],
                   noise=0.004)
    assert names(out) == ["left_click"], det.last_decision


def test_decision_reason_is_reported():
    out, det = run([(1, OPEN, OPEN), (1.15, CLOSED, OPEN), (1, OPEN, OPEN)], noise=0.0)
    assert out == [] and "นาน" in det.last_decision      # ขยิบนานเกิน → บอกเหตุผล


def test_quick_clear_wink_clicks():
    # ขยิบเร็ว 0.2 วิ (แบบที่ผู้ใช้จริงทำ) แต่อีกตาเปิดปกติ → คลิก
    out, _ = run([(1, OPEN, OPEN), (0.2, CLOSED, OPEN), (1, OPEN, OPEN)], noise=0.004)
    assert names(out) == ["left_click"]


def test_quick_blink_with_lag_is_not_wink():
    # กะพริบเร็ว สองตาปิด (อีกตาลงต่ำด้วย) → ไม่ใช่ขยิบชัด → ไม่คลิก
    out, _ = run([(1, OPEN, OPEN), (0.07, CLOSED, OPEN * 0.6), (0.1, CLOSED, CLOSED),
                  (1, OPEN, OPEN)], noise=0.004)
    assert out == []


def test_two_quick_winks_same_eye_is_double_click():
    out, _ = run([(1, OPEN, OPEN), (0.2, CLOSED, OPEN), (0.25, OPEN, OPEN),
                  (0.2, CLOSED, OPEN), (1, OPEN, OPEN)], noise=0.004)
    assert names(out) == ["left_click", "double_click"]


def test_two_winks_far_apart_are_two_clicks():
    out, _ = run([(1, OPEN, OPEN), (0.2, CLOSED, OPEN), (2.0, OPEN, OPEN),
                  (0.2, CLOSED, OPEN), (1, OPEN, OPEN)], noise=0.004)
    assert names(out) == ["left_click", "left_click"]


def test_w_shaped_double_blink_in_one_episode():
    # ท่าดับเบิลคลิกจริงของผู้ใช้: ตาปิด → เปิดขึ้นครึ่งทาง (ไม่ถึง open_ratio) → ปิดอีกรอบ
    out, det = run([(1, OPEN, OPEN), (0.15, OPEN * 0.3, OPEN * 0.8), (0.12, OPEN * 0.6, OPEN * 0.85),
                    (0.15, OPEN * 0.35, OPEN * 0.8), (1, OPEN, OPEN)], noise=0.004)
    assert names(out) == ["double_click"], det.last_decision


def test_single_long_wink_is_not_double():
    out, _ = run([(1, OPEN, OPEN), (0.45, OPEN * 0.3, OPEN * 0.9), (1, OPEN, OPEN)], noise=0.008)
    assert names(out) == ["left_click"]


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
