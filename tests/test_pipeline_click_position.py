"""ทดสอบว่าคลิกลงตรงตำแหน่ง "ก่อนทำท่า" แม้ใบหน้าเบี้ยวระหว่างขยิบตาจนเคอร์เซอร์ทำนายเพี้ยน
รัน:  python -m pytest tests -q   หรือ   python tests/test_pipeline_click_position.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import core.mouse_controller as mcmod
from core.pipeline import Pipeline
from config.tuning import GestureParams, ScrollParams

OPEN, CLOSED, DT = 0.33, 0.06, 1 / 30


def make_pipeline():
    log = []
    mc = mcmod.MouseController
    mc._set_pos = lambda self, x, y: log.append(("move", round(x), round(y)))
    for name in ("left_click", "right_click", "double_click", "mouse_down", "mouse_up", "scroll"):
        setattr(mc, name, (lambda n: lambda self, *a, **k: log.append((n,) + tuple(
            None if v is None else round(v) for v in a)))(name))
    p = Pipeline()
    # ใช้ค่า default เสมอ (ไม่ขึ้นกับ data/gesture_tuning.json ที่ผู้ใช้จูนไว้)
    p.gesture_params, p.scroll_params = GestureParams(), ScrollParams()
    p.gesture_detector.set_params(p.gesture_params)
    p.mouse_controller.enable()
    p.camera.get_frame_size = lambda: (640, 480)
    p.gesture_detector.set_reference(OPEN, OPEN)
    p._reset_runtime_state()
    return p, log


def drive(p, frames, t0=100.0):
    """frames: [(ear_l, ear_r, smile, (pred_x, pred_y)[, pitch])] ที่ 30 FPS → เวลาเฟรมถัดไป"""
    state = {}
    p.extractor.extract_click_features = lambda lm: np.array(
        [state["el"], state["er"], state["sm"]])
    p.extractor.extract_cursor_features = lambda lm, w, h: np.zeros(7)
    p.extractor.extract_head_pitch_signal = lambda lm: state["pitch"]
    p.cursor_predictor.is_loaded = lambda: True
    p.cursor_predictor.predict = lambda f: state["pos"]
    t = t0
    for fr in frames:
        el, er, sm, pos = fr[:4]
        state.update(el=el, er=er, sm=sm, pos=pos, pitch=fr[4] if len(fr) > 4 else 0.0)
        with p._lock:
            p._process_landmarks([None], t, swap_eyes=False)
        t += DT
    return t


def test_left_click_lands_where_the_cursor_was_before_the_wink():
    p, log = make_pipeline()
    target = (800.0, 400.0)
    frames = [(OPEN, OPEN, 0.0, target)] * 20
    # ขยิบตาซ้าย 0.5 วิ — ระหว่างนั้นการทำนายเพี้ยนไปไกล (ใบหน้าเบี้ยว/ศีรษะขยับ)
    frames += [(CLOSED, OPEN, 0.0, (1100.0, 650.0))] * 15
    frames += [(OPEN, OPEN, 0.0, (1100.0, 650.0))] * 3     # ลืมตา แต่ค่ายังเพี้ยนอยู่ครู่หนึ่ง
    frames += [(OPEN, OPEN, 0.0, (805.0, 402.0))] * 15
    drive(p, frames)
    clicks = [l for l in log if l[0] == "left_click"]
    assert len(clicks) == 1, log
    assert abs(clicks[0][1] - 800) <= 10 and abs(clicks[0][2] - 400) <= 10, clicks
    # และเคอร์เซอร์ไม่เคยถูกลากไปตำแหน่งเพี้ยนระหว่างขยิบ
    moves = [l for l in log if l[0] == "move"]
    assert all(m[1] < 900 for m in moves), moves


def test_right_click_and_normal_blink():
    p, log = make_pipeline()
    pos = (500.0, 300.0)
    frames = [(OPEN, OPEN, 0.0, pos)] * 15
    frames += [(CLOSED, CLOSED, 0.0, (700.0, 500.0))] * 4   # กะพริบปกติ → ไม่คลิก
    frames += [(OPEN, OPEN, 0.0, pos)] * 15
    frames += [(OPEN, CLOSED, 0.0, (900.0, 600.0))] * 15    # ขยิบขวา
    frames += [(OPEN, OPEN, 0.0, pos)] * 15
    drive(p, frames)
    assert [l[0] for l in log if l[0] not in ("move",)] == ["right_click"], log


def test_cursor_resumes_after_gesture():
    p, log = make_pipeline()
    frames = [(OPEN, OPEN, 0.0, (400.0, 300.0))] * 15
    frames += [(CLOSED, OPEN, 0.0, (400.0, 300.0))] * 15
    frames += [(OPEN, OPEN, 0.0, (400.0, 300.0))] * 10
    frames += [(OPEN, OPEN, 0.0, (1200.0, 700.0))] * 10
    drive(p, frames)
    last_move = [l for l in log if l[0] == "move"][-1]
    assert abs(last_move[1] - 1200) < 5, last_move


def _enter_scroll(pos):
    """ยิ้ม 0.8 วิ → เข้าโหมด scroll, เลิกยิ้มแล้วนั่งนิ่ง 0.8 วิ (ตั้งจุดอ้างอิง)"""
    return ([(OPEN, OPEN, 0.0, pos)] * 15 + [(OPEN, OPEN, 0.3, pos)] * 24
            + [(OPEN, OPEN, 0.0, pos)] * 24)


def _scrolls(log):
    return [l[1] for l in log if l[0] == "scroll"]


def test_scroll_down_by_nodding_even_at_bottom_edge_and_with_narrow_eyes():
    p, log = make_pipeline()
    bottom = (900.0, 1079.0)      # เคอร์เซอร์อยู่ขอบล่างจอ (เดิม: ก้มต่อไม่ได้ → เลื่อนลงไม่ได้)
    frames = _enter_scroll(bottom)
    # ก้มหน้า: สัญญาณ +0.06 และตาดูหรี่ลงเหลือ 70% (เดิม: ตรึงเคอร์เซอร์ → หยุดเลื่อน)
    frames += [(OPEN * 0.7, OPEN * 0.7, 0.0, bottom, 0.06)] * 45
    drive(p, frames)
    assert p.get_status()["mode"] == "scroll"
    s = _scrolls(log)
    assert s and all(v < 0 for v in s), log      # เลื่อนลง
    assert sum(s) <= -5, s
    assert not [l for l in log if l[0] in ("left_click", "right_click", "double_click")]


def test_scroll_up_then_exit_with_smile_and_cursor_resumes():
    p, log = make_pipeline()
    pos = (600.0, 400.0)
    frames = _enter_scroll(pos)
    frames += [(OPEN, OPEN, 0.0, pos, -0.06)] * 30           # เงย → เลื่อนขึ้น
    frames += [(OPEN, OPEN, 0.0, pos, 0.0)] * 15             # กลับท่าปกติ → หยุด
    frames += [(OPEN, OPEN, 0.3, pos, 0.0)] * 24             # ยิ้มอีกครั้ง → ออกโหมด
    frames += [(OPEN, OPEN, 0.0, (1000.0, 700.0), 0.0)] * 20
    drive(p, frames)
    s = _scrolls(log)
    assert s and all(v > 0 for v in s), s
    assert p.get_status()["mode"] == "normal"
    last_move = [l for l in log if l[0] == "move"][-1]
    assert abs(last_move[1] - 1000) < 5, last_move


def test_small_head_jitter_in_scroll_mode_does_not_scroll():
    p, log = make_pipeline()
    pos = (600.0, 400.0)
    frames = _enter_scroll(pos)
    frames += [(OPEN, OPEN, 0.0, pos, 0.008 * ((i % 6) - 3) / 3) for i in range(90)]
    drive(p, frames)
    assert _scrolls(log) == []


def test_wink_in_scroll_mode_does_not_click():
    p, log = make_pipeline()
    pos = (600.0, 400.0)
    frames = _enter_scroll(pos)
    frames += [(CLOSED, OPEN, 0.0, pos)] * 15 + [(OPEN, OPEN, 0.0, pos)] * 15
    drive(p, frames)
    assert not [l for l in log if l[0] == "left_click"], log
    assert p.get_status()["mode"] == "scroll"


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
