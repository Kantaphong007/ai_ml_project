"""ทดสอบ event เมาส์ที่ส่งจริงบน macOS (ดักที่ CGEventPost ไม่คลิกจริง)
รัน:  python -m pytest tests -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.mouse_controller as m

pytestmark = pytest.mark.skipif(not (m._IS_MAC and m.Quartz is not None), reason="macOS เท่านั้น")


def _capture(monkeypatch):
    posted = []
    Q = m.Quartz
    monkeypatch.setattr(Q, "CGEventPost", lambda tap, ev: posted.append(
        (Q.CGEventGetType(ev), Q.CGEventGetIntegerValueField(ev, Q.kCGMouseEventClickState))))
    mc = m.MouseController()
    mc.enable()
    return mc, posted


def test_double_click_sends_click_count_2(monkeypatch):
    # macOS นับดับเบิลคลิกจาก click count ใน event — คลิกเดี่ยว 2 ครั้งแอปไม่เห็นเป็นดับเบิลคลิก
    mc, posted = _capture(monkeypatch)
    mc.double_click(400, 300)
    Q = m.Quartz
    clicks = [(t, c) for t, c in posted if t in (Q.kCGEventLeftMouseDown, Q.kCGEventLeftMouseUp)]
    assert clicks == [(Q.kCGEventLeftMouseDown, 1), (Q.kCGEventLeftMouseUp, 1),
                      (Q.kCGEventLeftMouseDown, 2), (Q.kCGEventLeftMouseUp, 2)]


def test_drag_moves_with_dragged_events(monkeypatch):
    mc, posted = _capture(monkeypatch)
    mc.mouse_down(500, 300)
    mc.move_to(600, 350)
    mc.mouse_up()
    Q = m.Quartz
    kinds = [t for t, _ in posted]
    assert kinds[-3:] == [Q.kCGEventLeftMouseDown, Q.kCGEventLeftMouseDragged, Q.kCGEventLeftMouseUp]
