"""ทดสอบโหมด relative / hybrid (HeadPointer) ด้วยมุมศีรษะจำลอง + noise
รัน:  python -m pytest tests -q   หรือ   python tests/test_head_pointer.py
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.head_pointer import HeadPointer

DT = 1 / 30
W, H = 1470, 956


def run(path, target=None, seed=0, noise=(0.15, 0.2), pointer=None):
    """path: [(yaw, pitch)] องศา ทีละเฟรม → list ตำแหน่งเคอร์เซอร์"""
    rnd = random.Random(seed)
    p = pointer or HeadPointer(W, H)
    return [p.update(i * DT, y + rnd.gauss(0, noise[0]), pt + rnd.gauss(0, noise[1]),
                     target(i) if target else None) for i, (y, pt) in enumerate(path)]


def moved_frames(out, skip=30):
    return sum(a != b for a, b in zip(out[skip:], out[skip + 1:]))


def test_still_head_cursor_does_not_move():
    assert moved_frames(run([(0, 0)] * 1800)) == 0


def test_natural_sway_does_not_move_cursor():
    path = [(0.5 * math.sin(2 * math.pi * 0.25 * i * DT), 0.6 * math.sin(2 * math.pi * 0.2 * i * DT))
            for i in range(1800)]
    assert moved_frames(run(path)) == 0


def test_direction_right_and_down():
    right = run([(0, 0)] * 30 + [(10 * min(1, i / 9), 0) for i in range(40)])
    down = run([(0, 0)] * 30 + [(0, 10 * min(1, i / 9)) for i in range(40)])
    assert right[-1][0] > W / 2 + 200 and abs(right[-1][1] - H / 2) < 20
    assert down[-1][1] > H / 2 + 200 and abs(down[-1][0] - W / 2) < 20


def test_slow_turn_is_precise_fast_turn_is_far():
    slow = run([(0, 0)] * 30 + [(5 * min(1, i / 30), 0) for i in range(60)])
    fast = run([(0, 0)] * 30 + [(5 * min(1, i / 3), 0) for i in range(60)])
    d_slow, d_fast = slow[-1][0] - W / 2, fast[-1][0] - W / 2
    assert 20 < d_slow < 120              # หันช้า 5° → ขยับนิดเดียว เล็งละเอียดได้
    assert d_fast > 2 * d_slow            # หันเร็วระยะเท่ากัน → ไปไกลกว่า (pointer acceleration)


def test_sticks_at_edge_and_follows_face_when_coming_back():
    out = run([(0, 0)] * 30 + [(45 * min(1, i / 9), 0) for i in range(40)]
              + [(45 * (1 - min(1, i / 20)), 0) for i in range(40)])
    assert out[69][0] >= W - 5                       # หันเลยขอบ → ติดขอบ
    assert abs(out[-1][0] - W / 2) < 360             # หันกลับหน้าตรง → กลับมาแถวกลางจอ (เดิมไปค้างขอบซ้าย)


def test_fast_out_slow_back_does_not_drift_away():
    path = [(0, 0)] * 30
    for _ in range(4):   # หันซ้ายเร็ว (gain สูง) แล้วกลับช้า (gain ต่ำ) — ระยะไป/กลับไม่เท่ากัน
        path += ([(-12 * min(1, i / 6), 0) for i in range(6)] + [(-12, 0)] * 10
                 + [(-12 * (1 - i / 75), 0) for i in range(75)] + [(0, 0)] * 15)
    out = run(path)
    assert abs(out[-1][0] - W / 2) < 260, out[-1]    # หน้าตรง → ไม่ห่างกลางจอเกินระยะที่ยอมให้ (เดิมค้าง x≈140)


def test_hybrid_lands_near_ml_target_after_flick():
    target = lambda i: (1200, 450)
    fast = run([(0, 0)] * 30 + [(8 * min(1, i / 4), 0) for i in range(40)], target=target)
    peak_err = max(abs(o[0] - 1200) for o in fast)
    assert abs(fast[-1][0] - 1200) < 0.5 * peak_err  # สะบัดจบ → ไหลเข้าหาเป้าของโมเดล ML



def test_resync_ignores_angle_change_while_held():
    p = HeadPointer(W, H)
    run([(0, 0)] * 30, pointer=p)
    before = p.pos
    p.resync()                                       # เช่น ระหว่างขยิบตา (ตรึงเคอร์เซอร์)
    after = run([(6, 4)] * 30, pointer=p)            # มุมเปลี่ยนไปตอนตรึง → ห้ามกระโดด
    assert after[-1] == before


def test_baseline_neutral_maps_to_screen_center():
    # Baseline บอกว่าหน้าตรงของผู้ใช้คือ yaw +6° (กล้องอยู่เยื้อง) → หันมาที่ +6° = กลางจอ
    p = HeadPointer(W, H)
    p.set_neutral(6.0, -3.0)
    p.set_position(100, 100)                          # เคอร์เซอร์เริ่มที่มุมซ้ายบน
    path = [(6.0 - 20 * (1 - min(1, i / 15)), -3.0) for i in range(60)]  # หันจาก -14° มาหน้าตรง
    out = run(path, pointer=p)
    assert abs(out[-1][0] - W / 2) < 360, out[-1]     # จบที่แถวกลางจอ ไม่ใช่ค้างซ้าย


def test_cursor_never_moves_against_head_direction():
    """การแก้ตำแหน่งต้องไม่ "ยึก": เคอร์เซอร์ขยับทิศเดียวกับมุมหัว (หลังกรอง) เสมอ หรือไม่ขยับ"""
    rnd = random.Random(5)
    path, cur = [(0, 0)] * 30, 0.0
    for _ in range(30):
        nxt = rnd.uniform(-20, 20)
        n = rnd.randint(4, 40)
        path += [(cur + (nxt - cur) * min(1, i / n), 0) for i in range(n)] + [(nxt, 0)] * rnd.randint(3, 20)
        cur = nxt
    for target in (None, lambda i: (W / 2 + 150 + 40 * math.sin(i / 7), H / 2)):   # รวมโมเดลที่คลาดมาก
        p = HeadPointer(W, H)
        p.set_neutral(0, 0)
        prev_pos, reversals = None, 0
        for i, (yaw, pitch) in enumerate(path):
            before = p._prev
            pos = p.update(i * DT, yaw + rnd.gauss(0, 0.04), pitch, target(i) if target else None)
            if prev_pos is not None and before is not None:
                da, dx = p._prev[0] - before[0], pos[0] - prev_pos[0]
                if abs(dx) > 0.5 and (dx > 0) != (da > 0):
                    reversals += 1
            prev_pos = pos
        assert reversals == 0, reversals


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
