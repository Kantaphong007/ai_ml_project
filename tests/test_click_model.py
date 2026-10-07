"""ทดสอบเส้นทาง ML ของการคลิก: ฟีเจอร์ episode, GestureDetector + classifier, การสร้าง dataset
รัน:  python -m pytest tests -q   หรือ   python tests/test_click_model.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.episode_features import FEATURE_NAMES, compute_features, mirror_frames
from core.gesture_detector import GestureDetector

OPEN, CLOSED, DT = 0.30, 0.06, 1 / 30


class FakeClassifier:
    """คืนคลาสตายตัว แยกตามจุดตัดสิน (end/hold) และจดฟีเจอร์ที่ได้รับ"""

    def __init__(self, end_cls, hold_cls=0, conf=0.9):
        self.end_cls, self.hold_cls, self.conf = end_cls, hold_cls, conf
        self.seen = []

    def predict(self, feats):
        self.seen.append(feats)
        is_hold = feats[FEATURE_NAMES.index("is_hold")] == 1.0
        return (self.hold_cls if is_hold else self.end_cls), self.conf


def run(segments, clf=None, hook=None):
    det = GestureDetector(OPEN, OPEN)
    det.set_classifier(clf)
    det.episode_hook = hook
    t, out = 0.0, []
    for dur, el, er in segments:
        for _ in range(int(round(dur / DT))):
            out += det.update(t, el, er, 0.0)
            t += DT
    return out


def test_features_mirror_symmetry():
    frames = [(i * DT, 0.2 + 0.01 * i, 0.7 - 0.01 * i, 0.0) for i in range(12)]
    f = dict(zip(FEATURE_NAMES, compute_features(frames, 0.0, 0.4)))
    m = dict(zip(FEATURE_NAMES, compute_features(mirror_frames(frames), 0.0, 0.4)))
    assert abs(f["min_l"] - m["min_r"]) < 1e-12 and abs(f["mean_r"] - m["mean_l"]) < 1e-12
    assert abs(f["asym_mean"] + m["asym_mean"]) < 1e-12
    assert f["frac_l_closed"] == m["frac_r_closed"]
    for k in ("duration", "min_lo", "min_hi", "asym_abs_mean", "close_speed"):
        assert abs(f[k] - m[k]) < 1e-12, k


def test_ml_decides_click_at_episode_end():
    clf = FakeClassifier(end_cls=2)          # โมเดลบอกว่าเป็นขยิบขวา
    out = run([(1, OPEN, OPEN), (0.5, CLOSED, OPEN), (1, OPEN, OPEN)], clf)
    assert out == ["right_click"]            # ใช้คำตอบโมเดล ไม่ใช่ rule (ซึ่งจะได้ left_click)
    assert len(clf.seen) == 1 and len(clf.seen[0]) == len(FEATURE_NAMES)


def test_ml_unsure_falls_back_to_rule():
    # ML ไม่มั่นใจ → ไม่ทิ้งท่าทาง ใช้ rule ตัดสินแทน (ขยิบซ้าย 0.5 วิ = left_click)
    det_out = run([(1, OPEN, OPEN), (0.5, CLOSED, OPEN), (1, OPEN, OPEN)],
                  FakeClassifier(end_cls=0, conf=0.3))
    assert det_out == ["left_click"]


def test_ml_confident_no_action_is_respected():
    out = run([(1, OPEN, OPEN), (0.5, CLOSED, OPEN), (1, OPEN, OPEN)],
              FakeClassifier(end_cls=0, conf=0.9))
    assert out == []


def test_ml_drag_decided_at_hold_point():
    out = run([(1, OPEN, OPEN), (1.6, CLOSED, OPEN), (1, OPEN, OPEN)],
              FakeClassifier(end_cls=1, hold_cls=4))
    assert out == ["drag_start", "drag_end"]  # ตัดสินที่ hold → กดค้าง, ลืมตา → ปล่อย


def test_ml_long_closure_not_drag_means_nothing():
    out = run([(1, OPEN, OPEN), (1.6, CLOSED, CLOSED), (1, OPEN, OPEN)],
              FakeClassifier(end_cls=1, hold_cls=0))
    assert out == []


def test_ml_double_click_gets_previous_episode_features():
    clf = FakeClassifier(end_cls=0)
    run([(1, OPEN, OPEN), (0.12, CLOSED, CLOSED), (0.2, OPEN, OPEN),
         (0.12, CLOSED, CLOSED), (1, OPEN, OPEN)], clf)
    gap = FEATURE_NAMES.index("prev_gap")
    assert len(clf.seen) == 2
    assert clf.seen[0][gap] == 3.0                     # ครั้งแรก: ไม่มี episode ก่อนหน้า
    assert 0.1 < clf.seen[1][gap] < 0.4                # ครั้งที่สอง: ห่างจากครั้งแรก ~0.2 วิ


def test_hook_records_every_decision_in_rule_mode():
    recs = []
    out = run([(1, OPEN, OPEN), (0.5, CLOSED, OPEN), (1, OPEN, OPEN),
               (1.6, CLOSED, OPEN), (1, OPEN, OPEN)],
              hook=lambda kind, start, t, feats: recs.append(kind))
    assert recs == ["end", "hold"]
    assert out == ["left_click", "drag_start", "drag_end"]   # rule-based ยังทำงานปกติ


def test_dataset_labels_from_trials():
    import pandas as pd
    from config.tuning import GestureParams
    from training.signal_data import Session
    from training.train_click_model import build_dataset

    rows, t = [], 0.0

    def seg(trial, label, phase, dur, el, er):
        nonlocal t
        for _ in range(int(round(dur / DT))):
            rows.append(dict(session="s", trial=trial, label=label, phase=phase, t=t,
                             ear_l=el, ear_r=er, delta_smile=0.0, mouth_ratio=0.9, pitch=0.0))
            t += DT

    seg(1, "wink_left", "gap", 1.6, OPEN, OPEN)
    seg(1, "wink_left", "go", 0.5, OPEN, OPEN)
    seg(1, "wink_left", "go", 0.5, CLOSED, OPEN)
    seg(1, "wink_left", "go", 1.5, OPEN, OPEN)
    seg(2, "drag", "gap", 1.6, OPEN, OPEN)
    seg(2, "drag", "go", 0.5, OPEN, OPEN)
    seg(2, "drag", "go", 1.6, OPEN, CLOSED)
    seg(2, "drag", "go", 1.1, OPEN, OPEN)
    seg(3, "double_blink", "gap", 1.6, OPEN, OPEN)
    seg(3, "double_blink", "go", 0.5, OPEN, OPEN)
    for _ in range(2):
        seg(3, "double_blink", "go", 0.12, CLOSED, CLOSED)
        seg(3, "double_blink", "go", 0.25, OPEN, OPEN)
    seg(3, "double_blink", "go", 1.3, OPEN, OPEN)

    sess = Session("s", pd.DataFrame(rows))
    X, y, groups, sids = build_dataset([sess], GestureParams(), augment=True)
    assert X.shape[1] == len(FEATURE_NAMES)
    # ต้นฉบับ: ขยิบซ้าย(1), drag(4), กะพริบครั้งแรก(0), ครั้งที่สอง(3)
    # ภาพสะท้อน: ขยิบขวา(2), drag(4), 0, 3
    assert sorted(y.tolist()) == [0, 0, 1, 2, 3, 3, 4, 4], y
    assert len(set(groups)) == 4                       # ต้นฉบับ+ภาพสะท้อนอยู่กลุ่มเดียวกัน


def test_double_click_works_when_ml_decides_single_winks():
    # ML มั่นใจว่าแต่ละครั้งเป็นขยิบซ้าย → ชั้นลำดับเวลารวมสองครั้งติดกันเป็นดับเบิลคลิก
    out = run([(1, OPEN, OPEN), (0.2, CLOSED, OPEN), (0.25, OPEN, OPEN),
               (0.2, CLOSED, OPEN), (1, OPEN, OPEN)], FakeClassifier(end_cls=1))
    assert out == ["left_click", "double_click"]


def test_double_blink_works_even_if_ml_says_no_action():
    out = run([(1, OPEN, OPEN), (0.12, CLOSED, CLOSED), (0.2, OPEN, OPEN),
               (0.12, CLOSED, CLOSED), (1, OPEN, OPEN)], FakeClassifier(end_cls=0))
    assert out == ["double_click"]


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
