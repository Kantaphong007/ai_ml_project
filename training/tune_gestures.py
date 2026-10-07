"""
EYE ORDER COME AI — Tune Gestures (จูนค่าการตรวจจับท่าทาง + scroll จากข้อมูลจริง)

อ่าน data/gesture_signals.csv (จาก calibration/signal_recorder.py) แล้ว:
  1. replay สัญญาณผ่าน GestureDetector "ตัวเดียวกับที่ใช้งานจริง" เฟรมต่อเฟรม
  2. วัดผล: ความแม่นต่อท่า (balanced accuracy) + จำนวนคลิกผิดตอนไม่ได้ทำท่า (ครั้ง/นาที)
  3. ค้นหาพารามิเตอร์ (random search + ปรับทีละตัว) ที่ได้คะแนนดีที่สุด
  4. คำนวณ deadzone / ความเร็วเต็มของโหมด scroll จากท่าก้ม/เงย และ noise ตอนนั่งนิ่ง
  5. บันทึก data/gesture_tuning.json → Pipeline โหลดอัตโนมัติตอนกด Start

ใช้ก่อน train_click_model.py: ค่าที่จูน (close/open ratio ฯลฯ) กำหนดว่า episode ถูกตัดตรงไหน
ซึ่งโมเดล ML ใช้เป็น input — จูนใหม่เมื่อไหร่ควรเทรนโมเดล click ใหม่ด้วย

Usage:
    python training/tune_gestures.py                # จูนและบันทึก
    python training/tune_gestures.py --dry-run      # ดูผลอย่างเดียว ไม่บันทึก
    python training/tune_gestures.py --eval-only    # วัดผลค่าปัจจุบันอย่างเดียว
"""
import argparse
import os
import random
import sys
import time
from collections import Counter
from dataclasses import fields, replace

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.tuning import GestureParams, load_tuning, save_tuning
from training.signal_data import (
    EXPECTED, GRACE_S, SIGNALS_PATH, Session, estimate_smile_pitch_comp, evaluate, load_sessions,
)

REG_WEIGHT = 0.01    # ค่าที่ได้คะแนนเท่ากัน เลือกตัวที่ใกล้ default (กัน overfit)

# ช่วงค้นหา (open_ratio = close_ratio + open_gap)
SEARCH_SPACE = {
    "close_ratio": (0.40, 0.75),
    "open_gap": (0.06, 0.22),
    "wink_asym": (0.10, 0.40),
    "wink_side_frac": (0.40, 0.85),
    "blink_depth": (0.30, 0.85),
    "wink_min_s": (0.10, 0.25),          # ผู้ใช้ต้องการขยิบสั้น — ไม่ค้นหาค่าที่ยาวกว่านี้
    "clear_wink_min_s": (0.07, 0.12),
    "clear_wink_other": (0.55, 0.80),
    "quick_wink_max_s": (0.20, 0.50),
    "wink_max_s": (0.70, 1.15),
    "drag_hold_s": (1.00, 1.40),
    "blink_max_s": (0.25, 0.60),
    "double_blink_gap_s": (0.35, 1.00),
    "smile_on": (0.08, 0.40),
    "smile_hold_s": (0.30, 0.80),
    "open_confirm_s": (0.00, 0.12),
}


# ──────────────────────────────────────────────
# Search
# ──────────────────────────────────────────────
def _to_vec(gp):
    v = {k: getattr(gp, k) for k in SEARCH_SPACE if k != "open_gap"}
    v["open_gap"] = gp.open_ratio - gp.close_ratio
    return v


def _from_vec(base, v):
    gp = replace(base, **{k: x for k, x in v.items() if k != "open_gap"})
    gp.open_ratio = gp.close_ratio + v["open_gap"]
    gp.freeze_ratio = max(base.freeze_ratio, gp.open_ratio + 0.03)
    gp.smile_off = gp.smile_on * 0.6
    gp.smile_freeze = min(base.smile_freeze, gp.smile_on * 0.5)
    gp.wink_max_s = min(gp.wink_max_s, gp.drag_hold_s - 0.05)
    gp.wink_min_s = min(gp.wink_min_s, gp.wink_max_s - 0.1)
    return gp


def _distance(v, ref):
    return float(np.mean([abs(v[k] - ref[k]) / (hi - lo) for k, (lo, hi) in SEARCH_SPACE.items()]))


def search(sessions, start, iters, seed):
    rng = random.Random(seed)
    ref = _to_vec(GestureParams())

    def objective(v):
        r = evaluate(sessions, _from_vec(start, v))
        return r["score"] - REG_WEIGHT * _distance(v, ref), r

    best_v = _to_vec(start)
    best_obj, best_r = objective(best_v)
    t0 = time.time()

    # 1) random search: ครึ่งแรกสุ่มทั้งช่วง ครึ่งหลังสุ่มแคบรอบค่าที่ดีที่สุด (±15%)
    for i in range(iters):
        v = {}
        for k, (lo, hi) in SEARCH_SPACE.items():
            if i < iters // 2:
                v[k] = rng.uniform(lo, hi)
            else:
                d = 0.15 * (hi - lo)
                v[k] = min(hi, max(lo, best_v[k] + rng.uniform(-d, d)))
        obj, r = objective(v)
        if obj > best_obj:
            best_v, best_obj, best_r = v, obj, r
        if (i + 1) % 50 == 0:
            print(f"    iter {i + 1}/{iters}  best score={best_r['score']:.3f} "
                  f"(acc={best_r['balanced_acc']:.3f}, FP/min={best_r['fp_per_min']:.2f})  "
                  f"{time.time() - t0:.0f}s")

    # 2) ปรับละเอียดทีละพารามิเตอร์
    for _ in range(2):
        for k, (lo, hi) in SEARCH_SPACE.items():
            for x in np.linspace(lo, hi, 9):
                v = dict(best_v, **{k: float(x)})
                obj, r = objective(v)
                if obj > best_obj + 1e-9:
                    best_v, best_obj, best_r = v, obj, r
    return _from_vec(start, best_v), best_r


# ──────────────────────────────────────────────
# Scroll
# ──────────────────────────────────────────────
def _ema(x, a):
    out, s = np.empty(len(x)), None
    for i, v in enumerate(x):
        s = v if s is None else s + a * (v - s)
        out[i] = s
    return out


def tune_scroll(sessions, sp):
    """deadzone/full_speed จากขนาดการก้ม/เงยจริง และ noise ตอนนั่งนิ่ง"""
    peaks = {"nod_down": [], "nod_up": []}
    jitter = []
    for s in sessions:
        g = s.df
        sig = _ema(g.pitch.to_numpy(), sp.smoothing)
        t = g.t.to_numpy()
        for trial, label, t0, t1 in s.trials:
            gap = (g.trial.to_numpy() == trial) & (g.phase.to_numpy() == "gap") & (t >= t0 - 0.8)
            if gap.sum() < 5:
                continue
            base = float(np.median(sig[gap]))
            jitter.append(float(np.percentile(np.abs(sig[gap] - base), 95)))
            if label in peaks:
                go = (t >= t0) & (t <= t1)
                d = sig[go] - base
                peaks[label].append(float(np.percentile(d, 95 if label == "nod_down" else 5)))

    if not peaks["nod_down"] or not peaks["nod_up"]:
        print("  ⚠️ ไม่มีข้อมูลก้ม/เงย — คงค่า scroll เดิม")
        return sp, None
    down = float(np.median(peaks["nod_down"]))
    up = float(np.median(peaks["nod_up"]))
    jit = float(np.median(jitter)) if jitter else 0.005
    new = replace(sp)
    if down < 0 < up:
        # สัญญาณกลับทิศจากที่คาด (กล้องกลับหัว/วางมุมแปลก) → กลับทิศการเลื่อน
        new.invert = not sp.invert
        down, up = -down, -up
    amp = min(abs(down), abs(up))
    new.deadzone = round(max(3.0 * jit, 0.25 * amp), 4)
    new.full_speed = round(max(new.deadzone * 1.6, 0.8 * amp), 4)
    info = {"nod_down": down, "nod_up": up, "jitter95": jit,
            "n_down": len(peaks["nod_down"]), "n_up": len(peaks["nod_up"])}
    return new, info


# ──────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────
def signal_summary(sessions):
    """สรุปสัญญาณดิบต่อท่า — ใช้ดูว่าข้อมูลสมเหตุสมผลก่อนเชื่อผลจูน"""
    rows = []
    for s in sessions:
        g = s.df
        rl = g.ear_l.to_numpy() / s.ref_l
        rr = g.ear_r.to_numpy() / s.ref_r
        lo = np.minimum(rl, rr)
        t = g.t.to_numpy()
        for _, label, t0, t1 in s.trials:
            m = (t >= t0) & (t <= t1 + GRACE_S)
            if m.sum() < 3 or label == "idle":
                continue
            i = np.argmin(lo[m])
            dt = np.diff(t[m], prepend=t[m][0])
            rows.append({
                "label": label,
                "min_closed": lo[m][i],
                "other_eye": np.maximum(rl, rr)[m][i],
                "closed_s": float(dt[lo[m] < 0.6].sum()),
                "max_smile": float(g.delta_smile.to_numpy()[m].max()),
            })
    if not rows:
        return
    df = pd.DataFrame(rows).groupby("label").median(numeric_only=True)
    print("\n  สรุปสัญญาณ (median ต่อท่า, ratio = EAR ÷ ตาเปิด):")
    print("    min_closed = ตาที่ปิดลงต่ำสุด | other_eye = อีกตาตอนนั้น | closed_s = เวลาที่ ratio<0.6")
    print("    " + df.round(3).to_string().replace("\n", "\n    "))


def print_result(title, r):
    print(f"\n  {title}: score={r['score']:.3f}  balanced acc={r['balanced_acc']:.1%}  "
          f"FP={r['fp_per_min']:.2f}/นาที")
    for label in EXPECTED:
        if label in r["acc"]:
            conf = ", ".join(f"{k}×{n}" for k, n in r["confusion"][label].most_common())
            print(f"    {label:<13} {r['acc'][label]:6.1%}   [{conf}]")


def main():
    ap = argparse.ArgumentParser(description="จูนค่าการตรวจจับท่าทางจากข้อมูลจริง")
    ap.add_argument("--data", default=SIGNALS_PATH)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="ไม่บันทึกไฟล์")
    ap.add_argument("--eval-only", action="store_true", help="วัดผลค่าปัจจุบันอย่างเดียว")
    args = ap.parse_args()

    sessions = load_sessions(args.data)
    if not sessions:
        return
    n_trials = Counter(lb for s in sessions for _, lb, _, _ in s.trials)
    print(f"  📂 {len(sessions)} session, {sum(len(s.frames) for s in sessions)} เฟรม, "
          f"ท่า: {dict(n_trials)}")
    few = [k for k in EXPECTED if 0 < n_trials.get(k, 0) < 5]
    if few:
        print(f"  ⚠️ ข้อมูลน้อย (<5 ครั้ง): {few} — ผลจูนอาจไม่นิ่ง ควรอัดเพิ่ม")

    current_gp, current_sp = load_tuning()
    # ΔSmile ชดเชยการก้ม/เงย (เงยหน้าเดิมดูเหมือนยิ้ม → เลื่อนขึ้นไม่ได้)
    k = estimate_smile_pitch_comp(sessions)
    if k is not None:
        print(f"  😊 ค่าชดเชยการก้ม/เงยของ ΔSmile (จาก regression): {current_gp.smile_pitch_comp:.2f} → {k:.2f}")
        current_gp = replace(current_gp, smile_pitch_comp=k)
    sessions = [Session(s.sid, s.df, current_gp.smile_pitch_comp) for s in sessions]
    signal_summary(sessions)
    base = evaluate(sessions, current_gp)
    print_result("ค่าปัจจุบัน", base)
    default = evaluate(sessions, replace(GestureParams(), smile_pitch_comp=current_gp.smile_pitch_comp))
    print_result("ค่า default", default)
    if args.eval_only:
        return

    print(f"\n  🔎 ค้นหาพารามิเตอร์ ({args.iters} รอบ)...")
    start = current_gp if base["score"] >= default["score"] else replace(
        GestureParams(), smile_pitch_comp=current_gp.smile_pitch_comp)
    best_gp, best = search(sessions, start, args.iters, args.seed)
    print_result("ค่าที่จูนแล้ว", best)

    changed = {f.name: (getattr(current_gp, f.name), getattr(best_gp, f.name))
               for f in fields(GestureParams)
               if abs(float(getattr(current_gp, f.name)) - float(getattr(best_gp, f.name))) > 1e-6}
    if changed:
        print("\n  เปลี่ยนแปลง:")
        for k, (a, b) in changed.items():
            print(f"    {k:<20} {a:.3f} → {b:.3f}")

    new_sp, info = tune_scroll(sessions, current_sp)
    if info:
        print(f"\n  Scroll: ก้ม={info['nod_down']:+.4f} (n={info['n_down']})  "
              f"เงย={info['nod_up']:+.4f} (n={info['n_up']})  noise95={info['jitter95']:.4f}")
        print(f"    deadzone {current_sp.deadzone:.4f} → {new_sp.deadzone:.4f}   "
              f"full_speed {current_sp.full_speed:.4f} → {new_sp.full_speed:.4f}"
              + ("   (กลับทิศ)" if new_sp.invert != current_sp.invert else ""))

    if best["score"] < base["score"] - 1e-9:
        print("\n  ℹ️ ค่าที่ค้นหาได้ไม่ดีกว่าค่าปัจจุบัน — คงพารามิเตอร์ท่าทางเดิม")
        best_gp = current_gp

    if args.dry_run:
        print("\n  (dry-run: ไม่บันทึก)")
        return
    meta = {"updated": time.strftime("%Y-%m-%d %H:%M:%S"), "sessions": len(sessions),
            "trials": dict(n_trials), "score_before": round(base["score"], 4),
            "score_after": round(max(best["score"], base["score"]), 4),
            "acc_after": {k: round(v, 3) for k, v in best["acc"].items()},
            "fp_per_min_after": round(best["fp_per_min"], 3)}
    path = save_tuning(best_gp, new_sp, meta=meta)
    print(f"\n  ✅ บันทึก {path}\n     กด Start ใหม่เพื่อใช้ค่าที่จูนแล้ว")


if __name__ == "__main__":
    main()
