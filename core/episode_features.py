"""
EYE ORDER COME AI — Episode Features (ฟีเจอร์ของโมเดล Click Classification)

แนวทาง Hybrid:
  1. GestureDetector (rule) หา "episode" = ช่วงตั้งแต่เริ่มหลับตาจนลืมตา → บอกแค่ "จังหวะ"
  2. โมเดล ML จำแนกว่า episode นั้นคือท่าอะไร จากฟีเจอร์ในไฟล์นี้

จุดตัดสิน (decision point) มี 2 แบบ:
  - "end"  : ลืมตาครบแล้ว (episode สั้นกว่า drag_hold_s) → คลาส 0–3
  - "hold" : ยังหลับค้างจนถึง drag_hold_s → ใช้ข้อมูลถึงตอนนั้นตัดสินว่าเป็น drag หรือไม่

ไฟล์นี้ใช้ทั้งตอนใช้งานจริง (GestureDetector) และตอนเทรน (training/train_click_model.py)
→ ฟีเจอร์ตรงกันแน่นอน · pure python ไม่มี dependency (เร็วพอสำหรับ real-time)
"""

FEATURE_VERSION = 1

# คลาสของโมเดล (สอดคล้องกับ CLASS_NAMES 0–4 เดิม; ยิ้ม/scroll ใช้ threshold แยก)
EYE_CLASS_NAMES = {
    0: "No Action (Normal Blink)",
    1: "Left Wink",
    2: "Right Wink",
    3: "Double Blink",
    4: "Extended Wink (Drag)",
}
CLASS_TO_EVENT = {1: "left_click", 2: "right_click", 3: "double_click", 4: "drag_start"}
MIRROR_CLASS = {0: 0, 1: 2, 2: 1, 3: 3, 4: 4}

# ค่าคงที่ของฟีเจอร์ (ห้ามผูกกับพารามิเตอร์ที่จูน ไม่งั้นต้องเทรนใหม่ทุกครั้งที่จูน)
_ASYM_FRAME = 0.15      # ต่างกันเกินนี้ = เฟรมนั้นตาข้างหนึ่งปิดมากกว่าชัดเจน
_NO_PREV_GAP = 3.0      # ไม่มี episode ก่อนหน้า (หรือห่างเกินนี้) ใช้ค่านี้

FEATURE_NAMES = [
    "duration",          # วินาที (จุด hold = เวลาที่ค้างถึงตอนตัดสิน)
    "min_l", "min_r",    # ratio ต่ำสุดของแต่ละตา (1 = เปิดปกติ, 0 = ปิดสนิท)
    "mean_l", "mean_r",
    "min_lo",            # ตาที่ปิดมากกว่า ลงต่ำสุดแค่ไหน
    "min_hi",            # ตาที่เปิดกว้างกว่า ลงต่ำสุดแค่ไหน (ต่ำทั้งคู่ = กะพริบ)
    "asym_mean",         # mean(ratio_R − ratio_L): + = ตาซ้ายปิดมากกว่า
    "asym_abs_mean",
    "asym_at_min",       # ความต่างสองตา ณ จังหวะที่ปิดลึกสุด
    "frac_l_closed",     # สัดส่วนเฟรมที่ตาซ้ายปิดมากกว่าตาขวาชัดเจน
    "frac_r_closed",
    "close_speed",       # ratio ที่ลดลงต่อวินาทีจนถึงจุดต่ำสุด (กะพริบ = เร็วมาก)
    "smile_mean",        # ยิ้มระหว่าง episode (ยิ้มทำให้ตาหรี่)
    "prev_gap",          # วินาทีจาก episode ก่อนหน้าจบ → episode นี้เริ่ม
    "prev_duration",
    "prev_min_hi",
    "prev_asym_abs",
    "is_hold",           # 1 = ตัดสินที่จุด hold, 0 = ตัดสินตอนลืมตา
]


def episode_summary(frames, end_t):
    """ข้อมูลย่อของ episode ที่จบแล้ว เก็บไว้เป็น 'ก่อนหน้า' ของ episode ถัดไป"""
    if not frames:
        return None
    return {
        "end": end_t,
        "duration": end_t - frames[0][0],
        "min_hi": min(max(rl, rr) for _, rl, rr, _ in frames),
        "asym_abs": sum(abs(rr - rl) for _, rl, rr, _ in frames) / len(frames),
    }


def compute_features(frames, start_t, decision_t, prev=None, is_hold=False):
    """
    Args:
        frames: list ของ (t, ratio_l, ratio_r, delta_smile) ระหว่าง episode
        start_t: เวลาเริ่ม episode
        decision_t: เวลาที่ตัดสิน (ลืมตา หรือ ถึง drag_hold_s)
        prev: episode_summary() ของ episode ก่อนหน้า หรือ None
        is_hold: ตัดสินที่จุด hold หรือไม่

    Returns:
        list[float] ตามลำดับ FEATURE_NAMES
    """
    n = len(frames)
    if n == 0:
        frames = [(start_t, 1.0, 1.0, 0.0)]
        n = 1
    rls = [f[1] for f in frames]
    rrs = [f[2] for f in frames]
    asym = [r - l for l, r in zip(rls, rrs)]
    los = [min(l, r) for l, r in zip(rls, rrs)]
    i_min = min(range(n), key=los.__getitem__)
    t_min = frames[i_min][0]
    drop = max(0.0, 1.0 - los[i_min])
    close_speed = min(drop / max(t_min - start_t, 1.0 / 30.0), 30.0)

    if prev is not None and start_t - prev["end"] < _NO_PREV_GAP:
        p_gap, p_dur = start_t - prev["end"], prev["duration"]
        p_hi, p_asym = prev["min_hi"], prev["asym_abs"]
    else:
        p_gap, p_dur, p_hi, p_asym = _NO_PREV_GAP, 0.0, 1.0, 0.0

    return [
        decision_t - start_t,
        min(rls), min(rrs),
        sum(rls) / n, sum(rrs) / n,
        los[i_min],
        min(max(l, r) for l, r in zip(rls, rrs)),
        sum(asym) / n,
        sum(abs(a) for a in asym) / n,
        asym[i_min],
        sum(a > _ASYM_FRAME for a in asym) / n,
        sum(a < -_ASYM_FRAME for a in asym) / n,
        close_speed,
        sum(f[3] for f in frames) / n,
        p_gap, p_dur, p_hi, p_asym,
        1.0 if is_hold else 0.0,
    ]


def mirror_frames(frames):
    """สลับตาซ้าย↔ขวา (ใช้ทำ data augmentation: ขยิบซ้าย ↔ ขยิบขวา)"""
    return [(t, rr, rl, sm) for t, rl, rr, sm in frames]
