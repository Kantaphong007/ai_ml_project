# EYE ORDER COME AI (ตาสั่งมา)

ควบคุมเมาส์ด้วยการหันศีรษะ และสั่งคลิกด้วยการขยิบตา/ยิ้ม ผ่านกล้องเว็บแคม — ไม่ต้องใช้มือ

| ท่าทาง | คำสั่ง |
|---|---|
| หันศีรษะ | เลื่อนเคอร์เซอร์ |
| ขยิบตาซ้าย / ขวา (สั้นๆ) | คลิกซ้าย / คลิกขวา |
| กะพริบ/ขยิบเร็วๆ 2 ครั้ง (ตาข้างเดิม หรือสองตาพร้อมกัน) | ดับเบิลคลิก |
| หลับตาข้างเดียวค้าง 1.2 วิ → หันหัวลาก → ลืมตา | ลากวาง (drag) |
| ยิ้มค้าง + ก้ม/เงย → หุบยิ้ม | เลื่อนหน้าจอ (scroll) ลง/ขึ้น → ออก |
| กะพริบตาปกติ | ไม่ทำอะไร |

---

## ติดตั้ง

ต้องใช้ Python 3.9–3.12 (mediapipe 0.10.14)

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python main.py          # GUI
.venv/bin/python main.py --cli    # เมนู CLI
```

โมเดล Face Landmarker (`models/face_landmarker.task`) ดาวน์โหลดอัตโนมัติครั้งแรกถ้ายังไม่มี

**macOS:** System Settings → Privacy & Security → เปิดสิทธิ์ **Camera** และ **Accessibility**
ให้แอปที่ใช้รัน (Terminal / VS Code) แล้วปิด-เปิดแอปนั้นใหม่ (ไม่ได้สิทธิ์ Accessibility = เมาส์ไม่ขยับ)

## ใช้งาน

1. **📐 Baseline** — นั่งหน้าปกติ 3 วินาที (ค่าตาเปิด/ปากปกติ)
2. **▶ Start** — ใช้ได้ทันทีในโหมด relative (มุมหน้าตรงตอนทำ Baseline = กลางจอ)

ทำเพิ่มเพื่อเปิดความสามารถ ML เต็มรูปแบบ:

| ขั้น | ปุ่ม | ได้อะไร |
|---|---|---|
| 3 | 🎯 Calibrate → 🏋️ Train Cursor | เปิด AI ช่วยเคอร์เซอร์ (โหมด hybrid / absolute) |
| 4 | 🎙 Record Signals (2–3 รอบ) → 🔧 Tune → 🏋️ Train Click | จูนการตรวจจับท่าทาง + โมเดลจำแนกท่าตา |
| 5 | 📊 Evaluate | รายงาน + กราฟ (`data/plots/`) |

**Settings:** Cursor Mode · Smoothing · Cursor Speed (ความไว) · Stability (ความนิ่งตอนหยุด) · Mirror ·
ช่อง **Decision** บนหน้าจอบอกเหตุผลทุกครั้งว่าทำไมท่าล่าสุดคลิก/ไม่คลิก

---

## ส่วนที่เป็น Machine Learning

### 1. เคอร์เซอร์ — Regression + Hybrid pointing

- **อินพุต:** มุมศีรษะจาก *facial transformation matrix* ของ MediaPipe Face Landmarker
  (fit โมเดลหน้า 3 มิติทั้งหน้า noise ≈ 0.04–0.08°) + ตำแหน่งปลายจมูกเทียบกรอบหน้า
- **โมเดล:** เทียบ Ridge / SVR / Random Forest / Gradient Boosting / MLP ด้วย GridSearchCV
  + **GroupKFold แยกตามการเยี่ยมจุด** (กันข้อมูลรั่วระหว่าง train/test)
- **Feature selection:** เทียบชุดฟีเจอร์อัตโนมัติด้วย CV แล้วเลือกชุด MAE ต่ำสุด
- **EdgeSafeRegressor:** ทำนาย = ส่วนเชิงเส้น + ส่วนแก้ความโค้งเฉพาะในช่วงที่ calibrate
  → หันเลยขอบจอแล้วเคอร์เซอร์ติดขอบ (SVR/MLP เดิมวิ่งกลับเข้ากลางจอ)
- **โหมดใช้งาน** (`core/head_pointer.py`):
  - **Hybrid (ค่าเริ่มต้น)** — relative + pointer acceleration สำหรับเล็งละเอียด และใช้โมเดล regression
    บอก "ตำแหน่งที่หน้าหันไป" เพื่อกันเคอร์เซอร์หลุดจากทิศหน้า (แนวคิดใกล้ *MAGIC pointing*,
    Zhai et al., CHI 1999: ทิศที่มองพาไปใกล้เป้าแบบหยาบ ผู้ใช้เล็งละเอียดเอง)
  - **Relative** — เหมือน head mouse ทั่วไป ไม่ต้อง calibrate (ตำแหน่งที่หน้าหันไปประมาณจาก
    มุมหน้าตรงใน Baseline + ช่วงที่ผู้ใช้หันจริงซึ่งเรียนระหว่างใช้งาน)
  - **Gain modulation (ทั้งสองโหมด):** relative สะสมความคลาดได้ (หันเลยขอบ / หันออกเร็วกลับช้า)
    แก้โดยปรับ "ความเร็ว" ไม่ใช่ดึงตำแหน่ง — หันเข้าหาตำแหน่งที่หน้าหันไป = เร็วขึ้น,
    หันออก = ช้าลง/หยุดรอ → เคอร์เซอร์ขยับทิศเดียวกับหัวเสมอ ไม่มีการยึกกลับ ไม่ขยับเองตอนหัวนิ่ง
    และห่างจากทิศหน้าได้ไม่เกิน ~180–230 px (มีเทสต์ยืนยันทั้งสองข้อ)
  - **Absolute** — โมเดล regression ล้วน + One-Euro filter + deadzone ที่ปรับตาม noise ที่โมเดลวัดได้

### 2. คลิก — Hybrid episode classifier

- **Detector (rule):** หา "episode" การหลับตา จาก EAR ÷ ค่าตาเปิดที่เรียนต่อเนื่อง (ทนต่อการก้ม/เงย/แสง)
- **ดับเบิลคลิก:** (1) ตาปิด 2 รอบในช่วงเดียว — ปิด → เปิดขึ้นครึ่งทาง → ปิด (รูปตัว W)
  วัดจากข้อมูลจริง: ท่าอื่นเปิดขึ้นระหว่างหลับตาไม่เกิน 0.04 ส่วนดับเบิลคลิกถึง 0.29 → เกณฑ์ 0.06
  (2) ปิดตาสั้นๆ แบบเดียวกัน 2 ครั้งติดกัน — ชั้นลำดับเวลาที่ทำงานต่อจากการตัดสินของ ML/rule
  และบน macOS ส่ง event พร้อม click count = 2 (แอป Mac ไม่นับคลิกเดี่ยว 2 ครั้งเป็นดับเบิลคลิก)
- **Classifier (ML):** 19 ฟีเจอร์ต่อ episode (ระยะเวลา, ระดับปิดแต่ละตา, ความต่างสองตา, ความเร็วปิด,
  ช่วงห่างจาก episode ก่อนหน้า ฯลฯ) → 5 คลาส: ไม่ทำอะไร / ขยิบซ้าย / ขยิบขวา / ดับเบิลคลิก / drag
- **Data augmentation:** สลับตาซ้าย↔ขวา (ข้อมูล ×2)
- **ประเมิน 2 ระดับ:** episode (out-of-fold F1) และ **ใช้งานจริง** (replay สัญญาณทั้งระบบ
  แบบ leave-one-session-out เทียบ rule-based: ความแม่นต่อท่า + คลิกผิด/นาที)
- ML ไม่มั่นใจ → ใช้ rule ตัดสินแทน · ถ้า ML แพ้ rule-based ตอนประเมิน ระบบเลือก rule ให้อัตโนมัติ

### 3. จูนพารามิเตอร์จากข้อมูล (`training/tune_gestures.py`)

replay สัญญาณที่อัดไว้ผ่าน detector ตัวจริง → random search + coordinate refinement หา threshold
ที่ให้ balanced accuracy สูงสุดและคลิกผิดน้อยสุด, ประมาณค่าชดเชยการก้ม/เงยของสัญญาณยิ้ม
ด้วย regression, และคำนวณ deadzone ของ scroll จากขนาดการก้ม/เงยจริง

---

## โครงสร้าง

```
main.py                     GUI / CLI
config/   settings.py       ค่าคงที่ + การตั้งค่าผู้ใช้
          tuning.py         พารามิเตอร์ที่จูนจากข้อมูล (data/gesture_tuning.json)
core/     pipeline.py       กล้อง → ใบหน้า → ฟีเจอร์ → เคอร์เซอร์/ท่าทาง → เมาส์
          face_mesh.py      MediaPipe Face Landmarker + มุมศีรษะ
          feature_extractor.py  EAR, ΔSmile, ฟีเจอร์เคอร์เซอร์, สัญญาณก้ม/เงย
          head_pointer.py   โหมด relative / hybrid
          cursor_predictor.py, cursor_models.py   โหมด absolute (regression)
          gesture_detector.py, episode_features.py, click_classifier.py   คลิก
          smile.py          ΔSmile ชดเชยการก้ม/เงย
          mouse_controller.py   Win32 / macOS Quartz / pyautogui
calibration/                Baseline, Calibrate (13 จุด), Record Signals
training/                   Train Cursor, Tune, Train Click, Evaluate
gui/                        หน้าต่างหลัก, Settings, overlay, คีย์บอร์ดบนจอ
tests/                      pytest (.venv/bin/python -m pytest tests -q)
data/ models/               ข้อมูลผู้ใช้ / โมเดลที่เทรนแล้ว
```
