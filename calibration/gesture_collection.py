"""
EYE ORDER COME AI — Prompt-Guided Gesture Collection
ระบบเก็บข้อมูลท่าทางใบหน้า (Classification Dataset)

แสดงคำแนะนำท่าทาง + ภาพตัวอย่าง + แถบนับถอยหลัง
สุ่มแสดงคำสั่ง 6 คลาส, ท่าละ 20-30 รอบ

บันทึก [EAR_L, EAR_R, ΔSmile] × 30 เฟรม + class label
"""
import time
import os
import sys
import json
import random
import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.camera import CameraStream
from core.face_mesh import FaceMeshDetector
from core.feature_extractor import FeatureExtractor
from utils.sliding_queue import SlidingQueue
from calibration.baseline_recorder import BaselineRecorder
from config.settings import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, FPS,
    SLIDING_WINDOW_SIZE, GESTURE_REPS_PER_CLASS,
    NUM_GESTURE_CLASSES, CLASS_NAMES,
    GESTURE_DATA_PATH, BASELINE_DATA_PATH
)


# คำอธิบายและคำแนะนำสำหรับแต่ละ gesture class
GESTURE_INSTRUCTIONS = {
    0: {
        "name": "Normal Blink",
        "thai": "กะพริบตาธรรมชาติ",
        "instruction": "กะพริบตาตามปกติ 1 ครั้ง (เร็วๆ)",
        "color": (200, 200, 200),  # Gray
    },
    1: {
        "name": "Left Wink",
        "thai": "ขยิบตาซ้าย",
        "instruction": "หลับตาซ้ายค้างไว้ ~0.5 วินาที ตาขวาลืมปกติ",
        "color": (255, 100, 100),  # Blue
    },
    2: {
        "name": "Right Wink",
        "thai": "ขยิบตาขวา",
        "instruction": "หลับตาขวาค้างไว้ ~0.5 วินาที ตาซ้ายลืมปกติ",
        "color": (100, 255, 100),  # Green
    },
    3: {
        "name": "Double Blink",
        "thai": "กะพริบตาสองครั้ง",
        "instruction": "กะพริบตาเร็วๆ 2 ครั้งติดกัน",
        "color": (100, 100, 255),  # Red
    },
    4: {
        "name": "Extended Wink",
        "thai": "ขยิบตาค้างยาว (Drag)",
        "instruction": "หลับตาข้างใดข้างหนึ่งค้างนาน >1 วินาที",
        "color": (0, 200, 255),  # Orange
    },
    5: {
        "name": "Smile + Nod",
        "thai": "ยิ้ม + พยักหน้า (Scroll)",
        "instruction": "ยิ้มกว้าง (เห็นฟัน) ค้างไว้",
        "color": (255, 0, 255),  # Magenta
    },
}


class GestureCollector:
    """เก็บข้อมูลท่าทางใบหน้าสำหรับ classification model
    
    Flow:
        1. โหลดค่าฐาน (baseline) → ถ้ายังไม่มีจะบันทึกใหม่
        2. สุ่มแสดงคำสั่งท่าทาง
        3. ให้ผู้ใช้ทำท่าทาง
        4. บันทึก sliding window 30 เฟรม + class label
        5. ทำซ้ำจนครบจำนวน
    """
    
    def __init__(self):
        self.camera = CameraStream(CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, FPS)
        self.detector = FaceMeshDetector()
        self.extractor = FeatureExtractor(self.detector)
        self.sliding_queue = SlidingQueue(
            window_size=SLIDING_WINDOW_SIZE, num_features=3
        )
    
    def _load_or_record_baseline(self):
        """โหลดค่าฐาน, ถ้ายังไม่มีจะบันทึกใหม่"""
        baseline = BaselineRecorder.load_baseline()
        if baseline is not None:
            print(f"  📂 โหลดค่าฐานจาก: {BASELINE_DATA_PATH}")
            self.extractor.set_baseline_mouth_ratio(baseline.get("baseline_mouth_ratio"))
            return baseline
        
        print("  ⚠️ ยังไม่มีค่าฐาน จะบันทึกใหม่...")
        recorder = BaselineRecorder()
        baseline = recorder.record()
        if baseline is not None:
            self.extractor.set_baseline_mouth_ratio(baseline.get("baseline_mouth_ratio"))
        return baseline
    
    def run(self, reps_per_class=None, classes=None):
        """เริ่มเก็บข้อมูลท่าทาง
        
        Args:
            reps_per_class: จำนวนรอบต่อคลาส (None = ใช้จาก settings)
            classes: list ของ class ที่ต้องการเก็บ (None = ทุกคลาส)
        
        Returns:
            pd.DataFrame: ข้อมูลที่เก็บได้ หรือ None ถ้ายกเลิก
        """
        if reps_per_class is None:
            reps_per_class = GESTURE_REPS_PER_CLASS
        if classes is None:
            classes = list(range(NUM_GESTURE_CLASSES))
        
        print("=" * 60)
        print("   GESTURE COLLECTION")
        print(f"   คลาส: {len(classes)} คลาส × {reps_per_class} รอบ")
        print(f"   Window: {SLIDING_WINDOW_SIZE} เฟรม = {SLIDING_WINDOW_SIZE/FPS:.1f} วินาที")
        print("=" * 60)
        
        # โหลดค่าฐาน
        baseline = self._load_or_record_baseline()
        if baseline is None:
            print("  ❌ ไม่สามารถบันทึกค่าฐานได้")
            return None
        
        self.camera.start()
        self.extractor.set_frame_size(*self.camera.get_frame_size())
        
        # สร้างรายการท่าทางที่ต้องเก็บ (สุ่มลำดับ)
        gesture_queue = []
        for cls in classes:
            gesture_queue.extend([cls] * reps_per_class)
        random.shuffle(gesture_queue)
        
        all_data = []
        total_tasks = len(gesture_queue)
        
        # สร้างชื่อ columns
        feature_names = FeatureExtractor.get_click_sliding_feature_names(
            SLIDING_WINDOW_SIZE
        )
        
        print(f"\n  ทั้งหมด {total_tasks} ท่าทาง")
        print("  กด SPACE เพื่อเริ่ม, ESC เพื่อยกเลิก\n")
        
        task_idx = 0
        
        for gesture_class in gesture_queue:
            task_idx += 1
            info = GESTURE_INSTRUCTIONS[gesture_class]
            
            # ── Phase 1: แสดงคำแนะนำ + รอผู้ใช้พร้อม ──
            while True:
                frame = self.camera.read()
                if frame is not None:
                    display = frame.copy()
                    h, w = display.shape[:2]
                    
                    # พื้นหลังคำแนะนำ
                    overlay = display.copy()
                    cv2.rectangle(overlay, (0, 0), (w, 120), (30, 30, 30), -1)
                    cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
                    
                    # ข้อความ
                    cv2.putText(display,
                               f"[{task_idx}/{total_tasks}] Class {gesture_class}: "
                               f"{info['name']}",
                               (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                               info["color"], 2)
                    cv2.putText(display, info["thai"],
                               (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                               (255, 255, 255), 1)
                    cv2.putText(display, info["instruction"],
                               (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                               (180, 180, 180), 1)
                    cv2.putText(display, "Press SPACE when ready, ESC to quit",
                               (15, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                               (0, 255, 255), 1)
                    
                    cv2.imshow("Gesture Collection", display)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    break
                elif key == 27:
                    self._cleanup()
                    return self._save_data(all_data, feature_names) if all_data else None
            
            # ── Phase 2: Countdown 3-2-1 (1 วินาทีต่อตัวเลข ให้เวลาเตรียมตัว) ──
            for countdown in [3, 2, 1]:
                countdown_start = time.time()
                while time.time() - countdown_start < 1.0:
                    frame = self.camera.read()
                    if frame is not None:
                        display = frame.copy()
                        h, w = display.shape[:2]
                        
                        # Banner คำสั่งด้านบน
                        overlay = display.copy()
                        cv2.rectangle(overlay, (0, 0), (w, 80), (30, 30, 30), -1)
                        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
                        cv2.putText(display, f"Get Ready for: {info['name']} ({info['thai']})",
                                   (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, info["color"], 2)
                        cv2.putText(display, info["instruction"],
                                   (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                        
                        # ตัวเลขนับถอยหลังใหญ่กลางจอ
                        cv2.putText(display, str(countdown),
                                   (w // 2 - 35, h // 2 + 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 4.0,
                                   (0, 255, 255), 6)
                        cv2.imshow("Gesture Collection", display)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:
                        self._cleanup()
                        return self._save_data(all_data, feature_names) if all_data else None
            
            # ── Phase 3: ให้ทำท่าทาง + บันทึก (ให้เวลาบันทึก 2.5 วินาที) ──
            print(f"  ▶ [{task_idx}/{total_tasks}] Recording Class {gesture_class} "
                  f"({info['name']})...")
            
            self.sliding_queue.clear()
            recording_start = time.time()
            recording_duration = 2.5  # ให้เวลาบันทึก 2.5 วินาที
            frames_collected = 0
            all_frame_features = []
            
            while time.time() - recording_start < recording_duration:
                frame = self.camera.read()
                if frame is None:
                    continue
                
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                landmarks = self.detector.detect(rgb)
                
                if landmarks is not None:
                    click_feat = self.extractor.extract_click_features(landmarks)
                    if click_feat is not None:
                        all_frame_features.append(click_feat)
                        frames_collected += 1
                
                # Display
                display = frame.copy()
                h, w = display.shape[:2]
                
                elapsed = time.time() - recording_start
                progress = min(elapsed / recording_duration, 1.0)
                
                # Banner สถานะกำลังบันทึก
                cv2.circle(display, (25, 25), 10, (0, 0, 255), -1)
                cv2.putText(display, f"REC — DO GESTURE NOW: {info['name']}",
                           (45, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                           (0, 0, 255), 2)
                cv2.putText(display, info["instruction"],
                           (45, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                           (255, 255, 255), 1)
                
                # Progress bar ขนาดใหญ่ล่างจอ
                bar_w = w - 40
                cv2.rectangle(display, (20, h - 35), (20 + bar_w, h - 15),
                             (50, 50, 50), -1)
                cv2.rectangle(display, (20, h - 35),
                             (20 + int(bar_w * progress), h - 15),
                             info["color"], -1)
                
                remaining_sec = max(0.0, recording_duration - elapsed)
                cv2.putText(display, f"Time left: {remaining_sec:.1f}s",
                           (w // 2 - 50, h - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                           (255, 255, 255), 1)
                
                cv2.imshow("Gesture Collection", display)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    self._cleanup()
                    return self._save_data(all_data, feature_names) if all_data else None
            
            # ดึง 30 เฟรมที่ดีที่สุดจากทั้งหมดที่บันทึกได้
            if len(all_frame_features) >= SLIDING_WINDOW_SIZE:
                # เลือก 30 เฟรมต่อเนื่องที่มีสัญญาณ gesture เด่นชัดที่สุด (เช่น EAR ต่ำสุด หรือ Smile สูงสุด)
                feat_arr = np.array(all_frame_features)  # (N, 3) -> [EAR_L, EAR_R, ΔSmile]
                
                if gesture_class in (1, 4):  # Left Wink / Extended Left Wink -> min EAR_L
                    scores = [np.mean(feat_arr[i:i+SLIDING_WINDOW_SIZE, 0]) 
                              for i in range(len(feat_arr) - SLIDING_WINDOW_SIZE + 1)]
                    best_start = int(np.argmin(scores))
                elif gesture_class in (2, 4):  # Right Wink -> min EAR_R
                    scores = [np.mean(feat_arr[i:i+SLIDING_WINDOW_SIZE, 1]) 
                              for i in range(len(feat_arr) - SLIDING_WINDOW_SIZE + 1)]
                    best_start = int(np.argmin(scores))
                elif gesture_class in (0, 3):  # Blink / Double Blink -> min (EAR_L + EAR_R)
                    scores = [np.mean(feat_arr[i:i+SLIDING_WINDOW_SIZE, :2]) 
                              for i in range(len(feat_arr) - SLIDING_WINDOW_SIZE + 1)]
                    best_start = int(np.argmin(scores))
                elif gesture_class == 5:  # Smile -> max ΔSmile
                    scores = [np.mean(feat_arr[i:i+SLIDING_WINDOW_SIZE, 2]) 
                              for i in range(len(feat_arr) - SLIDING_WINDOW_SIZE + 1)]
                    best_start = int(np.argmax(scores))
                else:
                    best_start = 0
                
                selected_30 = feat_arr[best_start:best_start + SLIDING_WINDOW_SIZE]
                flat_features = selected_30.flatten()
                row = list(flat_features) + [gesture_class]
                all_data.append(row)
                print(f"    ✅ บันทึกสำเร็จ ({len(all_frame_features)} frames recorded, selected best 30 window)")
            else:
                print(f"    ⚠️ เก็บไม่ครบ ({len(all_frame_features)}/{SLIDING_WINDOW_SIZE}) - ข้าม")
            
            # พักแป๊บหนึ่งระหว่างรอบ
            time.sleep(0.4)
        
        self._cleanup()
        
        if all_data:
            return self._save_data(all_data, feature_names)
        return None
    
    def _save_data(self, all_data, feature_names):
        """บันทึกข้อมูลเป็น CSV"""
        columns = feature_names + ["class"]
        df = pd.DataFrame(all_data, columns=columns)
        
        # ถ้ามีไฟล์เก่า ให้ต่อท้าย
        os.makedirs(os.path.dirname(GESTURE_DATA_PATH), exist_ok=True)
        if os.path.exists(GESTURE_DATA_PATH):
            existing = pd.read_csv(GESTURE_DATA_PATH)
            df = pd.concat([existing, df], ignore_index=True)
            print(f"\n  📎 ต่อท้ายข้อมูลเก่า (รวม {len(df)} ตัวอย่าง)")
        
        df.to_csv(GESTURE_DATA_PATH, index=False)
        
        print(f"\n  ✅ Gesture Collection เสร็จสิ้น!")
        print(f"     ข้อมูลทั้งหมด: {len(df)} ตัวอย่าง")
        print(f"     บันทึกไว้ที่: {GESTURE_DATA_PATH}")
        print(f"\n  จำนวนต่อคลาส:")
        class_counts = df["class"].value_counts().sort_index()
        for cls, count in class_counts.items():
            cls_int = int(cls)
            name = CLASS_NAMES.get(cls_int, f"Class {cls_int}")
            print(f"     Class {cls_int} ({name}): {count}")
        
        return df
    
    def _cleanup(self):
        """ปิด resources"""
        self.camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Gesture Collection")
    parser.add_argument("--reps", type=int, default=GESTURE_REPS_PER_CLASS,
                       help=f"จำนวนรอบต่อคลาส (default: {GESTURE_REPS_PER_CLASS})")
    parser.add_argument("--classes", type=int, nargs="+", default=None,
                       help="คลาสที่ต้องการเก็บ (default: ทุกคลาส)")
    args = parser.parse_args()
    
    collector = GestureCollector()
    result = collector.run(reps_per_class=args.reps, classes=args.classes)
