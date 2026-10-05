"""
EYE ORDER COME AI — Baseline Recorder
บันทึกค่าฐาน (neutral face) สำหรับ:
  - Mouth width baseline → ใช้คำนวณ ΔSmile
  - EAR baseline → ใช้ reference
"""
import time
import json
import numpy as np
import cv2

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.camera import CameraStream
from core.face_mesh import FaceMeshDetector
from core.feature_extractor import FeatureExtractor
from config.settings import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, FPS,
    BASELINE_DURATION_SEC, BASELINE_DATA_PATH
)


class BaselineRecorder:
    """บันทึกค่าฐานจากหน้าปกติของผู้ใช้
    
    ให้ผู้ใช้นั่งมองจอด้วยสีหน้าปกติ 3 วินาที
    แล้วบันทึกค่าเฉลี่ย EAR และ mouth width เป็นค่าฐาน
    """
    
    def __init__(self):
        self.camera = CameraStream(CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, FPS)
        self.detector = FaceMeshDetector()
        self.extractor = FeatureExtractor(self.detector)
    
    def record(self, duration_sec=None, show_preview=True):
        """บันทึกค่าฐาน
        
        Args:
            duration_sec: ระยะเวลาบันทึก (วินาที), None = ใช้ค่าจาก settings
            show_preview: แสดงหน้าต่าง preview หรือไม่
        
        Returns:
            dict: {
                "baseline_mouth_width": float,
                "baseline_ear_l": float,
                "baseline_ear_r": float,
                "num_frames": int,
            }
        """
        if duration_sec is None:
            duration_sec = BASELINE_DURATION_SEC
        
        print("=" * 60)
        print("   BASELINE RECORDING")
        print("   นั่งมองจอด้วยสีหน้าปกติ ไม่ยิ้ม ไม่กะพริบตา")
        print(f"   จะบันทึกเป็นเวลา {duration_sec} วินาที")
        print("=" * 60)
        print("\n  กด SPACE เพื่อเริ่มบันทึก...")
        
        self.camera.start()
        
        # รอให้ผู้ใช้กด SPACE
        while True:
            frame = self.camera.read()
            if frame is not None and show_preview:
                display = frame.copy()
                cv2.putText(display, "Press SPACE to start baseline recording",
                           (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                           (0, 255, 255), 2)
                cv2.putText(display, "Keep a neutral face - no smile, no blink",
                           (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                           (200, 200, 200), 1)
                cv2.imshow("Baseline Recording", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                break
            elif key == 27:  # ESC
                self.camera.stop()
                cv2.destroyAllWindows()
                return None
        
        print("\n  ▶ กำลังบันทึกค่าฐาน...")
        
        # บันทึกค่า
        mouth_widths = []
        ear_ls = []
        ear_rs = []
        
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < duration_sec:
            frame = self.camera.read()
            if frame is None:
                continue
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = self.detector.detect(rgb)
            
            if landmarks is not None:
                # EAR
                click_feat = self.extractor.extract_click_features(landmarks)
                if click_feat is not None:
                    ear_ls.append(click_feat[0])
                    ear_rs.append(click_feat[1])
                
                # Mouth width
                mw = self.extractor.compute_baseline_mouth_width(landmarks)
                if mw is not None:
                    mouth_widths.append(mw)
                
                frame_count += 1
            
            if show_preview:
                elapsed = time.time() - start_time
                progress = elapsed / duration_sec
                display = frame.copy()
                
                # Progress bar
                bar_width = 400
                bar_x = 120
                bar_y = 40
                cv2.rectangle(display, (bar_x, bar_y), 
                             (bar_x + bar_width, bar_y + 25), (50, 50, 50), -1)
                cv2.rectangle(display, (bar_x, bar_y),
                             (bar_x + int(bar_width * progress), bar_y + 25),
                             (0, 255, 0), -1)
                cv2.putText(display, f"Recording... {elapsed:.1f}s / {duration_sec}s",
                           (30, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                           (0, 255, 0), 2)
                cv2.putText(display, f"Frames: {frame_count}",
                           (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                           (200, 200, 200), 1)
                cv2.imshow("Baseline Recording", display)
                cv2.waitKey(1)
        
        self.camera.stop()
        cv2.destroyAllWindows()
        
        if len(mouth_widths) == 0:
            print("  ❌ ไม่สามารถตรวจจับใบหน้าได้ กรุณาลองใหม่")
            return None
        
        # คำนวณค่าเฉลี่ย
        baseline = {
            "baseline_mouth_width": float(np.mean(mouth_widths)),
            "baseline_ear_l": float(np.mean(ear_ls)),
            "baseline_ear_r": float(np.mean(ear_rs)),
            "num_frames": frame_count,
        }
        
        # บันทึกลงไฟล์
        os.makedirs(os.path.dirname(BASELINE_DATA_PATH), exist_ok=True)
        with open(BASELINE_DATA_PATH, 'w') as f:
            json.dump(baseline, f, indent=2)
        
        print(f"\n  ✅ บันทึกค่าฐานเสร็จสิ้น ({frame_count} frames)")
        print(f"     Mouth width: {baseline['baseline_mouth_width']:.6f}")
        print(f"     EAR_L:       {baseline['baseline_ear_l']:.4f}")
        print(f"     EAR_R:       {baseline['baseline_ear_r']:.4f}")
        print(f"     บันทึกไว้ที่: {BASELINE_DATA_PATH}")
        
        return baseline
    
    @staticmethod
    def load_baseline():
        """โหลดค่าฐานจากไฟล์
        
        Returns:
            dict หรือ None
        """
        if os.path.exists(BASELINE_DATA_PATH):
            with open(BASELINE_DATA_PATH, 'r') as f:
                return json.load(f)
        return None


if __name__ == "__main__":
    recorder = BaselineRecorder()
    result = recorder.record()
    if result:
        print("\n  ค่าฐานที่บันทึกได้:")
        for k, v in result.items():
            print(f"    {k}: {v}")
