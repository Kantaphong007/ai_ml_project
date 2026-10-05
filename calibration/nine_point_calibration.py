"""
EYE ORDER COME AI — 9-Point Calibration
ระบบ Calibration สำหรับเก็บข้อมูลความสัมพันธ์ระหว่าง
ตำแหน่งปลายจมูก/การหันหัว → พิกัดหน้าจอ (Regression Dataset)

แสดงจุดเป้าหมาย 9 จุดทีละจุด:
  - มุมจอ 4 จุด
  - กึ่งกลางขอบจอ 4 จุด
  - จุดกึ่งกลางจอ 1 จุด

แต่ละจุด: หน่วง 0.5 วินาที → บันทึก 1.5 วินาที (45 เฟรม/จุด)
รวม ~405 ตัวอย่าง/เซสชัน
"""
import time
import os
import sys
import random
import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.camera import CameraStream
from core.face_mesh import FaceMeshDetector
from core.feature_extractor import FeatureExtractor
from config.settings import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, FPS,
    SCREEN_WIDTH, SCREEN_HEIGHT,
    NINE_POINTS, CALIBRATION_DELAY_SEC, CALIBRATION_DURATION_SEC,
    CALIBRATION_DATA_PATH, load_user_settings
)


class NinePointCalibration:
    """9-Point Calibration สำหรับเก็บข้อมูล cursor regression
    
    Flow:
        1. แสดง fullscreen สีดำ
        2. แสดงจุดเป้าหมายสีแดงทีละจุด (สุ่มลำดับ)
        3. ผู้ใช้จ้องมองจุด
        4. หน่วง 0.5 วินาที → บันทึก 1.5 วินาที
        5. ทำซ้ำจนครบ 9 จุด
        6. บันทึก CSV
    """
    
    def __init__(self):
        # ใช้ค่า mirror เดียวกับตอนใช้งานจริง ไม่งั้นฟีเจอร์ซ้าย-ขวาจะกลับด้านกับโมเดลที่เทรนไว้
        mirror = load_user_settings().get("mirror", True)
        self.camera = CameraStream(CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, FPS, mirror=mirror)
        self.detector = FaceMeshDetector()
        self.extractor = FeatureExtractor(self.detector)
        
        # สร้างตำแหน่ง pixel จริงจาก normalized positions
        self.target_points = [
            (int(nx * SCREEN_WIDTH), int(ny * SCREEN_HEIGHT))
            for nx, ny in NINE_POINTS
        ]
    
    def run(self, num_rounds=2, show_camera=True):
        """เริ่ม calibration
        
        Args:
            num_rounds: จำนวนรอบ (ทำซ้ำ 9 จุด กี่รอบ)
            show_camera: แสดง camera preview ข้างๆ หรือไม่
        
        Returns:
            pd.DataFrame: ข้อมูลที่เก็บได้ หรือ None ถ้ายกเลิก
        """
        print("=" * 60)
        print("   9-POINT CALIBRATION")
        print(f"   หน้าจอ: {SCREEN_WIDTH} x {SCREEN_HEIGHT}")
        print(f"   จำนวนจุด: {len(self.target_points)} จุด × {num_rounds} รอบ")
        print("=" * 60)
        
        self.camera.start()
        
        all_data = []
        feature_names = FeatureExtractor.get_cursor_feature_names()
        
        # สร้าง fullscreen window
        cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN,
                             cv2.WINDOW_FULLSCREEN)
        
        if show_camera:
            cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Camera", 320, 240)
        
        for round_idx in range(num_rounds):
            print(f"\n  ── รอบที่ {round_idx + 1}/{num_rounds} ──")
            
            # สุ่มลำดับจุด
            point_order = list(range(len(self.target_points)))
            random.shuffle(point_order)
            
            for point_num, point_idx in enumerate(point_order):
                target_x, target_y = self.target_points[point_idx]
                
                print(f"  จุดที่ {point_num + 1}/9: ({target_x}, {target_y})")
                
                # ── Phase 1: แสดงจุด + countdown ──
                phase1_start = time.time()
                while time.time() - phase1_start < CALIBRATION_DELAY_SEC:
                    frame = self.camera.read()
                    
                    # วาดหน้าจอ calibration
                    canvas = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
                    
                    # วาดจุดเป้าหมาย (วงกลมสีแดง + กากบาท)
                    cv2.circle(canvas, (target_x, target_y), 25, (0, 0, 255), -1)
                    cv2.circle(canvas, (target_x, target_y), 30, (0, 0, 200), 2)
                    cv2.line(canvas, (target_x - 35, target_y),
                            (target_x + 35, target_y), (255, 255, 255), 1)
                    cv2.line(canvas, (target_x, target_y - 35),
                            (target_x, target_y + 35), (255, 255, 255), 1)
                    
                    # ข้อความสถานะ
                    remaining = CALIBRATION_DELAY_SEC - (time.time() - phase1_start)
                    cv2.putText(canvas, f"Point your NOSE at the RED dot - hold still in {remaining:.1f}s",
                               (SCREEN_WIDTH // 2 - 380, SCREEN_HEIGHT - 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(canvas,
                               f"Round {round_idx+1}/{num_rounds}  |  "
                               f"Point {point_num+1}/9",
                               (SCREEN_WIDTH // 2 - 150, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)
                    
                    cv2.imshow("Calibration", canvas)
                    
                    if show_camera and frame is not None:
                        cv2.imshow("Camera", frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC
                        self._cleanup()
                        return None
                
                # ── Phase 2: บันทึกข้อมูล ──
                phase2_start = time.time()
                point_frames = 0
                
                while time.time() - phase2_start < CALIBRATION_DURATION_SEC:
                    frame = self.camera.read()
                    if frame is None:
                        continue
                    
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    landmarks = self.detector.detect(rgb)
                    
                    if landmarks is not None:
                        fw, fh = self.camera.get_frame_size()
                        cursor_features = self.extractor.extract_cursor_features(
                            landmarks, fw, fh
                        )
                        
                        if cursor_features is not None:
                            recorded_x = int(np.interp(target_x, [int(0.05 * SCREEN_WIDTH), int(0.95 * SCREEN_WIDTH)], [0, SCREEN_WIDTH]))
                            recorded_y = int(np.interp(target_y, [int(0.05 * SCREEN_HEIGHT), int(0.95 * SCREEN_HEIGHT)], [0, SCREEN_HEIGHT]))
                            row = list(cursor_features) + [recorded_x, recorded_y]
                            all_data.append(row)
                            point_frames += 1
                    
                    # วาดหน้าจอ + recording indicator
                    canvas = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
                    
                    # จุดเป้าหมาย (เขียว = กำลังบันทึก)
                    cv2.circle(canvas, (target_x, target_y), 25, (0, 255, 0), -1)
                    cv2.circle(canvas, (target_x, target_y), 30, (0, 200, 0), 2)
                    cv2.line(canvas, (target_x - 35, target_y),
                            (target_x + 35, target_y), (255, 255, 255), 1)
                    cv2.line(canvas, (target_x, target_y - 35),
                            (target_x, target_y + 35), (255, 255, 255), 1)
                    
                    cv2.putText(canvas, f"Recording Point {point_num+1}...",
                                (target_x - 100, target_y + 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow("Calibration", canvas)
                    
                    if show_camera and frame is not None:
                        cv2.imshow("Camera", frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:
                        self._cleanup()
                        return None
                
                print(f"    → บันทึกได้ {point_frames} เฟรม")
        
        self._cleanup()
        
        # สร้าง DataFrame
        columns = feature_names + ["screen_x", "screen_y"]
        df = pd.DataFrame(all_data, columns=columns)
        
        # บันทึก CSV (ต่อท้ายสะสมถ้ามีข้อมูลเดิม)
        os.makedirs(os.path.dirname(CALIBRATION_DATA_PATH), exist_ok=True)
        if os.path.exists(CALIBRATION_DATA_PATH):
            try:
                existing_df = pd.read_csv(CALIBRATION_DATA_PATH)
                df = pd.concat([existing_df, df], ignore_index=True)
                print(f"     📎 สะสมต่อท้ายข้อมูลเดิม (รวมทั้งหมด {len(df)} ตัวอย่าง)")
            except Exception:
                pass
        
        df.to_csv(CALIBRATION_DATA_PATH, index=False)
        
        print(f"\n  ✅ บันทึกข้อมูล Calibration เรียบร้อยแล้ว!")
        print(f"     ข้อมูลทั้งหมดในระบบ: {len(df)} ตัวอย่าง")
        print(f"     บันทึกไว้ที่: {CALIBRATION_DATA_PATH}")
        print(f"\n  สถิติ:")
        print(df.describe().to_string())
        print("\n  💡 คุณสามารถกดปุ่ม '🏋️ Train Cursor' บนหน้าจอหลักเพื่อเทรนโมเดลด้วยตัวเองได้เลยครับ")
        
        return df
    
    def _cleanup(self):
        """ปิด resources"""
        self.camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="9-Point Calibration")
    parser.add_argument("--rounds", type=int, default=2,
                       help="จำนวนรอบ calibration (default: 2)")
    parser.add_argument("--no-camera", action="store_true",
                       help="ไม่แสดง camera preview")
    args = parser.parse_args()
    
    cal = NinePointCalibration()
    result = cal.run(num_rounds=args.rounds, show_camera=not args.no_camera)
