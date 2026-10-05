"""
EYE ORDER COME AI (ตาสั่งมา)
ระบบควบคุมเคอร์เซอร์เมาส์และสั่งการด้วยสายตาผ่านกล้องเว็บแคม

Entry Point — เปิด GUI หลัก

Usage:
    python main.py          # เปิด GUI หลัก
    python main.py --cli    # โหมด CLI (ไม่มี GUI)
"""
import os
import warnings

# ปิดข้อความเตือนที่ไม่เกี่ยวกับการทำงาน (protobuf deprecation / log ของ MediaPipe-TFLite)
warnings.filterwarnings("ignore", message=".*GetPrototype.*")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import os
import argparse

# ตั้ง project root ให้ import ได้
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def run_gui():
    """เปิดโปรแกรมแบบ GUI"""
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtGui import QFont
    from gui.main_window import MainWindow
    
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    app.setStyle("Fusion")
    
    # Dark palette
    from PyQt6.QtGui import QPalette, QColor
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(15, 15, 26))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(208, 208, 224))
    palette.setColor(QPalette.ColorRole.Base, QColor(20, 20, 42))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(30, 30, 55))
    palette.setColor(QPalette.ColorRole.Text, QColor(208, 208, 224))
    palette.setColor(QPalette.ColorRole.Button, QColor(30, 30, 58))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(208, 208, 224))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(80, 120, 200))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


def run_cli():
    """โหมด CLI — เมนูเลือกฟังก์ชัน"""
    print("\n" + "╔" + "═" * 50 + "╗")
    print("║    EYE ORDER COME AI — ตาสั่งมา               ║")
    print("║    CLI Mode                                    ║")
    print("╚" + "═" * 50 + "╝")
    
    while True:
        print("\n  เลือกฟังก์ชัน:")
        print("  ─────────────────────────────────")
        print("  1. 📐 บันทึกค่าฐาน (Baseline)")
        print("  2. 🎯 9-Point Calibration")
        print("  3. ✋ เก็บข้อมูลท่าทาง (Gesture Collection)")
        print("  4. 🏋️  เทรนโมเดล Cursor (Regression)")
        print("  5. 🏋️  เทรนโมเดล Click (Classification)")
        print("  6. 📊 ประเมินผลโมเดล")
        print("  7. ▶  เริ่มควบคุมเมาส์ด้วยสายตา")
        print("  0. ❌ ออก")
        print("  ─────────────────────────────────")
        
        choice = input("  เลือก [0-7]: ").strip()
        
        if choice == "0":
            print("\n  👋 ออกจากโปรแกรม")
            break
        elif choice == "1":
            from calibration.baseline_recorder import BaselineRecorder
            recorder = BaselineRecorder()
            recorder.record()
        elif choice == "2":
            from calibration.nine_point_calibration import NinePointCalibration
            cal = NinePointCalibration()
            cal.run()
        elif choice == "3":
            from calibration.gesture_collection import GestureCollector
            collector = GestureCollector()
            collector.run()
        elif choice == "4":
            from training.train_cursor_model import main as train_cursor
            train_cursor()
        elif choice == "5":
            from training.train_click_model import main as train_click
            train_click()
        elif choice == "6":
            from training.evaluate import generate_full_report
            generate_full_report()
        elif choice == "7":
            print("\n  🔄 กำลังเตรียมระบบ...")
            from core.pipeline import Pipeline
            import cv2
            
            pipe = Pipeline()
            if pipe.initialize():
                pipe.start()
                print("  ▶ ระบบเริ่มทำงาน!")
                print("  กด Ctrl+C เพื่อหยุด")
                try:
                    while pipe.is_running:
                        status = pipe.get_status()
                        sys.stdout.write(
                            f"\r  FPS: {status['fps']:.0f} | "
                            f"Face: {'✅' if status['face_detected'] else '❌'} | "
                            f"Mode: {status['mode']} | "
                            f"Action: {status['last_action'] or '—'}"
                            f"          "
                        )
                        sys.stdout.flush()
                        import time
                        time.sleep(0.1)
                except KeyboardInterrupt:
                    pass
                finally:
                    pipe.stop()
            else:
                print("  ❌ ไม่สามารถเริ่มระบบได้")
        else:
            print("  ⚠️ กรุณาเลือก 0-7")


def main():
    parser = argparse.ArgumentParser(
        description="EYE ORDER COME AI — ระบบควบคุมเคอร์เซอร์ด้วยสายตา"
    )
    parser.add_argument("--cli", action="store_true",
                       help="โหมด CLI (ไม่มี GUI)")
    args = parser.parse_args()
    
    if args.cli:
        run_cli()
    else:
        run_gui()


if __name__ == "__main__":
    main()
