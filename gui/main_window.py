"""
EYE ORDER COME AI — Main Window (PyQt6)
หน้าต่างหลักของโปรแกรม
"""
import sys
import os
import subprocess
import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QFrame, QGroupBox, QMessageBox,
    QStatusBar, QProgressBar, QApplication
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QFont, QImage, QPixmap, QIcon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pipeline import Pipeline
from gui.overlay_widget import OverlayWidget
from gui.onscreen_keyboard import OnScreenKeyboard
from gui.settings_dialog import SettingsDialog
from config.settings import (
    CURSOR_MODEL_PATH, CLICK_MODEL_PATH, load_user_settings
)


class MainWindow(QMainWindow):
    """หน้าต่างหลักของ EYE ORDER COME AI
    
    ประกอบด้วย:
      - Camera preview พร้อม landmarks
      - Status panel (EAR, ΔSmile, Head Pose, FPS)
      - Control buttons (Start, Calibrate, Train, Keyboard, Settings)
      - Model status indicators
    """

    # pipeline เรียก callback จาก thread ของมันเอง → ส่งผ่าน signal ให้ Qt อัปเดต widget ใน GUI thread
    _frame_ready = pyqtSignal(object)
    _status_ready = pyqtSignal(object)
    
    def __init__(self):
        super().__init__()
        self._frame_ready.connect(self._on_frame)
        self._status_ready.connect(self._on_status)
        
        self.pipeline = Pipeline()
        self.overlay = OverlayWidget()
        self.keyboard = OnScreenKeyboard()
        
        self._running = False
        
        self.setWindowTitle("EYE ORDER COME AI — ตาสั่งมา")
        self.setMinimumSize(900, 650)
        self.setStyleSheet(self._get_stylesheet())
        
        self._build_ui()
        self._update_model_status()
        self._apply_settings(load_user_settings())
        
        # Timer สำหรับอัพเดท UI
        self._ui_timer = QTimer(self)
        self._ui_timer.setInterval(33)  # ~30 FPS
        self._ui_timer.timeout.connect(self._update_ui)
        
        # Connect keyboard
        self.keyboard.keyPressed.connect(self._on_keyboard_key)
    
    def _get_stylesheet(self):
        return """
            QMainWindow {
                background-color: #0f0f1a;
            }
            QLabel {
                color: #d0d0e0;
            }
            QGroupBox {
                border: 1px solid #2a2a45;
                border-radius: 10px;
                margin-top: 12px;
                padding-top: 18px;
                font-weight: bold;
                color: #8888aa;
                background-color: #14142a;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px;
            }
            QPushButton {
                background-color: #1e1e3a;
                color: #d0d0e0;
                border: 1px solid #3a3a5a;
                border-radius: 8px;
                padding: 10px 18px;
                font-size: 13px;
                font-weight: bold;
                min-height: 20px;
            }
            QPushButton:hover {
                background-color: #2a2a50;
                border-color: #5588cc;
            }
            QPushButton:pressed {
                background-color: #3a3a60;
            }
            QPushButton:disabled {
                background-color: #181830;
                color: #555566;
                border-color: #252540;
            }
            QStatusBar {
                background-color: #0a0a15;
                color: #888899;
                border-top: 1px solid #222240;
            }
        """
    
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # ══════════════ LEFT: Camera Preview ══════════════
        left_layout = QVBoxLayout()
        
        # Title
        title = QLabel("👁️ EYE ORDER COME AI")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        title.setStyleSheet("color: #6688cc; margin-bottom: 5px;")
        left_layout.addWidget(title)
        
        subtitle = QLabel("ระบบควบคุมเคอร์เซอร์ด้วยปลายจมูกและการหันหัว (Nose & Head Motion)")
        subtitle.setFont(QFont("Segoe UI", 10))
        subtitle.setStyleSheet("color: #666688; margin-bottom: 10px;")
        left_layout.addWidget(subtitle)
        
        # Camera frame
        cam_group = QGroupBox("📷 Camera Preview")
        cam_layout = QVBoxLayout(cam_group)
        
        self._camera_label = QLabel()
        self._camera_label.setFixedSize(480, 360)
        self._camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._camera_label.setStyleSheet("""
            QLabel {
                background-color: #0a0a15;
                border: 2px solid #2a2a45;
                border-radius: 8px;
            }
        """)
        self._camera_label.setText("📷 Camera Off")
        self._camera_label.setFont(QFont("Segoe UI", 14))
        cam_layout.addWidget(self._camera_label)
        
        left_layout.addWidget(cam_group)
        main_layout.addLayout(left_layout, stretch=3)
        
        # ══════════════ RIGHT: Controls & Status ══════════════
        right_layout = QVBoxLayout()
        right_layout.setSpacing(10)
        
        # ── Model Status ──
        model_group = QGroupBox("🤖 Model Status")
        model_layout = QVBoxLayout(model_group)
        
        self._cursor_model_label = QLabel("Cursor Model: ❌ Not loaded")
        self._cursor_model_label.setFont(QFont("Segoe UI", 10))
        model_layout.addWidget(self._cursor_model_label)
        
        self._click_model_label = QLabel("Click: —")
        self._click_model_label.setFont(QFont("Segoe UI", 10))
        model_layout.addWidget(self._click_model_label)
        
        right_layout.addWidget(model_group)
        
        # ── Status Panel ──
        status_group = QGroupBox("📊 Live Status")
        status_layout = QGridLayout(status_group)
        
        font_mono = QFont("Consolas", 11)
        
        labels = [
            ("FPS:", "_fps_val"), ("Face:", "_face_val"),
            ("EAR L:", "_ear_l_val"), ("EAR R:", "_ear_r_val"),
            ("ΔSmile:", "_smile_val"), ("Mode:", "_mode_val"),
            ("Action:", "_action_val"), ("State:", "_state_val"),
            ("Ratio:", "_ratio_val"), ("Scroll:", "_scroll_val"),
        ]
        
        for i, (text, attr) in enumerate(labels):
            row, col = divmod(i, 2)
            lbl = QLabel(text)
            lbl.setFont(font_mono)
            lbl.setStyleSheet("color: #888899;")
            status_layout.addWidget(lbl, row, col * 2)
            
            val = QLabel("—")
            val.setFont(font_mono)
            val.setStyleSheet("color: #aabbdd;")
            setattr(self, attr, val)
            status_layout.addWidget(val, row, col * 2 + 1)
        
        right_layout.addWidget(status_group)
        
        # ── Control Buttons ──
        ctrl_group = QGroupBox("🎮 Controls")
        ctrl_layout = QVBoxLayout(ctrl_group)
        ctrl_layout.setSpacing(6)
        
        self._start_btn = QPushButton("▶  Start Eye Control")
        self._start_btn.setStyleSheet("""
            QPushButton {
                background-color: #1a3a2a;
                border-color: #2a6a4a;
                color: #88ddaa;
                font-size: 14px;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #2a5a3a;
            }
        """)
        self._start_btn.clicked.connect(self._toggle_pipeline)
        ctrl_layout.addWidget(self._start_btn)
        
        # Calibration buttons
        cal_layout = QHBoxLayout()
        
        self._baseline_btn = QPushButton("📐 Baseline")
        self._baseline_btn.clicked.connect(self._run_baseline)
        cal_layout.addWidget(self._baseline_btn)
        
        self._calibrate_btn = QPushButton("🎯 Calibrate")
        self._calibrate_btn.clicked.connect(self._run_calibration)
        cal_layout.addWidget(self._calibrate_btn)
        
        ctrl_layout.addLayout(cal_layout)
        
        # Data collection
        self._gesture_btn = QPushButton("✋ Collect Gestures")
        self._gesture_btn.clicked.connect(self._run_gesture_collection)
        ctrl_layout.addWidget(self._gesture_btn)

        # จูนการตรวจจับท่าทาง/scroll ด้วยข้อมูลจริง
        tune_layout = QHBoxLayout()
        self._record_signals_btn = QPushButton("🎙 Record Signals")
        self._record_signals_btn.setToolTip("บันทึกสัญญาณตา/ยิ้ม/ก้มเงย พร้อม label สำหรับจูนค่า")
        self._record_signals_btn.clicked.connect(self._run_signal_recorder)
        tune_layout.addWidget(self._record_signals_btn)
        self._tune_btn = QPushButton("🔧 Tune Gestures")
        self._tune_btn.setToolTip("ค้นหาค่าที่ดีที่สุดจากข้อมูลที่บันทึก → data/gesture_tuning.json")
        self._tune_btn.clicked.connect(self._run_tuner)
        tune_layout.addWidget(self._tune_btn)
        ctrl_layout.addLayout(tune_layout)
        
        # Training buttons
        train_layout = QHBoxLayout()
        
        self._train_cursor_btn = QPushButton("🏋️ Train Cursor")
        self._train_cursor_btn.clicked.connect(self._train_cursor)
        train_layout.addWidget(self._train_cursor_btn)
        
        self._train_click_btn = QPushButton("🏋️ Train Click")
        self._train_click_btn.clicked.connect(self._train_click)
        train_layout.addWidget(self._train_click_btn)
        
        ctrl_layout.addLayout(train_layout)
        
        # Evaluate
        self._eval_btn = QPushButton("📊 Evaluate Models")
        self._eval_btn.clicked.connect(self._evaluate_models)
        ctrl_layout.addWidget(self._eval_btn)
        
        # Keyboard toggle
        self._keyboard_btn = QPushButton("⌨️  On-Screen Keyboard")
        self._keyboard_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a2a4a;
                border-color: #4a4a7a;
                color: #aabbff;
            }
            QPushButton:hover {
                background-color: #3a3a6a;
            }
        """)
        self._keyboard_btn.clicked.connect(self._toggle_keyboard)
        ctrl_layout.addWidget(self._keyboard_btn)
        
        # Settings
        self._settings_btn = QPushButton("⚙️  Settings")
        self._settings_btn.clicked.connect(self._open_settings)
        ctrl_layout.addWidget(self._settings_btn)
        
        right_layout.addWidget(ctrl_group)
        right_layout.addStretch()
        
        main_layout.addLayout(right_layout, stretch=2)
        
        # ── Status Bar ──
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)
        self._statusbar.showMessage("Ready — กรุณา Calibrate และ Train Models ก่อนใช้งาน")
    
    def _update_model_status(self):
        """อัพเดทสถานะโมเดล"""
        if os.path.exists(CURSOR_MODEL_PATH):
            self._cursor_model_label.setText("Cursor Model: ✅ Loaded")
            self._cursor_model_label.setStyleSheet("color: #88ddaa;")
        else:
            self._cursor_model_label.setText("Cursor Model: ❌ Not trained")
            self._cursor_model_label.setStyleSheet("color: #ff8888;")
        
        # คลิกใช้ตัวตรวจจับท่าทางแบบ state machine (core/gesture_detector.py) ไม่ต้องเทรนโมเดล
        self._click_model_label.setText("Click: ✅ Gesture detector (ขยิบตา/ยิ้ม)")
        self._click_model_label.setStyleSheet("color: #88ddaa;")
    
    def _toggle_pipeline(self):
        """เริ่ม/หยุด pipeline"""
        if self._running:
            self.pipeline.stop()
            self._running = False
            self._start_btn.setText("▶  Start Eye Control")
            self._start_btn.setStyleSheet("""
                QPushButton {
                    background-color: #1a3a2a;
                    border-color: #2a6a4a;
                    color: #88ddaa;
                    font-size: 14px;
                    min-height: 30px;
                }
            """)
            self._ui_timer.stop()
            self.overlay.hide()
            self._statusbar.showMessage("Stopped")
            self._camera_label.setText("📷 Camera Off")
            self._enable_buttons(True)
        else:
            if not self.pipeline.initialize():
                QMessageBox.warning(
                    self, "Warning",
                    "ไม่สามารถโหลดโมเดลได้\n"
                    "กรุณา Calibrate + Train ก่อนใช้งาน"
                )
                return
            
            self.pipeline.on_frame_update = self._frame_ready.emit
            self.pipeline.on_status_update = self._status_ready.emit
            self.pipeline.start()
            self._running = True
            self._start_btn.setText("⏸  Stop")
            self._start_btn.setStyleSheet("""
                QPushButton {
                    background-color: #3a1a1a;
                    border-color: #6a2a2a;
                    color: #ff8888;
                    font-size: 14px;
                    min-height: 30px;
                }
            """)
            self._ui_timer.start()
            self.overlay.show()
            self._statusbar.showMessage("Running — กดปุ่ม Pause/Break เพื่อหยุด/เริ่มชั่วคราว")
            self._enable_buttons(False)
    
    def _enable_buttons(self, enabled):
        """เปิด/ปิดปุ่มขณะ pipeline ทำงาน"""
        for btn in [self._baseline_btn, self._calibrate_btn,
                    self._gesture_btn, self._train_cursor_btn,
                    self._train_click_btn, self._eval_btn,
                    self._record_signals_btn, self._tune_btn]:
            btn.setEnabled(enabled)
    
    def _on_frame(self, frame):
        """Callback: อัพเดทภาพจากกล้อง"""
        if frame is None:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            480, 360, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self._camera_label.setPixmap(pixmap)
    
    def _on_status(self, status):
        """Callback: อัพเดทสถานะ"""
        self.overlay.update_status(status)
        
        # อัพเดท keyboard dwell
        if self.keyboard.isVisible() and self._running:
            self.keyboard.handle_cursor_position(
                status.get("cursor_x", 0),
                status.get("cursor_y", 0)
            )
    
    def _update_ui(self):
        """อัพเดท status panel"""
        if not self._running:
            return
        
        status = self.pipeline.get_status()
        
        self._fps_val.setText(f"{status.get('fps', 0):.0f}")
        
        face = status.get("face_detected", False)
        self._face_val.setText("✅" if face else "❌")
        self._face_val.setStyleSheet(
            "color: #88ddaa;" if face else "color: #ff8888;"
        )
        
        self._ear_l_val.setText(f"{status.get('ear_l', 0):.3f}")
        self._ear_r_val.setText(f"{status.get('ear_r', 0):.3f}")
        self._smile_val.setText(f"{status.get('delta_smile', 0)*100:.1f}%")
        
        mode = status.get("mode", "normal")
        self._mode_val.setText(mode.upper())
        mode_colors = {"normal": "#88ddaa", "drag": "#ff8800", "scroll": "#8888ff"}
        self._mode_val.setStyleSheet(f"color: {mode_colors.get(mode, '#aabbdd')};")
        
        self._action_val.setText(status.get("last_action", "—") or "—")
        self._state_val.setText(status.get("gesture_state", "—"))
        self._ratio_val.setText(f"{status.get('ratio_l', 0):.2f} / {status.get('ratio_r', 0):.2f}")
        self._scroll_val.setText(f"{status.get('scroll_offset', 0):+.3f}" if mode == "scroll" else "—")
    
    def _run_baseline(self):
        """รัน baseline recording"""
        self._statusbar.showMessage("Running baseline recording...")
        self._run_script("calibration/baseline_recorder.py")
    
    def _run_calibration(self):
        """รัน 9-point calibration"""
        self._statusbar.showMessage("Running 9-point calibration...")
        self._run_script("calibration/nine_point_calibration.py")
    
    def _run_gesture_collection(self):
        """รัน gesture collection"""
        self._statusbar.showMessage("Running gesture collection...")
        self._run_script("calibration/gesture_collection.py")
    
    def _run_signal_recorder(self):
        """บันทึกสัญญาณสำหรับจูนการตรวจจับท่าทาง"""
        self._statusbar.showMessage("Recording gesture signals...")
        self._run_script("calibration/signal_recorder.py")

    def _run_tuner(self):
        """จูนพารามิเตอร์จากข้อมูลที่บันทึก (ผลใช้ตอนกด Start ครั้งถัดไป)"""
        self._statusbar.showMessage("Tuning gestures — ผลจะใช้เมื่อกด Start ครั้งถัดไป")
        self._run_script("training/tune_gestures.py")

    def _train_cursor(self):
        """เทรน cursor model"""
        self._statusbar.showMessage("Training cursor model...")
        self._run_script("training/train_cursor_model.py")
        self._update_model_status()
    
    def _train_click(self):
        """เทรน click model"""
        self._statusbar.showMessage("Training click model...")
        self._run_script("training/train_click_model.py")
        self._update_model_status()
    
    def _evaluate_models(self):
        """ประเมินโมเดล"""
        self._statusbar.showMessage("Evaluating models...")
        self._run_script("training/evaluate.py")
    
    def _run_script(self, script_path):
        """รัน Python script แยก process"""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_path = os.path.join(project_root, script_path)
        
        try:
            # Windows: เปิดหน้าต่าง console ใหม่ / macOS, Linux: log ออก terminal ที่เปิดโปรแกรม
            flags = subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
            subprocess.Popen([sys.executable, full_path], cwd=project_root, creationflags=flags)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"ไม่สามารถรัน script:\n{e}")
    
    def _toggle_keyboard(self):
        """เปิด/ปิด on-screen keyboard"""
        self.keyboard.toggle_visibility()
    
    def _on_keyboard_key(self, key):
        """จัดการเมื่อกดปุ่มบน on-screen keyboard"""
        if self._running and self.pipeline.mouse_controller.is_enabled():
            mc = self.pipeline.mouse_controller
            if key == 'backspace':
                mc.type_key('backspace')
            elif key == 'enter':
                mc.type_key('enter')
            elif key == 'space':
                mc.type_key('space')
            elif key == 'left':
                mc.type_key('left')
            elif key == 'right':
                mc.type_key('right')
            elif key in ('SHIFT', 'SYMBOLS', 'ALPHA', 'LANG'):
                pass  # จัดการใน keyboard widget แล้ว
            else:
                mc.type_key(key)
    
    def _apply_settings(self, settings):
        """นำค่าตั้งค่าไปปรับใช้กับระบบทันที"""
        self.pipeline.cursor_predictor.set_smoothing_frames(
            settings.get("smoothing", 5)
        )
        self.pipeline.cursor_predictor.set_speed(
            settings.get("speed", 1.0)
        )
        self.pipeline.cursor_predictor.set_inversion(
            settings.get("invert_x", False),
            settings.get("invert_y", False)
        )
        self.pipeline.mouse_controller.set_speed(settings.get("speed", 1.0))
        self.pipeline.camera.set_mirror(settings.get("mirror", True))
        
        self.keyboard.set_dwell_time(settings.get("dwell_time", 800))
        self.keyboard.setWindowOpacity(settings.get("keyboard_opacity", 85) / 100.0)
        
        if settings.get("show_overlay", True):
            if self._running:
                self.overlay.show()
        else:
            self.overlay.hide()
            
    def _open_settings(self):
        """เปิดเมนูตั้งค่า"""
        dialog = SettingsDialog(self)
        if dialog.exec():
            settings = dialog.get_settings()
            self._apply_settings(settings)
    
    def closeEvent(self, event):
        """ปิดโปรแกรม"""
        if self._running:
            self.pipeline.stop()
        self.overlay.close()
        self.keyboard.close()
        event.accept()
