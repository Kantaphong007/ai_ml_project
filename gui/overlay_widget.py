"""
EYE ORDER COME AI — Overlay Widget
แสดง transparent overlay บนหน้าจอ แสดงสถานะระบบ
"""
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QColor, QPainter, QPen, QBrush

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import SCREEN_WIDTH, SCREEN_HEIGHT


class OverlayWidget(QWidget):
    """Transparent overlay แสดงสถานะบนหน้าจอ
    
    แสดง:
      - โหมดปัจจุบัน (Normal / Drag / Scroll)
      - ค่า EAR, ΔSmile
      - Action ล่าสุด
      - FPS
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("EYE ORDER — Overlay")
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.Tool |
            Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        
        # ตำแหน่ง: มุมบนขวา
        self.setGeometry(SCREEN_WIDTH - 280, 10, 260, 200)
        
        self._status = {}
        self._build_ui()
    
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 10, 15, 10)
        layout.setSpacing(3)
        
        font = QFont("Consolas", 10)
        
        self._mode_label = QLabel("Mode: Normal")
        self._mode_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self._mode_label.setStyleSheet("color: #00FF88;")
        layout.addWidget(self._mode_label)
        
        self._ear_label = QLabel("EAR L: 0.00  R: 0.00")
        self._ear_label.setFont(font)
        self._ear_label.setStyleSheet("color: #AACCFF;")
        layout.addWidget(self._ear_label)
        
        self._state_label = QLabel("State: idle")
        self._state_label.setFont(font)
        self._state_label.setStyleSheet("color: #CCCCFF;")
        layout.addWidget(self._state_label)

        self._smile_label = QLabel("ΔSmile: 0.00%")
        self._smile_label.setFont(font)
        self._smile_label.setStyleSheet("color: #FFCC88;")
        layout.addWidget(self._smile_label)
        
        self._action_label = QLabel("Action: —")
        self._action_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self._action_label.setStyleSheet("color: #FF8888;")
        layout.addWidget(self._action_label)
        
        self._fps_label = QLabel("FPS: 0")
        self._fps_label.setFont(font)
        self._fps_label.setStyleSheet("color: #888888;")
        layout.addWidget(self._fps_label)
        
        self._face_label = QLabel("Face: ❌")
        self._face_label.setFont(font)
        self._face_label.setStyleSheet("color: #FF4444;")
        layout.addWidget(self._face_label)
    
    def update_status(self, status):
        """อัพเดทสถานะ
        
        Args:
            status: dict จาก Pipeline.get_status()
        """
        self._status = status
        
        mode = status.get("mode", "normal")
        mode_colors = {
            "normal": "#00FF88",
            "drag": "#FF8800",
            "scroll": "#8888FF",
        }
        self._mode_label.setText(f"Mode: {mode.upper()}")
        self._mode_label.setStyleSheet(
            f"color: {mode_colors.get(mode, '#FFFFFF')};"
        )
        
        ear_l = status.get("ear_l", 0)
        ear_r = status.get("ear_r", 0)
        self._ear_label.setText(f"EAR L: {ear_l:.3f}  R: {ear_r:.3f}")
        
        self._state_label.setText(
            f"State: {status.get('gesture_state', '—')}  "
            f"({status.get('ratio_l', 0):.2f}/{status.get('ratio_r', 0):.2f})")

        ds = status.get("delta_smile", 0)
        self._smile_label.setText(f"ΔSmile: {ds*100:.1f}%")
        
        action = status.get("last_action", "—")
        if action:
            self._action_label.setText(f"Action: {action}")
        
        fps = status.get("fps", 0)
        self._fps_label.setText(f"FPS: {fps:.0f}")
        
        face = status.get("face_detected", False)
        if face:
            self._face_label.setText("Face: ✅")
            self._face_label.setStyleSheet("color: #00FF88;")
        else:
            self._face_label.setText("Face: ❌")
            self._face_label.setStyleSheet("color: #FF4444;")
    
    def paintEvent(self, event):
        """วาดพื้นหลัง semi-transparent"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # พื้นหลัง
        painter.setBrush(QBrush(QColor(20, 20, 35, 200)))
        painter.setPen(QPen(QColor(60, 60, 100, 150), 1))
        painter.drawRoundedRect(self.rect(), 12, 12)
        
        painter.end()
