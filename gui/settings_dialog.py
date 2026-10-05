"""
EYE ORDER COME AI — Settings Dialog
เมนูตั้งค่า
"""
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QSlider, QSpinBox, QCheckBox, QPushButton,
    QGroupBox, QComboBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    load_user_settings, save_user_settings
)


class SettingsDialog(QDialog):
    """เมนูตั้งค่าระบบ"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("⚙️ Settings — EYE ORDER COME AI")
        self.setFixedSize(450, 500)
        self.setStyleSheet("""
            QDialog {
                background-color: #1a1a2e;
                color: #e0e0e0;
            }
            QGroupBox {
                border: 1px solid #333355;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
                font-weight: bold;
                color: #aaaacc;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 5px;
            }
            QLabel {
                color: #ccccdd;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #333355;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #6688cc;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSpinBox, QComboBox {
                background-color: #262640;
                color: #e0e0e0;
                border: 1px solid #444466;
                border-radius: 4px;
                padding: 4px;
            }
            QPushButton {
                background-color: #334466;
                color: white;
                border: 1px solid #445577;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #445588;
            }
        """)
        
        self._settings = load_user_settings()
        self._build_ui()
    
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # ── Mouse Settings ──
        mouse_group = QGroupBox("🖱️ Mouse Control")
        mouse_layout = QFormLayout(mouse_group)
        
        self._smoothing_slider = QSlider(Qt.Orientation.Horizontal)
        self._smoothing_slider.setRange(1, 15)
        self._smoothing_slider.setValue(self._settings["smoothing"])
        self._smoothing_label = QLabel(str(self._settings["smoothing"]))
        self._smoothing_slider.valueChanged.connect(
            lambda v: self._smoothing_label.setText(str(v))
        )
        h = QHBoxLayout()
        h.addWidget(self._smoothing_slider)
        h.addWidget(self._smoothing_label)
        mouse_layout.addRow("Smoothing Frames:", h)
        
        self._speed_slider = QSlider(Qt.Orientation.Horizontal)
        self._speed_slider.setRange(1, 30)
        self._speed_slider.setValue(int(self._settings["speed"] * 10))
        self._speed_label = QLabel(f"{self._settings['speed']:.1f}")
        self._speed_slider.valueChanged.connect(
            lambda v: self._speed_label.setText(f"{v/10:.1f}")
        )
        h2 = QHBoxLayout()
        h2.addWidget(self._speed_slider)
        h2.addWidget(self._speed_label)
        mouse_layout.addRow("Cursor Speed:", h2)
        
        self._invert_x_check = QCheckBox("Invert X (กลับทิศเมาส์ซ้าย-ขวา)")
        self._invert_x_check.setChecked(self._settings["invert_x"])
        self._invert_x_check.setStyleSheet("color: #ccccdd;")
        mouse_layout.addRow(self._invert_x_check)
        
        self._invert_y_check = QCheckBox("Invert Y (กลับทิศเมาส์บน-ล่าง)")
        self._invert_y_check.setChecked(self._settings["invert_y"])
        self._invert_y_check.setStyleSheet("color: #ccccdd;")
        mouse_layout.addRow(self._invert_y_check)
        
        layout.addWidget(mouse_group)
        
        # ── Camera Settings ──
        cam_group = QGroupBox("📷 Camera")
        cam_layout = QFormLayout(cam_group)
        
        self._camera_spin = QSpinBox()
        self._camera_spin.setRange(0, 5)
        self._camera_spin.setValue(self._settings["camera"])
        cam_layout.addRow("Camera Index:", self._camera_spin)
        
        self._mirror_check = QCheckBox("Mirror Mode (กลับภาพซ้าย-ขวา)")
        self._mirror_check.setChecked(self._settings.get("mirror", True))
        self._mirror_check.setStyleSheet("color: #ccccdd;")
        cam_layout.addRow(self._mirror_check)
        
        layout.addWidget(cam_group)
        
        # ── Keyboard Settings ──
        kb_group = QGroupBox("⌨️ On-Screen Keyboard")
        kb_layout = QFormLayout(kb_group)
        
        self._dwell_spin = QSpinBox()
        self._dwell_spin.setRange(300, 2000)
        self._dwell_spin.setSingleStep(100)
        self._dwell_spin.setValue(self._settings["dwell_time"])
        self._dwell_spin.setSuffix(" ms")
        kb_layout.addRow("Dwell Time:", self._dwell_spin)
        
        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(30, 100)
        self._opacity_slider.setValue(self._settings["keyboard_opacity"])
        self._opacity_label = QLabel(f"{self._settings['keyboard_opacity']}%")
        self._opacity_slider.valueChanged.connect(
            lambda v: self._opacity_label.setText(f"{v}%")
        )
        h3 = QHBoxLayout()
        h3.addWidget(self._opacity_slider)
        h3.addWidget(self._opacity_label)
        kb_layout.addRow("Opacity:", h3)
        
        layout.addWidget(kb_group)
        
        # ── Overlay ──
        overlay_group = QGroupBox("📊 Overlay")
        overlay_layout = QFormLayout(overlay_group)
        
        self._overlay_check = QCheckBox("Show status overlay")
        self._overlay_check.setChecked(self._settings["show_overlay"])
        self._overlay_check.setStyleSheet("color: #ccccdd;")
        overlay_layout.addRow(self._overlay_check)
        
        layout.addWidget(overlay_group)
        
        # ── Buttons ──
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        save_btn = QPushButton("💾 Save")
        save_btn.clicked.connect(self.accept)
        btn_layout.addWidget(save_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #443333;
                border: 1px solid #664444;
            }
            QPushButton:hover {
                background-color: #554444;
            }
        """)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        layout.addLayout(btn_layout)
    
    def accept(self):
        """เมื่อกดปุ่ม Save: เซฟลงไฟล์ JSON แล้วปิด Dialog"""
        settings = self.get_settings()
        save_user_settings(settings)
        super().accept()
    
    def get_settings(self):
        """ดึงค่าตั้งค่าปัจจุบัน"""
        return {
            "smoothing": self._smoothing_slider.value(),
            "speed": self._speed_slider.value() / 10.0,
            "camera": self._camera_spin.value(),
            "mirror": self._mirror_check.isChecked(),
            "invert_x": self._invert_x_check.isChecked(),
            "invert_y": self._invert_y_check.isChecked(),
            "dwell_time": self._dwell_spin.value(),
            "keyboard_opacity": self._opacity_slider.value(),
            "show_overlay": self._overlay_check.isChecked(),
        }
