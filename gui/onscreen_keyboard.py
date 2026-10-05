"""
EYE ORDER COME AI — On-Screen Virtual Keyboard
คีย์บอร์ดเสมือนบนหน้าจอ ควบคุมด้วยสายตา (Dwell-click)
เปิด/ปิดได้จาก GUI หลัก
"""
import sys
import os
import time

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QFrame, QSizePolicy
)
from PyQt6.QtCore import (
    Qt, QTimer, QPoint, QPropertyAnimation, QEasingCurve, pyqtSignal
)
from PyQt6.QtGui import QFont, QColor, QPainter, QPen, QBrush

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    KEYBOARD_KEY_SIZE, KEYBOARD_OPACITY, KEYBOARD_DWELL_TIME_MS,
    KEYBOARD_FONT_SIZE
)


class DwellButton(QPushButton):
    """ปุ่มที่มี Dwell-click (ค้าง cursor = กด)"""
    
    dwellActivated = pyqtSignal(str)  # ส่ง key เมื่อ dwell สำเร็จ
    
    def __init__(self, key_label, key_action=None, parent=None, dwell_time_ms=KEYBOARD_DWELL_TIME_MS):
        super().__init__(key_label, parent)
        self.key_label = key_label
        self.key_action = key_action or key_label
        self.dwell_time_ms = dwell_time_ms
        
        self._dwell_progress = 0.0  # 0-1
        self._dwell_active = False
        self._dwell_timer = QTimer(self)
        self._dwell_timer.setInterval(16)  # ~60 FPS
        self._dwell_timer.timeout.connect(self._update_dwell)
        self._dwell_start_time = 0
        
        self.setMinimumSize(KEYBOARD_KEY_SIZE, KEYBOARD_KEY_SIZE)
        self.setFont(QFont("Segoe UI", KEYBOARD_FONT_SIZE, QFont.Weight.Bold))
        self._apply_style()
    
    def set_dwell_time(self, ms: int):
        self.dwell_time_ms = max(100, ms)
    
    def _apply_style(self):
        self.setStyleSheet("""
            QPushButton {
                background-color: rgba(45, 45, 60, 230);
                color: #E0E0E0;
                border: 1px solid rgba(100, 100, 140, 150);
                border-radius: 8px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: rgba(70, 70, 100, 240);
                border: 2px solid rgba(100, 180, 255, 200);
                color: #FFFFFF;
            }
        """)
    
    def start_dwell(self):
        """เริ่ม dwell timer"""
        if not self._dwell_active:
            self._dwell_active = True
            self._dwell_start_time = time.time()
            self._dwell_progress = 0.0
            self._dwell_timer.start()
    
    def stop_dwell(self):
        """หยุด dwell timer"""
        self._dwell_active = False
        self._dwell_progress = 0.0
        self._dwell_timer.stop()
        self.update()
    
    def _update_dwell(self):
        """อัพเดท dwell progress"""
        if not self._dwell_active:
            return
        
        elapsed = (time.time() - self._dwell_start_time) * 1000
        self._dwell_progress = min(elapsed / self.dwell_time_ms, 1.0)
        
        if self._dwell_progress >= 1.0:
            # Dwell สำเร็จ!
            self._dwell_timer.stop()
            self._dwell_active = False
            self._dwell_progress = 0.0
            self.dwellActivated.emit(self.key_action)
            self._flash_confirm()
        
        self.update()
    
    def _flash_confirm(self):
        """Flash สีเขียวเมื่อกดสำเร็จ"""
        self.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 200, 100, 240);
                color: white;
                border: 2px solid rgba(0, 255, 150, 255);
                border-radius: 8px;
                padding: 5px;
            }
        """)
        QTimer.singleShot(200, self._apply_style)
    
    def paintEvent(self, event):
        """วาด dwell progress ring"""
        super().paintEvent(event)
        
        if self._dwell_progress > 0:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            # วาด progress arc รอบปุ่ม
            pen = QPen(QColor(100, 180, 255, 200), 3)
            painter.setPen(pen)
            
            margin = 4
            rect = self.rect().adjusted(margin, margin, -margin, -margin)
            
            # วาด arc (span = progress * 360 degrees * 16 units)
            start_angle = 90 * 16  # เริ่มจากด้านบน
            span_angle = int(-self._dwell_progress * 360 * 16)
            painter.drawArc(rect, start_angle, span_angle)
            
            painter.end()


class OnScreenKeyboard(QWidget):
    """คีย์บอร์ดเสมือนบนหน้าจอ
    
    Features:
      - QWERTY layout (lower/upper/symbols)
      - Dwell-click (ค้าง cursor 800ms = กด)
      - Semi-transparent, always on top
      - Draggable
      - Text preview
    
    Signals:
      keyPressed(str): เมื่อกดปุ่ม (ส่ง key action string)
    """
    
    keyPressed = pyqtSignal(str)
    
    LAYOUTS = {
        'lower': [
            ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
            ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'],
            ['⇧', 'z', 'x', 'c', 'v', 'b', 'n', 'm', '⌫'],
            ['123', '🌐', ' ', '←', '→', '⏎'],
        ],
        'upper': [
            ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
            ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
            ['⇧', 'Z', 'X', 'C', 'V', 'B', 'N', 'M', '⌫'],
            ['123', '🌐', ' ', '←', '→', '⏎'],
        ],
        'symbols': [
            ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
            ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')'],
            ['ABC', '-', '=', '_', '+', '[', ']', '{', '}', '⌫'],
            ['123', '🌐', ' ', '.', ',', '?', '⏎'],
        ],
    }
    
    # Mapping จากปุ่มพิเศษ → PyAutoGUI key name
    SPECIAL_KEYS = {
        '⇧': 'SHIFT',
        '⌫': 'backspace',
        '⏎': 'enter',
        '←': 'left',
        '→': 'right',
        ' ': 'space',
        '🌐': 'LANG',
        '123': 'SYMBOLS',
        'ABC': 'ALPHA',
    }
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.current_layout = 'lower'
        self._shift_active = False
        self._typed_text = ""
        self._buttons = []
        self._current_hover_btn = None
        self.dwell_time_ms = KEYBOARD_DWELL_TIME_MS
        
        # Window properties
        self.setWindowTitle("EYE ORDER — Keyboard")
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool |
            Qt.WindowType.FramelessWindowHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowOpacity(KEYBOARD_OPACITY)
        
        # Dragging
        self._drag_pos = None
        
        self._build_ui()
        self._load_layout('lower')
    
    def set_dwell_time(self, ms: int):
        """ตั้งค่าเวลา Dwell-click (ms)"""
        self.dwell_time_ms = max(100, ms)
        for btn in self._buttons:
            if hasattr(btn, 'set_dwell_time'):
                btn.set_dwell_time(ms)
    
    def _build_ui(self):
        """สร้าง UI"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(5)
        
        # ── Title bar (draggable) ──
        title_bar = QFrame()
        title_bar.setStyleSheet("""
            QFrame {
                background-color: rgba(30, 30, 50, 220);
                border-radius: 8px;
                border: 1px solid rgba(80, 80, 120, 150);
            }
        """)
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(10, 5, 10, 5)
        
        title_label = QLabel("⌨️ EYE ORDER Keyboard")
        title_label.setFont(QFont("Segoe UI", 10))
        title_label.setStyleSheet("color: #AAAACC; border: none;")
        title_layout.addWidget(title_label)
        
        title_layout.addStretch()
        
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(25, 25)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 80, 80, 150);
                color: white;
                border: none;
                border-radius: 12px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: rgba(255, 50, 50, 220);
            }
        """)
        close_btn.clicked.connect(self.hide)
        title_layout.addWidget(close_btn)
        
        main_layout.addWidget(title_bar)
        
        # ── Text preview ──
        self._text_preview = QLabel("")
        self._text_preview.setFont(QFont("Consolas", 14))
        self._text_preview.setStyleSheet("""
            QLabel {
                background-color: rgba(20, 20, 35, 230);
                color: #FFFFFF;
                border: 1px solid rgba(80, 80, 120, 150);
                border-radius: 6px;
                padding: 8px 12px;
                min-height: 30px;
            }
        """)
        self._text_preview.setAlignment(Qt.AlignmentFlag.AlignLeft |
                                        Qt.AlignmentFlag.AlignVCenter)
        main_layout.addWidget(self._text_preview)
        
        # ── Keyboard grid ──
        self._keyboard_frame = QFrame()
        self._keyboard_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(25, 25, 40, 220);
                border-radius: 10px;
                border: 1px solid rgba(60, 60, 90, 150);
            }
        """)
        self._keyboard_layout = QVBoxLayout(self._keyboard_frame)
        self._keyboard_layout.setContentsMargins(8, 8, 8, 8)
        self._keyboard_layout.setSpacing(4)
        
        main_layout.addWidget(self._keyboard_frame)
    
    def _load_layout(self, layout_name):
        """โหลดและแสดง keyboard layout"""
        self.current_layout = layout_name
        
        # ลบ buttons เก่า
        for btn in self._buttons:
            btn.deleteLater()
        self._buttons.clear()
        
        # ลบ layout เก่า
        while self._keyboard_layout.count():
            item = self._keyboard_layout.takeAt(0)
            if item.layout():
                while item.layout().count():
                    child = item.layout().takeAt(0)
                    if child.widget():
                        child.widget().deleteLater()
        
        layout = self.LAYOUTS[layout_name]
        
        for row_keys in layout:
            row_layout = QHBoxLayout()
            row_layout.setSpacing(4)
            
            for key in row_keys:
                action = self.SPECIAL_KEYS.get(key, key)
                btn = DwellButton(key, action, dwell_time_ms=self.dwell_time_ms)
                
                # ปุ่มพิเศษกว้างกว่า
                if key == ' ':
                    btn.setMinimumWidth(KEYBOARD_KEY_SIZE * 3)
                    btn.setText("Space")
                elif key in ('⇧', '⌫', '⏎', '123', 'ABC'):
                    btn.setMinimumWidth(int(KEYBOARD_KEY_SIZE * 1.4))
                
                btn.dwellActivated.connect(self._on_key_activated)
                btn.clicked.connect(lambda checked, a=action: self._on_key_activated(a))
                
                self._buttons.append(btn)
                row_layout.addWidget(btn)
            
            self._keyboard_layout.addLayout(row_layout)
        
        self.adjustSize()
    
    def _on_key_activated(self, action):
        """จัดการเมื่อกดปุ่ม"""
        if action == 'SHIFT':
            self._shift_active = not self._shift_active
            if self._shift_active:
                self._load_layout('upper')
            else:
                self._load_layout('lower')
            return
        
        if action == 'SYMBOLS':
            self._load_layout('symbols')
            return
        
        if action == 'ALPHA':
            self._load_layout('lower')
            return
        
        if action == 'LANG':
            # สลับภาษา (TODO: implement Thai layout)
            return
        
        if action == 'backspace':
            self._typed_text = self._typed_text[:-1]
        elif action == 'enter':
            self._typed_text += '\n'
        elif action == 'space':
            self._typed_text += ' '
        elif action in ('left', 'right'):
            pass  # arrow keys ไม่เพิ่มใน preview
        else:
            self._typed_text += action
            # Auto-unshift หลังพิมพ์ตัวใหญ่
            if self._shift_active and self.current_layout == 'upper':
                self._shift_active = False
                self._load_layout('lower')
        
        # อัพเดท preview
        display_text = self._typed_text[-50:]  # แสดงแค่ 50 ตัวล่าสุด
        self._text_preview.setText(display_text + "│")
        
        # ส่ง signal
        self.keyPressed.emit(action)
    
    def handle_cursor_position(self, x, y):
        """ตรวจสอบว่า cursor อยู่บนปุ่มไหน แล้วจัดการ dwell
        
        Args:
            x, y: ตำแหน่ง cursor บนหน้าจอ (screen coords)
        """
        if not self.isVisible():
            return
        
        # แปลงเป็น widget coords
        local_pos = self.mapFromGlobal(QPoint(int(x), int(y)))
        
        # หาปุ่มที่ cursor อยู่บน
        hovered_btn = None
        for btn in self._buttons:
            btn_rect = btn.geometry()
            # ต้องคำนวณตำแหน่งจาก parent layouts
            global_btn_pos = btn.mapToGlobal(QPoint(0, 0))
            btn_screen_rect = btn.rect().translated(global_btn_pos)
            
            if btn_screen_rect.contains(QPoint(int(x), int(y))):
                hovered_btn = btn
                break
        
        if hovered_btn != self._current_hover_btn:
            # เปลี่ยนปุ่ม hover
            if self._current_hover_btn:
                self._current_hover_btn.stop_dwell()
            
            self._current_hover_btn = hovered_btn
            
            if hovered_btn:
                hovered_btn.start_dwell()
        
    def toggle_visibility(self):
        """เปิด/ปิด keyboard"""
        if self.isVisible():
            self.hide()
        else:
            self.show()
    
    def clear_text(self):
        """ล้างข้อความ preview"""
        self._typed_text = ""
        self._text_preview.setText("│")
    
    # ── Dragging ──
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.pos()
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        if self._drag_pos is not None:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        self._drag_pos = None
        super().mouseReleaseEvent(event)
