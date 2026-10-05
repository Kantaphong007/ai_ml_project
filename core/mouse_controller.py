"""
EYE ORDER COME AI — Mouse Controller
ควบคุมเมาส์ผ่าน Win32 API โดยตรง (ctypes) และใช้ PyAutoGUI เฉพาะคีย์บอร์ด

ทำไมไม่ใช้ pyautogui.click() เหมือนเดิม:
  - pyautogui มี FAILSAFE: ถ้าเคอร์เซอร์อยู่มุมจอ (0,0) จะ raise exception ทุกคำสั่ง
    ซึ่งเคอร์เซอร์ที่คุมด้วยศีรษะไปถึงมุมจอได้บ่อย → คลิกที่ขอบ/มุมจอไม่ติด
    และระบบถูกปิดเองโดยไม่รู้ตัว
  - pyautogui มี PAUSE + ต้องอ่านตำแหน่งก่อนคลิก ทำให้ช้าและคลิกคลาดตำแหน่งได้
"""
import sys
import os
import time
import pyautogui

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    SCREEN_WIDTH, SCREEN_HEIGHT,
    CURSOR_SPEED_MULTIPLIER, SCROLL_SPEED
)

# ปิด failsafe ของ pyautogui (ใช้กับคีย์บอร์ดเท่านั้น) — การหยุดฉุกเฉินใช้ปุ่ม Pause/Break ใน Pipeline
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

import ctypes

_IS_WINDOWS = (sys.platform == 'win32')
if _IS_WINDOWS:
    _user32 = ctypes.windll.user32
    _LEFTDOWN, _LEFTUP = 0x0002, 0x0004
    _RIGHTDOWN, _RIGHTUP = 0x0008, 0x0010
    _WHEEL = 0x0800
    _WHEEL_DELTA = 120


class MouseController:
    """ควบคุมเมาส์ผ่าน Win32 API

    รองรับ:
      - เลื่อนเคอร์เซอร์ (พร้อม clamping ไม่ให้ออกนอกจอ)
      - คลิกซ้าย / คลิกขวา / ดับเบิลคลิก ที่พิกัดที่ระบุ (หรือที่ตำแหน่งปัจจุบัน)
      - Drag mode (mouse down/up)
      - Scroll up/down
      - พิมพ์ตัวอักษร (สำหรับ on-screen keyboard)
    """

    def __init__(self):
        self._drag_mode = False
        self._scroll_mode = False
        self._speed = CURSOR_SPEED_MULTIPLIER
        self._enabled = False
        self._last_pos = None

    def enable(self):
        """เปิดการควบคุม"""
        self._enabled = True

    def disable(self):
        """ปิดการควบคุม (ปล่อยเมาส์ที่ค้างอยู่ด้วย)"""
        if self._drag_mode:
            self.mouse_up()
        self._enabled = False

    def is_enabled(self):
        return self._enabled

    @staticmethod
    def _clamp(x, y):
        return (max(0, min(int(round(x)), SCREEN_WIDTH - 1)),
                max(0, min(int(round(y)), SCREEN_HEIGHT - 1)))

    def _set_pos(self, x, y):
        tx, ty = self._clamp(x, y)
        if self._last_pos == (tx, ty):
            return
        if _IS_WINDOWS:
            _user32.SetCursorPos(tx, ty)
        else:
            pyautogui.moveTo(tx, ty, duration=0)
        self._last_pos = (tx, ty)

    def move_to(self, x, y, duration=0):
        """เลื่อนเคอร์เซอร์ไปตำแหน่ง (x, y) พิกเซลบนหน้าจอ"""
        if not self._enabled:
            return
        self._set_pos(x, y)

    def _button(self, down_flag, up_flag, x=None, y=None, repeat=1):
        if x is not None and y is not None:
            self._set_pos(x, y)
        if _IS_WINDOWS:
            for _ in range(repeat):
                _user32.mouse_event(down_flag, 0, 0, 0, 0)
                time.sleep(0.012)
                _user32.mouse_event(up_flag, 0, 0, 0, 0)
                if repeat > 1:
                    time.sleep(0.04)
        else:
            button = 'left' if down_flag == 0x0002 else 'right'
            pyautogui.click(button=button, clicks=repeat)

    def left_click(self, x=None, y=None):
        """คลิกซ้ายที่ (x, y) — ถ้าไม่ระบุคลิกที่ตำแหน่งปัจจุบัน"""
        if not self._enabled:
            return
        if _IS_WINDOWS:
            self._button(_LEFTDOWN, _LEFTUP, x, y)
        else:
            self._button(0x0002, 0x0004, x, y)

    def right_click(self, x=None, y=None):
        """คลิกขวา"""
        if not self._enabled:
            return
        if _IS_WINDOWS:
            self._button(_RIGHTDOWN, _RIGHTUP, x, y)
        else:
            self._button(0x0008, 0x0010, x, y)

    def double_click(self, x=None, y=None):
        """ดับเบิลคลิก"""
        if not self._enabled:
            return
        if _IS_WINDOWS:
            self._button(_LEFTDOWN, _LEFTUP, x, y, repeat=2)
        else:
            self._button(0x0002, 0x0004, x, y, repeat=2)

    def mouse_down(self, x=None, y=None):
        """กดเมาส์ค้าง (เริ่ม Drag mode)"""
        if not self._enabled:
            return
        if x is not None and y is not None:
            self._set_pos(x, y)
        if _IS_WINDOWS:
            _user32.mouse_event(_LEFTDOWN, 0, 0, 0, 0)
        else:
            pyautogui.mouseDown()
        self._drag_mode = True

    def mouse_up(self):
        """ปล่อยเมาส์ (จบ Drag mode)"""
        if _IS_WINDOWS:
            _user32.mouse_event(_LEFTUP, 0, 0, 0, 0)
        else:
            pyautogui.mouseUp()
        self._drag_mode = False

    def scroll(self, clicks):
        """เลื่อนหน้าจอ

        Args:
            clicks: จำนวน notch (บวก = เลื่อนขึ้น, ลบ = เลื่อนลง)
        """
        if not self._enabled:
            return
        amount = int(clicks)
        if amount == 0:
            return
        if _IS_WINDOWS:
            _user32.mouse_event(_WHEEL, 0, 0, amount * _WHEEL_DELTA, 0)
        else:
            pyautogui.scroll(amount * SCROLL_SPEED)

    def type_key(self, key):
        """กดปุ่มคีย์บอร์ด (สำหรับ on-screen keyboard)"""
        if not self._enabled:
            return
        try:
            pyautogui.press(key)
        except Exception:
            pass

    def type_text(self, text):
        """พิมพ์ข้อความ"""
        if not self._enabled:
            return
        try:
            pyautogui.typewrite(text, interval=0.02)
        except Exception:
            pass

    def hotkey(self, *keys):
        """กดปุ่มลัด เช่น hotkey('ctrl', 'c')"""
        if not self._enabled:
            return
        try:
            pyautogui.hotkey(*keys)
        except Exception:
            pass

    @property
    def is_dragging(self):
        return self._drag_mode

    @property
    def is_scrolling(self):
        return self._scroll_mode

    @is_scrolling.setter
    def is_scrolling(self, value):
        self._scroll_mode = value

    def set_speed(self, multiplier):
        """ตั้งค่า speed multiplier"""
        self._speed = max(0.1, multiplier)

    def get_current_position(self):
        """ตำแหน่งเมาส์ปัจจุบัน"""
        return pyautogui.position()
