"""
EYE ORDER COME AI — Camera Stream Manager
จัดการ OpenCV VideoCapture สำหรับอ่านเฟรมจากกล้องเว็บแคม
"""
import sys
import cv2
import threading
import time


class CameraStream:
    """จัดการสตรีมภาพจากกล้องเว็บแคม
    
    ใช้ thread แยกสำหรับอ่านเฟรม เพื่อไม่ให้ blocking main loop
    
    Example:
        >>> cam = CameraStream(camera_index=0)
        >>> cam.start()
        >>> frame = cam.read()
        >>> if frame is not None:
        ...     cv2.imshow("Preview", frame)
        >>> cam.stop()
    """
    
    def __init__(self, camera_index=0, width=640, height=480, fps=30, mirror=True):
        """
        Args:
            camera_index: index กล้อง (0 = default webcam)
            width: ความกว้างเฟรม
            height: ความสูงเฟรม
            fps: frame rate เป้าหมาย
            mirror: กลับภาพซ้าย-ขวา (Mirror mode)
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.mirror = mirror
        
        self._cap = None
        self._frame = None
        self._frame_id = 0          # เพิ่มทุกครั้งที่ได้เฟรมใหม่ (กันประมวลผลเฟรมเดิมซ้ำ)
        self._ret = False
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
    
    def set_mirror(self, mirror: bool):
        """เปิด/ปิด Mirror mode"""
        with self._lock:
            self.mirror = mirror
    
    def start(self):
        """เปิดกล้องและเริ่มอ่านเฟรมใน background thread
        
        Returns:
            self: สำหรับ method chaining
            
        Raises:
            RuntimeError: ถ้าเปิดกล้องไม่ได้
        """
        # DirectShow มีเฉพาะ Windows (เปิดกล้องเร็วกว่า) — macOS/Linux ใช้ backend อัตโนมัติ
        backend = cv2.CAP_DSHOW if sys.platform == 'win32' else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(self.camera_index, backend)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera index {self.camera_index}. "
                "Please check if a webcam is connected."
            )
        
        self._running = True
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()
        
        # รอให้เฟรมแรกพร้อม
        time.sleep(0.5)
        return self
    
    def _update(self):
        """Background thread: อ่านเฟรมจากกล้องต่อเนื่อง"""
        while self._running:
            ret, frame = self._cap.read()
            with self._lock:
                self._ret = ret
                if ret:
                    if self.mirror:
                        self._frame = cv2.flip(frame, 1)  # Mirror mode (เหมือนส่องกระจก)
                    else:
                        self._frame = frame  # Normal camera mode
                    self._frame_id += 1
            if not ret:
                time.sleep(0.005)
    
    def read(self):
        """อ่านเฟรมล่าสุด (thread-safe)
        
        Returns:
            np.ndarray หรือ None: เฟรมภาพ BGR, None ถ้าอ่านไม่สำเร็จ
        """
        with self._lock:
            if self._ret and self._frame is not None:
                return self._frame.copy()
        return None
    
    def read_with_id(self):
        """อ่านเฟรมล่าสุดพร้อมหมายเลขเฟรม (thread-safe)
        
        Returns:
            tuple: (frame_id, frame) — frame เป็น None ถ้ายังไม่มีเฟรม
        """
        with self._lock:
            if self._ret and self._frame is not None:
                return self._frame_id, self._frame.copy()
        return self._frame_id, None
    
    def read_rgb(self):
        """อ่านเฟรมล่าสุดในรูปแบบ RGB
        
        Returns:
            np.ndarray หรือ None: เฟรมภาพ RGB
        """
        frame = self.read()
        if frame is not None:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
    
    def is_opened(self):
        """ตรวจสอบว่ากล้องเปิดอยู่หรือไม่"""
        return self._cap is not None and self._cap.isOpened()
    
    def get_frame_size(self):
        """ขนาดเฟรมจริง (อาจต่างจากที่ตั้งค่า)
        
        Returns:
            tuple: (width, height)
        """
        if self._cap is not None:
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return w, h
        return self.width, self.height
    
    def stop(self):
        """หยุดอ่านเฟรมและปิดกล้อง"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
    
    def __del__(self):
        self.stop()
