"""
RTSP Handler - Video Stream Management
รองรับ RTSP stream จากกล้อง CCTV พร้อมการจัดการ buffer และ error handling
"""

import cv2
import threading
import time
import queue
from typing import Optional
import numpy as np

class RTSPHandler:
    def __init__(self, rtsp_url: str, buffer_size: int = 1):
        self.rtsp_url = rtsp_url
        self.buffer_size = buffer_size
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.running = False
        self.capture_thread = None
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        
        # Statistics
        self.total_frames = 0
        self.dropped_frames = 0
        self.last_frame_time = 0
        
        import re
        masked_url = re.sub(r"(://[^:/@]+:)[^/]*@", r"\1***@", rtsp_url or "")
        print(f"[INFO] Initializing RTSP connection: {masked_url}")
        self.connect()
        
    def connect(self):
        """Connect to RTSP stream"""
        try:
            # Release existing connection
            if self.cap is not None:
                self.cap.release()
                
            # Create new connection with basic parameters
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            
            # Set buffer size to reduce latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Set connection timeout
            self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
            
            # Test connection with retries
            connection_attempts = 3
            for attempt in range(connection_attempts):
                if self.cap.isOpened():
                    # Try to read a test frame
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        print(f"[OK] RTSP connected successfully")
                        print(f"[INFO] Resolution: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
                        print(f"[INFO] FPS: {self.cap.get(cv2.CAP_PROP_FPS)}")
                        
                        self.reconnect_attempts = 0
                        self.start_capture_thread()
                        return
                
                print(f"[WARNING] Connection attempt {attempt + 1}/{connection_attempts} failed, retrying...")
                time.sleep(1)
                
            raise ConnectionError(f"Cannot connect to RTSP stream after {connection_attempts} attempts")
            
        except Exception as e:
            print(f"[ERROR] RTSP connection failed: {e}")
            self.handle_reconnection()
            
    def start_capture_thread(self):
        """Start background thread for frame capture"""
        if self.capture_thread is not None and self.capture_thread.is_alive():
            self.running = False
            self.capture_thread.join(timeout=2)
            
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        print("[INFO] Frame capture thread started")
        
    def _capture_loop(self):
        """Background loop for capturing frames"""
        consecutive_failures = 0
        max_consecutive_failures = 10
        
        while self.running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    time.sleep(0.1)
                    consecutive_failures += 1
                    if consecutive_failures > max_consecutive_failures:
                        print("[WARNING] Too many consecutive failures, attempting reconnection...")
                        self.handle_reconnection()
                        consecutive_failures = 0
                    continue
                    
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures > max_consecutive_failures:
                        print("[WARNING] Failed to read frames consistently, attempting reconnection...")
                        self.handle_reconnection()
                        consecutive_failures = 0
                    time.sleep(0.01)  # Small delay before retry
                    continue
                
                # Reset failure counter on successful read
                consecutive_failures = 0
                self.total_frames += 1
                self.last_frame_time = time.time()
                
                # Add frame to queue (drop old frames if queue is full)
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    # Remove old frame and add new one
                    try:
                        self.frame_queue.get_nowait()
                        self.dropped_frames += 1
                    except queue.Empty:
                        pass
                    try:
                        self.frame_queue.put_nowait(frame)
                    except queue.Full:
                        pass  # Skip this frame
                        
            except Exception as e:
                print(f"[ERROR] Error in capture loop: {e}")
                consecutive_failures += 1
                time.sleep(0.1)
                if consecutive_failures > max_consecutive_failures:
                    self.handle_reconnection()
                    consecutive_failures = 0
                
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame from the stream"""
        try:
            # Get the most recent frame
            frame = None
            while not self.frame_queue.empty():
                frame = self.frame_queue.get_nowait()
            return frame
        except queue.Empty:
            return None
        except Exception as e:
            print(f"[ERROR] Error getting frame: {e}")
            return None
            
    def handle_reconnection(self):
        """Handle reconnection logic"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            print(f"[ERROR] Max reconnection attempts reached ({self.max_reconnect_attempts})")
            return
            
        self.reconnect_attempts += 1
        print(f"[INFO] Reconnecting... (attempt {self.reconnect_attempts}/{self.max_reconnect_attempts})")
        
        # Stop current capture
        self.running = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2)
            
        # Wait before reconnecting
        time.sleep(2)
        
        # Attempt reconnection
        self.connect()
        
    def is_connected(self) -> bool:
        """Check if stream is connected and receiving frames"""
        if self.cap is None or not self.cap.isOpened():
            return False
            
        # Check if we're receiving recent frames
        current_time = time.time()
        return (current_time - self.last_frame_time) < 5.0  # 5 seconds timeout
        
    def get_stream_info(self) -> dict:
        """Get stream information"""
        if self.cap is None:
            return {}
            
        return {
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": self.cap.get(cv2.CAP_PROP_FPS),
            "total_frames": self.total_frames,
            "dropped_frames": self.dropped_frames,
            "drop_rate": self.dropped_frames / max(self.total_frames, 1) * 100,
            "connected": self.is_connected()
        }
        
    def release(self):
        """Release resources"""
        print("[INFO] Releasing RTSP resources...")
        self.running = False
        
        # Stop capture thread
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=3)
            
        # Release video capture
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
        # Clear frame queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
                
        print("[OK] RTSP resources released")
        
    def __del__(self):
        """Destructor"""
        self.release()