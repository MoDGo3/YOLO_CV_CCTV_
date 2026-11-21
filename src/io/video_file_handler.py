"""
Video File Handler - Local Video File Management
รองรับการเปิดไฟล์วิดีโอ (MP4, AVI, MKV, etc.) สำหรับการทดสอบและวิเคราะห์
"""

import cv2
import threading
import time
import queue
from pathlib import Path
from typing import Optional
import numpy as np


class VideoFileHandler:
    def __init__(self, video_path: str, buffer_size: int = 1, loop: bool = False):
        """
        Initialize Video File Handler
        
        Args:
            video_path: Path to video file (MP4, AVI, MKV, etc.)
            buffer_size: Frame buffer size (default 1 for real-time)
            loop: Whether to loop video when finished (default False)
        """
        self.video_path = video_path
        self.buffer_size = buffer_size
        self.loop = loop
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.running = False
        self.capture_thread = None
        self.paused = False
        
        # Video properties
        self.total_frames = 0
        self.current_frame = 0
        self.fps = 0
        self.width = 0
        self.height = 0
        self.duration = 0
        
        # Statistics
        self.read_frames = 0
        self.dropped_frames = 0
        
        # Playback control
        self.playback_speed = 1.0  # 1.0 = normal, 0.5 = slow, 2.0 = fast
        
        print(f"[INFO] Initializing Video File: {video_path}")
        self.connect()
        
    def connect(self):
        """Open video file or webcam"""
        try:
            # Check if input is webcam index (integer)
            is_webcam = False
            try:
                webcam_index = int(self.video_path)
                is_webcam = True
            except ValueError:
                is_webcam = False
            
            if not is_webcam:
                # Check if file exists
                if not Path(self.video_path).exists():
                    raise FileNotFoundError(f"Video file not found: {self.video_path}")
            
            # Release existing connection
            if self.cap is not None:
                self.cap.release()
                
            # Open video file or webcam
            if is_webcam:
                self.cap = cv2.VideoCapture(webcam_index)
                print(f"[INFO] Opening Webcam index: {webcam_index}")
            else:
                self.cap = cv2.VideoCapture(self.video_path)
            
            if not self.cap.isOpened():
                source = f"webcam {webcam_index}" if is_webcam else self.video_path
                raise IOError(f"Cannot open: {source}")
            
            # Get video properties
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.duration = self.total_frames / self.fps if self.fps > 0 else 0
            
            print(f"[OK] Video file opened successfully")
            print(f"[INFO] Resolution: {self.width}x{self.height}")
            print(f"[INFO] FPS: {self.fps:.2f}")
            print(f"[INFO] Duration: {self.duration:.2f} seconds ({self.total_frames} frames)")
            print(f"[INFO] Loop mode: {'ON' if self.loop else 'OFF'}")
            
            self.start_capture_thread()
            
        except Exception as e:
            print(f"[ERROR] Video file open failed: {e}")
            raise
            
    def start_capture_thread(self):
        """Start background thread for frame capture"""
        if self.capture_thread is not None and self.capture_thread.is_alive():
            self.running = False
            self.capture_thread.join(timeout=2)
            
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        print("[INFO] Video frame capture thread started")
        
    def _capture_loop(self):
        """Background loop for capturing frames"""
        frame_delay = 1.0 / self.fps if self.fps > 0 else 1.0 / 30.0
        
        while self.running:
            try:
                # Handle pause
                if self.paused:
                    time.sleep(0.1)
                    continue
                
                if self.cap is None or not self.cap.isOpened():
                    time.sleep(0.1)
                    continue
                    
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    # End of video
                    if self.loop:
                        print("[INFO] Video ended, looping...")
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.current_frame = 0
                        continue
                    else:
                        print("[INFO] Video playback finished")
                        self.running = False
                        break
                
                self.current_frame += 1
                self.read_frames += 1
                
                # Update queue
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                        self.dropped_frames += 1
                    except queue.Empty:
                        pass
                        
                self.frame_queue.put(frame)
                
                # Control playback speed
                adjusted_delay = frame_delay / self.playback_speed
                time.sleep(adjusted_delay)
                
            except Exception as e:
                print(f"[WARNING] Frame capture error: {e}")
                time.sleep(0.1)
                
    def get_frame(self) -> Optional[np.ndarray]:
        """Get latest frame from buffer"""
        try:
            return self.frame_queue.get(timeout=1.0)
        except queue.Empty:
            return None
            
    def seek(self, frame_number: int):
        """Seek to specific frame"""
        if self.cap is not None:
            frame_number = max(0, min(frame_number, self.total_frames - 1))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self.current_frame = frame_number
            print(f"[INFO] Seeked to frame {frame_number}/{self.total_frames}")
            
    def seek_time(self, seconds: float):
        """Seek to specific time in seconds"""
        if self.fps > 0:
            frame_number = int(seconds * self.fps)
            self.seek(frame_number)
            
    def pause(self):
        """Pause video playback"""
        self.paused = True
        print("[INFO] Video paused")
        
    def resume(self):
        """Resume video playback"""
        self.paused = False
        print("[INFO] Video resumed")
        
    def toggle_pause(self):
        """Toggle pause/resume"""
        if self.paused:
            self.resume()
        else:
            self.pause()
            
    def set_playback_speed(self, speed: float):
        """Set playback speed (0.25 - 4.0)"""
        self.playback_speed = max(0.25, min(4.0, speed))
        print(f"[INFO] Playback speed: {self.playback_speed}x")
        
    def get_progress(self) -> dict:
        """Get current playback progress"""
        current_time = self.current_frame / self.fps if self.fps > 0 else 0
        return {
            "current_frame": self.current_frame,
            "total_frames": self.total_frames,
            "current_time": current_time,
            "duration": self.duration,
            "progress_percent": (self.current_frame / self.total_frames * 100) if self.total_frames > 0 else 0,
            "paused": self.paused,
            "loop": self.loop,
            "playback_speed": self.playback_speed
        }
        
    def is_connected(self) -> bool:
        """Check if video file is open and playback is active"""
        return self.cap is not None and self.cap.isOpened() and self.running
        
    def get_stream_info(self) -> dict:
        """Get video file information"""
        return {
            "path": self.video_path,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "duration": self.duration,
            "read_frames": self.read_frames,
            "dropped_frames": self.dropped_frames,
            "loop": self.loop
        }
        
    def release(self):
        """Release video file resources"""
        self.running = False
        self.paused = False
        
        if self.capture_thread is not None and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2)
            
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
        # Clear frame queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
                
        print("[INFO] Video file handler released")
        
    def __del__(self):
        """Destructor"""
        self.release()
