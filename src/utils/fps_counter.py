"""
FPS Counter - Performance Monitoring
ตรวจสอบและคำนวณ FPS แบบเรียลไทม์
"""

import time
from collections import deque
from typing import Optional

class FPSCounter:
    def __init__(self, window_size: int = 30):
        """
        Initialize FPS counter
        
        Args:
            window_size: Number of frames to average over
        """
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.last_time = time.time()
        self.frame_count = 0
        
        # Statistics
        self.min_fps = float('inf')
        self.max_fps = 0.0
        self.avg_fps = 0.0
        
        print(f"[INFO] FPS Counter initialized (window size: {window_size})")
        
    def update(self, processing_time: Optional[float] = None):
        """
        Update FPS counter
        
        Args:
            processing_time: Optional processing time for this frame
        """
        current_time = time.time()
        
        if processing_time is not None:
            # Use provided processing time
            frame_time = processing_time
        else:
            # Calculate time since last update
            frame_time = current_time - self.last_time
            
        self.frame_times.append(frame_time)
        self.last_time = current_time
        self.frame_count += 1
        
        # Calculate current FPS
        if frame_time > 0:
            current_fps = 1.0 / frame_time
            
            # Update min/max
            self.min_fps = min(self.min_fps, current_fps)
            self.max_fps = max(self.max_fps, current_fps)
            
        # Calculate average FPS over window
        if len(self.frame_times) > 0:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            if avg_frame_time > 0:
                self.avg_fps = 1.0 / avg_frame_time
                
    def get_fps(self) -> float:
        """Get current average FPS"""
        return self.avg_fps
        
    def get_current_fps(self) -> float:
        """Get instantaneous FPS (based on last frame only)"""
        if len(self.frame_times) > 0 and self.frame_times[-1] > 0:
            return 1.0 / self.frame_times[-1]
        return 0.0
        
    def get_statistics(self) -> dict:
        """Get detailed FPS statistics"""
        return {
            "current_fps": self.get_current_fps(),
            "average_fps": self.avg_fps,
            "min_fps": self.min_fps if self.min_fps != float('inf') else 0.0,
            "max_fps": self.max_fps,
            "frame_count": self.frame_count,
            "window_size": len(self.frame_times),
            "avg_frame_time": sum(self.frame_times) / len(self.frame_times) if self.frame_times else 0.0
        }
        
    def reset(self):
        """Reset FPS counter"""
        self.frame_times.clear()
        self.frame_count = 0
        self.min_fps = float('inf')
        self.max_fps = 0.0
        self.avg_fps = 0.0
        self.last_time = time.time()
        print("[INFO] FPS counter reset")
        
    def is_performing_well(self, target_fps: float = 20.0) -> bool:
        """Check if system is performing above target FPS"""
        return self.avg_fps >= target_fps
        
    def get_performance_status(self) -> str:
        """Get performance status as string"""
        if self.avg_fps >= 25:
            return "EXCELLENT"
        elif self.avg_fps >= 20:
            return "GOOD"
        elif self.avg_fps >= 15:
            return "FAIR"
        elif self.avg_fps >= 10:
            return "POOR"
        else:
            return "CRITICAL"
            
    def __str__(self) -> str:
        return f"FPS: {self.avg_fps:.1f} (Status: {self.get_performance_status()})"
        
    def __repr__(self) -> str:
        return f"FPSCounter(avg_fps={self.avg_fps:.1f}, frames={self.frame_count})"