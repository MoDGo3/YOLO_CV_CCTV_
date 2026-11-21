#!/usr/bin/env python3
"""
YOLOv8 Native Tracker Implementation
ใช้ Ultralytics YOLO built-in tracking แทน DeepSORT แยก
"""

import cv2
import numpy as np
from collections import defaultdict
from typing import List, Dict, Optional
import time

class YOLOv8NativeTracker:
    """
    YOLO v8 Native Tracker using Ultralytics built-in tracking
    Support: BoT-SORT, ByteTrack
    """
    
    def __init__(self, config, tracker_type="botsort"):
        """
        Initialize YOLO v8 native tracker
        
        Args:
            config: Configuration object
            tracker_type: 'botsort' or 'bytetrack'
        """
        self.config = config
        self.tracker_type = tracker_type
        
        # Track history for visualization
        self.track_history = defaultdict(lambda: [])
        self.max_track_length = 30  # จำนวน frames ที่เก็บ track history
        
        # Statistics
        self.total_tracks = 0
        self.active_tracks = {}
        self.track_ages = {}
        
        print(f"[INFO] Initializing YOLOv8 Native Tracker: {tracker_type}")
    
    def update(self, model, frame):
        """
        Update tracks using YOLO native tracking
        
        Args:
            model: YOLO model instance
            frame: Current frame
            
        Returns:
            List of Track objects
        """
        start_time = time.time()
        
        try:
            # Run YOLO tracking with persist=True for track continuity
            results = model.track(
                source=frame,
                conf=self.config.YOLO_CONFIDENCE,
                iou=self.config.YOLO_IOU_THRESHOLD,
                classes=[0],  # Person class only
                tracker=f"{self.tracker_type}.yaml",  # Use specified tracker
                persist=True,  # Enable track persistence
                verbose=False,
                show=False,
                save=False
            )
            
            tracks = []
            
            if results and len(results) > 0:
                result = results[0]
                
                # Check if tracking data is available
                if result.boxes is not None and result.boxes.is_track:
                    boxes = result.boxes.xywh.cpu().numpy()  # x, y, w, h format
                    track_ids = result.boxes.id.int().cpu().tolist()
                    confidences = result.boxes.conf.cpu().numpy()
                    
                    for box, track_id, conf in zip(boxes, track_ids, confidences):
                        x_center, y_center, width, height = box
                        
                        # Convert to x1, y1, x2, y2 format
                        x1 = int(x_center - width / 2)
                        y1 = int(y_center - height / 2)
                        x2 = int(x_center + width / 2)
                        y2 = int(y_center + height / 2)
                        
                        bbox = [x1, y1, x2, y2]
                        
                        # Create Track object
                        track = Track(
                            track_id=track_id,
                            bbox=bbox,
                            confidence=conf,
                            class_id=0,
                            state='confirmed'  # YOLO native tracks are confirmed
                        )
                        
                        tracks.append(track)
                        
                        # Update track statistics
                        if track_id not in self.active_tracks:
                            self.total_tracks += 1
                            self.track_ages[track_id] = 1
                        else:
                            self.track_ages[track_id] += 1
                            
                        self.active_tracks[track_id] = track
                        
                        # Update track history for visualization
                        self.track_history[track_id].append((x_center, y_center))
                        if len(self.track_history[track_id]) > self.max_track_length:
                            self.track_history[track_id].pop(0)
            
            # Clean up old tracks that are no longer active
            current_track_ids = [track.track_id for track in tracks]
            inactive_ids = [tid for tid in self.active_tracks.keys() if tid not in current_track_ids]
            
            for tid in inactive_ids:
                del self.active_tracks[tid]
                if tid in self.track_ages:
                    del self.track_ages[tid]
            
            inference_time = time.time() - start_time
            
            return tracks
            
        except Exception as e:
            print(f"[ERROR] YOLOv8 Tracking Error: {e}")
            return []
    
    def draw_tracks(self, frame, tracks):
        """
        Draw tracking lines and information
        
        Args:
            frame: Current frame
            tracks: List of Track objects
            
        Returns:
            Frame with tracking visualization
        """
        # Draw tracking lines
        for track_id, history in self.track_history.items():
            if len(history) > 1:
                # Convert to numpy array for OpenCV
                points = np.array(history, dtype=np.int32).reshape((-1, 1, 2))
                
                # Choose color based on track age
                age = self.track_ages.get(track_id, 1)
                if age < 5:
                    color = (0, 255, 255)  # Yellow for new tracks
                elif age < 20:
                    color = (0, 255, 0)    # Green for stable tracks
                else:
                    color = (255, 0, 0)    # Blue for long tracks
                
                # Draw tracking line
                cv2.polylines(frame, [points], isClosed=False, 
                            color=color, thickness=3)
                
                # Draw track ID at the latest position
                if history:
                    latest_pos = history[-1]
                    cv2.putText(frame, f"T{track_id}", 
                              (int(latest_pos[0]) - 10, int(latest_pos[1]) - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    def get_statistics(self):
        """Get tracker statistics"""
        return {
            'total_tracks': self.total_tracks,
            'active_tracks': len(self.active_tracks),
            'tracker_type': self.tracker_type,
            'max_track_age': max(self.track_ages.values()) if self.track_ages else 0,
            'avg_track_age': sum(self.track_ages.values()) / len(self.track_ages) if self.track_ages else 0
        }


class Track:
    """
    Enhanced Track class compatible with YOLO native tracking
    """
    
    def __init__(self, track_id, bbox, confidence, class_id, state='confirmed'):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.confidence = confidence
        self.class_id = class_id
        self.state = state
        
        # Demographics data (to be filled by face analysis)
        self.age = None
        self.gender = None
        self.ethnicity = None
        self.demographics_confidence = 0.0
        
        # Track metadata
        self.hits = 1
        self.time_since_update = 0
        self.history = []
        
    def update_demographics(self, age, gender, ethnicity, confidence):
        """Update demographics information"""
        self.age = age
        self.gender = gender  
        self.ethnicity = ethnicity
        self.demographics_confidence = confidence
        
    def get_center(self):
        """Get bounding box center point"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
        
    def get_area(self):
        """Get bounding box area"""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)
        
    def is_confirmed(self):
        """Check if track is confirmed"""
        return self.state == 'confirmed'


def create_tracker_config(tracker_type="botsort", config_path=None):
    """
    Create custom tracker configuration
    
    Args:
        tracker_type: 'botsort' or 'bytetrack'
        config_path: Path to save custom config
        
    Returns:
        Path to tracker config file
    """
    
    if tracker_type == "botsort":
        tracker_config = """
# Ultralytics YOLO 🚀, AGPL-3.0 license
# Default YOLO tracker settings for BoT-SORT tracker https://github.com/NirAhron/BoT-SORT

tracker_type: botsort  # tracker type, ['botsort', 'bytetrack']
track_high_thresh: 0.5  # threshold for the first association
track_low_thresh: 0.1   # threshold for the second association  
new_track_thresh: 0.6   # threshold for init new track if the detection does not match any tracks
track_buffer: 30        # buffer to calculate the time when to remove tracks
match_thresh: 0.8       # threshold for matching tracks
proximity_thresh: 0.5   # threshold for spatial proximity (IoU) minimums
appearance_thresh: 0.25 # threshold for appearance similarity (ReID)
with_reid: False        # with ReID module
# gmc_method: sparseOptFlow # method of global motion compensation  
# gmc_method: orb
# gmc_method: ecc

# BoT-SORT settings
fuse_score: True        # Whether to fuse confidence scores with the IoU distances before matching
"""
    else:  # bytetrack
        tracker_config = """
# Ultralytics YOLO 🚀, AGPL-3.0 license
# Default YOLO tracker settings for ByteTrack tracker https://github.com/FoundationVision/ByteTrack

tracker_type: bytetrack  # tracker type, ['botsort', 'bytetrack']
track_high_thresh: 0.5   # threshold for the first association
track_low_thresh: 0.1    # threshold for the second association
new_track_thresh: 0.6    # threshold for init new track if the detection does not match any tracks
track_buffer: 30         # buffer to calculate the time when to remove tracks
match_thresh: 0.8        # threshold for matching tracks
fuse_score: True         # Whether to fuse confidence scores with the IoU distances before matching
"""

    if config_path:
        with open(config_path, 'w') as f:
            f.write(tracker_config)
        return config_path
    else:
        # Save to configs directory
        config_filename = f"configs/{tracker_type}_custom.yaml"
        with open(config_filename, 'w') as f:
            f.write(tracker_config)
        return config_filename


if __name__ == "__main__":
    # Test the tracker
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    
    from config import Config
    
    config = Config()
    
    # Create custom tracker configs
    botsort_config = create_tracker_config("botsort")
    bytetrack_config = create_tracker_config("bytetrack") 
    
    print("[OK] YOLOv8 Native Tracker created successfully!")
    print(f"[INFO] BoT-SORT config: {botsort_config}")
    print(f"[INFO] ByteTrack config: {bytetrack_config}")
    
    # Initialize tracker
    tracker = YOLOv8NativeTracker(config, "botsort")
    print(f"[INFO] Tracker initialized: {tracker.tracker_type}")