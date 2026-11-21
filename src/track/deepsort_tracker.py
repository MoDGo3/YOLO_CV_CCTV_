"""
DeepSORT Tracker - Multi-Object Tracking with Deep Association Metric
ติดตามบุคคลด้วย DeepSORT algorithm พร้อม ID ที่คงที่
"""

import numpy as np
from typing import List, Optional, Dict, Any
import cv2
import time
from collections import defaultdict

# Try to import deep_sort modules
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_REALTIME_AVAILABLE = True
    print("[OK] Using deep_sort_realtime")
except ImportError as e:
    DEEPSORT_REALTIME_AVAILABLE = False
    print(f"[WARNING] deep_sort_realtime not available: {e}, using simplified tracking")

class Track:
    """Enhanced track class for person tracking with improved features"""
    def __init__(self, track_id: int, bbox: List[float], confidence: float):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.confidence = confidence
        self.hits = 1
        self.hit_streak = 1
        self.time_since_update = 0
        self.age = 1
        
        # Enhanced tracking features
        self.velocity = [0.0, 0.0]  # [dx, dy] velocity estimation
        self.predicted_bbox = bbox.copy()  # Predicted next position
        self.bbox_history = [bbox.copy()]  # History for smoothing
        self.confidence_history = [confidence]  # Confidence history
        
        # Demographics data
        self.demographics = {}
        self.demographics_history = []
        
        # Enhanced trajectory tracking
        self.trajectory = []
        self.smooth_trajectory = []  # Smoothed trajectory
        self.max_trajectory_length = 50  # เพิ่มจาก 30
        
        # State management
        self.state = 'tentative'  # tentative, confirmed, deleted
        self.quality_score = confidence  # Overall track quality
        
        self.update_trajectory()
    
    def predict_next_position(self):
        """Predict next bounding box position using velocity"""
        if abs(self.velocity[0]) < 50 and abs(self.velocity[1]) < 50:  # Reasonable velocity
            predicted = [
                self.bbox[0] + self.velocity[0],
                self.bbox[1] + self.velocity[1],
                self.bbox[2] + self.velocity[0],
                self.bbox[3] + self.velocity[1]
            ]
            self.predicted_bbox = predicted
            return predicted
        return self.bbox.copy()
    
    def get_smooth_center(self):
        """Get smoothed center point from trajectory"""
        if len(self.smooth_trajectory) > 0:
            return self.smooth_trajectory[-1]
        return self.get_center()
        
    def update(self, bbox: List[float], confidence: float):
        """Update track with new detection including velocity estimation"""
        # Calculate velocity
        if len(self.bbox_history) > 0:
            prev_center = self.get_center()
            new_center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            self.velocity = [new_center[0] - prev_center[0], new_center[1] - prev_center[1]]
        
        # Update bbox with smoothing
        if len(self.bbox_history) >= 3:
            # Simple moving average smoothing
            smooth_factor = 0.7
            self.bbox = [
                smooth_factor * bbox[i] + (1 - smooth_factor) * self.bbox[i]
                for i in range(4)
            ]
        else:
            self.bbox = bbox
            
        # Update confidence with history
        self.confidence = confidence
        self.confidence_history.append(confidence)
        if len(self.confidence_history) > 10:
            self.confidence_history.pop(0)
        
        # Update bbox history
        self.bbox_history.append(bbox.copy())
        if len(self.bbox_history) > 10:
            self.bbox_history.pop(0)
            
        # Update tracking stats
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0
        
        # Update quality score
        avg_confidence = sum(self.confidence_history) / len(self.confidence_history)
        stability_score = min(self.hit_streak / 10.0, 1.0)
        self.quality_score = (avg_confidence * 0.7 + stability_score * 0.3)
        self.age += 1
        
        if self.hit_streak >= 3:
            self.state = 'confirmed'
            
        self.update_trajectory()
        
    def mark_missed(self):
        """Mark track as missed in current frame"""
        self.hit_streak = 0
        self.time_since_update += 1
        self.age += 1
        
        # More conservative deletion - keep tracks longer
        if self.time_since_update > 50:  # Increased from 30 to 50
            self.state = 'deleted'
        elif self.time_since_update > 10 and self.state == 'confirmed':
            self.state = 'tentative'  # Demote to tentative if missing too long
            
    def update_trajectory(self):
        """Update trajectory with current center point"""
        center = self.get_center()
        self.trajectory.append(center)
        
        # Keep trajectory length manageable
        if len(self.trajectory) > self.max_trajectory_length:
            self.trajectory.pop(0)
            
    def get_center(self) -> tuple:
        """Get center point of bounding box"""
        x1, y1, x2, y2 = self.bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
        
    def get_area(self) -> float:
        """Get area of bounding box"""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)
        
    def to_tlbr(self) -> List[float]:
        """Convert to top-left bottom-right format"""
        return self.bbox
        
    def is_confirmed(self) -> bool:
        """Check if track is confirmed"""
        return self.state == 'confirmed'
        
    def is_deleted(self) -> bool:
        """Check if track should be deleted"""
        return self.state == 'deleted'
        
    def update_demographics(self, demographics_data):
        """Update demographics information"""
        if demographics_data:
            self.demographics = {
                'age': demographics_data.age,
                'gender': demographics_data.gender,
                'ethnicity': demographics_data.ethnicity,
                'group': demographics_data.group,
                'age_confidence': demographics_data.age_confidence,
                'gender_confidence': demographics_data.gender_confidence,
                'ethnicity_confidence': demographics_data.ethnicity_confidence,
                'face_bbox': demographics_data.face_bbox,
                'man_confidence': demographics_data.man_confidence,
                'woman_confidence': demographics_data.woman_confidence
            }
            
            # Keep history for stability
            self.demographics_history.append(self.demographics.copy())
            if len(self.demographics_history) > 10:
                self.demographics_history.pop(0)

class DeepSORTTracker:
    def __init__(self, config):
        self.config = config
        self.tracker = None
        self.tracks = {}  # track_id -> Track
        self.next_id = 1
        self.frame_count = 0
        
        # Tracking statistics
        self.total_tracks = 0
        self.active_tracks = 0
        self.confirmed_tracks = 0
        
        print(f"[INFO] Initializing DeepSORT Tracker")
        print(f"[INFO] Max Age: {config.DEEPSORT_MAX_AGE}")
        print(f"[INFO] Min Confidence: {config.DEEPSORT_MIN_CONFIDENCE}")
        
        self.initialize_tracker()
        
    def initialize_tracker(self):
        """Initialize the appropriate tracker"""
        if DEEPSORT_REALTIME_AVAILABLE:
            self.initialize_deepsort_realtime()
        else:
            self.initialize_simple_tracker()
            
    def initialize_deepsort_realtime(self):
        """Initialize DeepSort RealTime tracker"""
        try:
            self.tracker = DeepSort(
                max_age=self.config.DEEPSORT_MAX_AGE,
                n_init=self.config.DEEPSORT_N_INIT,
                nms_max_overlap=self.config.DEEPSORT_NMS_MAX_OVERLAP,
                max_cosine_distance=self.config.DEEPSORT_MAX_DIST,
                nn_budget=self.config.DEEPSORT_NN_BUDGET,
                gating_only_position=False,
                override_track_class=None,
                embedder="mobilenet",
                half=True,
                bgr=True,
                embedder_gpu=True,
                embedder_model_name=None,
                embedder_wts=None,
                polygon=False,
                today=None
            )
            print("[OK] DeepSORT RealTime initialized")
            
        except Exception as e:
            print(f"[ERROR] DeepSORT RealTime initialization failed: {e}")
            self.initialize_simple_tracker()
            
    def initialize_simple_tracker(self):
        """Initialize simple IoU-based tracker as fallback"""
        print("[INFO] Enhanced DeepSORT Tracker reset")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            self.cleanup()
        except:
            pass
        self.tracker = None
        
    def update(self, detections: List, frame: np.ndarray) -> List[Track]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of Detection objects from YOLO
            frame: Current frame for feature extraction
            
        Returns:
            List of active tracks
        """
        self.frame_count += 1
        
        if DEEPSORT_REALTIME_AVAILABLE and self.tracker is not None:
            return self.update_deepsort_realtime(detections, frame)
        else:
            return self.update_simple_tracker(detections)
            
    def update_deepsort_realtime(self, detections: List, frame: np.ndarray) -> List[Track]:
        """Update using DeepSORT RealTime"""
        try:
            # Convert detections to format expected by DeepSort
            # Format: List of tuples: ([left, top, w, h], confidence, class)
            raw_detections = []
            
            for detection in detections:
                if detection.confidence >= self.config.DEEPSORT_MIN_CONFIDENCE:
                    x1, y1, x2, y2 = detection.bbox
                    # Ensure valid bounding box
                    if x2 > x1 and y2 > y1:
                        w = x2 - x1
                        h = y2 - y1
                        # Format: ([left, top, w, h], confidence, class)
                        raw_detections.append(([x1, y1, w, h], detection.confidence, 0))
                    
            if not raw_detections:  # No valid detections
                # Mark all tracks as missed
                for track_id in list(self.tracks.keys()):
                    self.tracks[track_id].mark_missed()
                    if self.tracks[track_id].is_deleted():
                        del self.tracks[track_id]
                return list(self.tracks.values())
                    
            # Update tracker with correct API format
            tracks = self.tracker.update_tracks(raw_detections, embeds=None, frame=frame)
            
            # Convert to our Track objects
            active_tracks = []
            current_track_ids = set()
            
            for track in tracks:
                if track.is_confirmed() and hasattr(track, 'track_id'):
                    track_id = track.track_id
                    current_track_ids.add(track_id)
                    
                    # Get bbox - handle different DeepSort versions
                    try:
                        bbox = None
                        # Prefer tlbr (top-left, bottom-right)
                        if hasattr(track, 'to_tlbr'):
                            tlbr = track.to_tlbr()
                            if tlbr is not None and len(tlbr) == 4:
                                bbox = [float(tlbr[0]), float(tlbr[1]), float(tlbr[2]), float(tlbr[3])]
                        elif hasattr(track, 'to_ltrb'):
                            # Some versions expose ltrb (left, top, right, bottom)
                            ltrb = track.to_ltrb()
                            if ltrb is not None and len(ltrb) == 4:
                                bbox = [float(ltrb[0]), float(ltrb[1]), float(ltrb[2]), float(ltrb[3])]
                        elif hasattr(track, 'to_tlwh'):
                            # Convert tlwh (top-left, width, height) to tlbr
                            tlwh = track.to_tlwh()
                            if tlwh is not None and len(tlwh) == 4:
                                x, y, w, h = tlwh
                                bbox = [float(x), float(y), float(x + w), float(y + h)]
                        
                        if bbox is None:
                            # Fallback: skip if bbox not available
                            continue
                            
                        # Ensure valid bbox
                        if len(bbox) == 4 and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                            if track_id in self.tracks:
                                # Update existing track
                                self.tracks[track_id].update(bbox, 1.0)
                            else:
                                # Create new track
                                self.tracks[track_id] = Track(track_id, bbox, 1.0)
                                self.total_tracks += 1
                                
                            active_tracks.append(self.tracks[track_id])
                    except Exception as e:
                        print(f"[WARNING] Track bbox extraction error: {e}")
                        continue
                    
            # Mark missing tracks
            for track_id in list(self.tracks.keys()):
                if track_id not in current_track_ids:
                    self.tracks[track_id].mark_missed()
                    if self.tracks[track_id].is_deleted():
                        del self.tracks[track_id]
                        
            self.update_statistics()
            return active_tracks
            
        except Exception as e:
            print(f"[ERROR] DeepSORT update error: {e}")
            return []
            
    def update_simple_tracker(self, detections: List) -> List[Track]:
        """Update using simple IoU-based tracking"""
        current_detections = []
        
        # Filter detections by confidence
        for detection in detections:
            if detection.confidence >= self.config.DEEPSORT_MIN_CONFIDENCE:
                current_detections.append(detection)
                
        # Simple IoU-based association
        matched_tracks = []
        unmatched_detections = list(range(len(current_detections)))
        unmatched_tracks = list(self.tracks.keys())
        
        # Calculate IoU matrix
        if self.tracks and current_detections:
            iou_matrix = np.zeros((len(self.tracks), len(current_detections)))
            
            track_list = list(self.tracks.values())
            for t, track in enumerate(track_list):
                if track.is_deleted():
                    continue
                for d, detection in enumerate(current_detections):
                    iou_matrix[t, d] = self.calculate_iou(track.bbox, detection.bbox)
                    
            # Simple greedy matching with improved threshold
            iou_threshold = 0.2  # Lower threshold for better tracking
            while True:
                if iou_matrix.size == 0:
                    break
                    
                max_iou = np.max(iou_matrix)
                if max_iou < iou_threshold:
                    break
                    
                t_idx, d_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                
                # Match found
                track_id = track_list[t_idx].track_id
                detection = current_detections[d_idx]
                
                # Update track with detection
                self.tracks[track_id].update(detection.bbox, detection.confidence)
                matched_tracks.append(track_id)
                
                if track_id in unmatched_tracks:
                    unmatched_tracks.remove(track_id)
                if d_idx in unmatched_detections:
                    unmatched_detections.remove(d_idx)
                    
                # Remove from consideration
                iou_matrix[t_idx, :] = 0
                iou_matrix[:, d_idx] = 0
                
        # Create new tracks for unmatched detections
        for d_idx in unmatched_detections:
            detection = current_detections[d_idx]
            track_id = self.next_id
            self.next_id += 1
            
            self.tracks[track_id] = Track(track_id, detection.bbox, detection.confidence)
            self.total_tracks += 1
            
        # Mark unmatched tracks as missed
        for track_id in unmatched_tracks:
            self.tracks[track_id].mark_missed()
            
        # Remove deleted tracks
        deleted_tracks = [tid for tid, track in self.tracks.items() if track.is_deleted()]
        for track_id in deleted_tracks:
            del self.tracks[track_id]
            
        # Return active tracks
        active_tracks = [track for track in self.tracks.values() if not track.is_deleted()]
        
        self.update_statistics()
        return active_tracks
        
    def calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
            
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
        
    def update_statistics(self):
        """Update tracking statistics"""
        self.active_tracks = len([t for t in self.tracks.values() if not t.is_deleted()])
        self.confirmed_tracks = len([t for t in self.tracks.values() if t.is_confirmed()])
        
    def get_track_by_id(self, track_id: int) -> Optional[Track]:
        """Get track by ID"""
        return self.tracks.get(track_id)
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get tracking statistics"""
        return {
            "frame_count": self.frame_count,
            "total_tracks": self.total_tracks,
            "active_tracks": self.active_tracks,
            "confirmed_tracks": self.confirmed_tracks,
            "tracker_type": "DeepSORT" if DEEPSORT_REALTIME_AVAILABLE and self.tracker else "SimpleIoU",
            "config": {
                "max_age": self.config.DEEPSORT_MAX_AGE,
                "min_confidence": self.config.DEEPSORT_MIN_CONFIDENCE,
                "max_iou_distance": self.config.DEEPSORT_MAX_IOU_DISTANCE
            }
        }
        
    def reset(self):
        """Reset tracker state"""
        self.tracks.clear()
        self.next_id = 1
        self.frame_count = 0
        self.total_tracks = 0
        self.active_tracks = 0
        self.confirmed_tracks = 0
        print("[INFO] Tracker reset")
        
    def release(self):
        """Release resources"""
        if hasattr(self.tracker, 'release'):
            self.tracker.release()
        self.tracks.clear()
        print("[INFO] Tracker resources released")
        
    def __del__(self):
        """Destructor"""
        self.release()