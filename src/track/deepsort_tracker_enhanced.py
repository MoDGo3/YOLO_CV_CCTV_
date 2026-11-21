"""
Enhanced DeepSORT Tracker with Advanced Features
Improved tracking performance for CCTV applications
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Try to import DeepSORT
DEEPSORT_REALTIME_AVAILABLE = False
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_REALTIME_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] DeepSORT not available: {e} - using enhanced simple tracker")

@dataclass
class Detection:
    """Detection data structure"""
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int = 0

class EnhancedTrack:
    """Enhanced track class with advanced tracking features"""
    
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
        
        # Kalman filter state (simplified)
        self.position_variance = 10.0
        self.velocity_variance = 1.0
        
        # Demographics data
        self.demographics = {}
        self.demographics_history = []
        
        # Enhanced trajectory tracking
        self.trajectory = []
        self.smooth_trajectory = []  # Smoothed trajectory
        self.max_trajectory_length = 50
        
        # State management
        self.state = 'tentative'  # tentative, confirmed, deleted
        self.quality_score = confidence  # Overall track quality
        
        self.update_trajectory()
    
    def predict_next_position(self):
        """Predict next bounding box position using enhanced velocity estimation"""
        if abs(self.velocity[0]) < 100 and abs(self.velocity[1]) < 100:  # Reasonable velocity
            # Apply velocity with some noise reduction
            smooth_velocity = [
                self.velocity[0] * 0.8,  # Damping factor
                self.velocity[1] * 0.8
            ]
            
            predicted = [
                self.bbox[0] + smooth_velocity[0],
                self.bbox[1] + smooth_velocity[1],
                self.bbox[2] + smooth_velocity[0],
                self.bbox[3] + smooth_velocity[1]
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
        """Update track with new detection including advanced features"""
        # Calculate velocity with smoothing
        if len(self.bbox_history) > 0:
            prev_center = self.get_center()
            new_center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            new_velocity = [new_center[0] - prev_center[0], new_center[1] - prev_center[1]]
            
            # Smooth velocity to reduce noise
            alpha = 0.3  # Velocity smoothing factor
            self.velocity = [
                alpha * new_velocity[0] + (1 - alpha) * self.velocity[0],
                alpha * new_velocity[1] + (1 - alpha) * self.velocity[1]
            ]
        
        # Update bbox with adaptive smoothing
        if len(self.bbox_history) >= 3:
            # Adaptive smoothing based on confidence
            smooth_factor = min(0.8, confidence * 1.2)  # Higher confidence = less smoothing
            self.bbox = [
                smooth_factor * bbox[i] + (1 - smooth_factor) * self.bbox[i]
                for i in range(4)
            ]
        else:
            self.bbox = bbox
            
        # Update confidence with weighted history
        self.confidence = confidence
        self.confidence_history.append(confidence)
        if len(self.confidence_history) > 15:
            self.confidence_history.pop(0)
        
        # Update bbox history
        self.bbox_history.append(bbox.copy())
        if len(self.bbox_history) > 15:
            self.bbox_history.pop(0)
            
        # Update tracking stats
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0
        
        # Enhanced quality score calculation
        avg_confidence = sum(self.confidence_history) / len(self.confidence_history)
        stability_score = min(self.hit_streak / 15.0, 1.0)
        consistency_score = 1.0 - (np.std(self.confidence_history) if len(self.confidence_history) > 1 else 0)
        
        self.quality_score = (avg_confidence * 0.5 + stability_score * 0.3 + consistency_score * 0.2)
        self.age += 1
        
        # State transition with enhanced criteria
        if self.hit_streak >= 2 and self.quality_score > 0.4:
            self.state = 'confirmed'
        elif self.quality_score < 0.2 and self.age > 20:
            self.state = 'tentative'
            
        self.update_trajectory()
        
    def mark_missed(self):
        """Mark track as missed with adaptive deletion criteria"""
        self.hit_streak = 0
        self.time_since_update += 1
        self.age += 1
        
        # Adaptive deletion based on track quality
        max_age = 80 if self.quality_score > 0.5 else 50
        
        if self.time_since_update > max_age:
            self.state = 'deleted'
        elif self.time_since_update > 15 and self.state == 'confirmed':
            self.state = 'tentative'  # Demote to tentative
            
    def update_trajectory(self):
        """Update trajectory with enhanced smoothing"""
        center = self.get_center()
        self.trajectory.append(center)
        
        # Create smooth trajectory using moving average
        if len(self.trajectory) >= 3:
            # Weighted moving average (recent points have higher weight)
            weights = [0.2, 0.3, 0.5]  # Recent point gets highest weight
            smooth_center = [
                sum(w * self.trajectory[i][0] for i, w in zip(range(-3, 0), weights)) / sum(weights),
                sum(w * self.trajectory[i][1] for i, w in zip(range(-3, 0), weights)) / sum(weights)
            ]
            self.smooth_trajectory.append(smooth_center)
        else:
            self.smooth_trajectory.append(center)
        
        # Limit trajectory length
        if len(self.trajectory) > self.max_trajectory_length:
            self.trajectory.pop(0)
        if len(self.smooth_trajectory) > self.max_trajectory_length:
            self.smooth_trajectory.pop(0)
            
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
        """Check if track is confirmed with enhanced criteria"""
        return (self.state == 'confirmed' and 
                self.quality_score > 0.3 and 
                self.hit_streak >= 2)
        
    def is_deleted(self) -> bool:
        """Check if track should be deleted with enhanced criteria"""
        return (self.state == 'deleted' or 
                self.time_since_update > 80 or
                (self.quality_score < 0.1 and self.age > 10))
        
    def get_track_quality(self) -> Dict:
        """Get comprehensive track quality metrics"""
        return {
            'quality_score': self.quality_score,
            'hit_streak': self.hit_streak,
            'age': self.age,
            'avg_confidence': sum(self.confidence_history) / len(self.confidence_history),
            'confidence_stability': 1.0 - (np.std(self.confidence_history) if len(self.confidence_history) > 1 else 0),
            'velocity_magnitude': (self.velocity[0]**2 + self.velocity[1]**2)**0.5,
            'trajectory_length': len(self.trajectory),
            'state': self.state
        }
        
    def update_demographics(self, demographics_data):
        """Update demographics information with stability checks"""
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
            
            # Keep history for stability analysis
            self.demographics_history.append(self.demographics.copy())
            if len(self.demographics_history) > 10:
                self.demographics_history.pop(0)


class EnhancedDeepSORTTracker:
    """Enhanced DeepSORT tracker with advanced features and adaptive algorithms"""
    
    def __init__(self, config):
        self.config = config
        self.deepsort_tracker = None
        self.tracks = {}  # track_id -> EnhancedTrack
        self.next_track_id = 1
        
        # Enhanced tracking features
        self.track_history = {}  # Complete track history
        self.lost_tracks = {}    # Recently lost tracks for recovery
        self.track_quality_threshold = 0.3
        
        # Adaptive thresholding
        self.dynamic_confidence_threshold = config.DEEPSORT_MIN_CONFIDENCE
        self.frame_count = 0
        self.scene_analysis = {
            'density_history': [],
            'motion_level': 'normal',
            'last_tuning': 0
        }
        
        # Enhanced statistics
        self.stats = {
            'total_tracks_created': 0,
            'successful_associations': 0,
            'failed_associations': 0,
            'track_recoveries': 0,
            'quality_improvements': 0
        }
        
        print("[INFO] Initializing Enhanced DeepSORT Tracker")
        print(f"[INFO] Max Age: {config.DEEPSORT_MAX_AGE}")
        print(f"[INFO] Min Confidence: {config.DEEPSORT_MIN_CONFIDENCE}")
        print("[INFO] Advanced Features: Velocity prediction, Quality scoring, Adaptive thresholds")
        
        self.initialize_tracker()
        
    def initialize_tracker(self):
        """Initialize the appropriate tracker with enhanced configuration"""
        if DEEPSORT_REALTIME_AVAILABLE:
            try:
                self.deepsort_tracker = DeepSort(
                    max_age=self.config.DEEPSORT_MAX_AGE,
                    n_init=2,  # Reduced for faster confirmation
                    max_iou_distance=0.6,  # Slightly more permissive
                    max_cosine_distance=self.config.DEEPSORT_MAX_DIST,
                    nn_budget=100,
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
                print("[OK] Enhanced DeepSORT initialized successfully")
                return True
            except Exception as e:
                print(f"[ERROR] Enhanced DeepSORT initialization failed: {e}")
                self.deepsort_tracker = None
                return False
        else:
            print("[INFO] Using enhanced simple tracker mode")
            return False
    
    def update(self, detections, frame=None):
        """Enhanced update with adaptive algorithms and track recovery"""
        self.frame_count += 1
        
        if not detections:
            return self._age_tracks()
        
        # Analyze scene characteristics
        self._analyze_scene(detections)
        
        # Update adaptive threshold
        self._update_adaptive_threshold(detections)
        
        # Filter detections with enhanced criteria
        filtered_detections = self._filter_detections_enhanced(detections)
        
        if not filtered_detections:
            return self._age_tracks()
        
        # Main tracking update
        if DEEPSORT_REALTIME_AVAILABLE and self.deepsort_tracker:
            tracks = self._update_enhanced_deepsort(filtered_detections, frame)
        else:
            tracks = self._update_enhanced_simple_tracker(filtered_detections)
        
        # Post-processing enhancements
        tracks = self._post_process_tracks(tracks, filtered_detections)
        
        # Update tracking history and analytics
        self._update_analytics(tracks)
        
        return tracks
    
    def _analyze_scene(self, detections):
        """Analyze scene characteristics for adaptive behavior"""
        detection_count = len(detections)
        self.scene_analysis['density_history'].append(detection_count)
        
        # Keep last 30 frames for analysis
        if len(self.scene_analysis['density_history']) > 30:
            self.scene_analysis['density_history'].pop(0)
        
        # Analyze motion level based on detection stability
        if len(self.scene_analysis['density_history']) >= 10:
            recent_std = np.std(self.scene_analysis['density_history'][-10:])
            if recent_std > 3:
                self.scene_analysis['motion_level'] = 'high'
            elif recent_std < 1:
                self.scene_analysis['motion_level'] = 'low'
            else:
                self.scene_analysis['motion_level'] = 'normal'
    
    def _update_adaptive_threshold(self, detections):
        """Advanced adaptive threshold adjustment"""
        if not detections:
            return
            
        confidences = [d.confidence for d in detections]
        avg_confidence = sum(confidences) / len(confidences)
        detection_density = len(detections)
        
        # Multi-factor adaptive adjustment
        base_threshold = self.config.DEEPSORT_MIN_CONFIDENCE
        
        # Factor 1: Average confidence
        if avg_confidence > 0.8:
            confidence_factor = 1.1
        elif avg_confidence < 0.4:
            confidence_factor = 0.9
        else:
            confidence_factor = 1.0
        
        # Factor 2: Detection density
        if detection_density > 15:  # Crowded scene
            density_factor = 1.15
        elif detection_density < 3:  # Sparse scene
            density_factor = 0.85
        else:
            density_factor = 1.0
        
        # Factor 3: Motion level
        motion_factors = {'low': 0.9, 'normal': 1.0, 'high': 1.1}
        motion_factor = motion_factors[self.scene_analysis['motion_level']]
        
        # Combine factors with limits
        combined_factor = confidence_factor * density_factor * motion_factor
        self.dynamic_confidence_threshold = np.clip(
            base_threshold * combined_factor,
            base_threshold - 0.15,  # Lower bound
            base_threshold + 0.2    # Upper bound
        )
    
    def _filter_detections_enhanced(self, detections):
        """Enhanced detection filtering with multiple criteria"""
        filtered = []
        for detection in detections:
            if self._is_valid_detection_enhanced(detection):
                filtered.append(detection)
        return filtered
    
    def _is_valid_detection_enhanced(self, detection):
        """Enhanced detection validation with multiple quality checks"""
        # Basic confidence check
        if detection.confidence < self.dynamic_confidence_threshold:
            return False
        
        # Geometric validation
        x1, y1, x2, y2 = detection.bbox
        width, height = x2 - x1, y2 - y1
        aspect_ratio = width / height if height > 0 else 0
        area = width * height
        
        # Enhanced quality criteria
        size_valid = width >= 15 and height >= 30 and area >= 600
        aspect_valid = 0.1 <= aspect_ratio <= 2.0  # More permissive for CCTV
        confidence_valid = detection.confidence >= 0.15  # Minimum absolute threshold
        
        # Additional checks for very small detections
        if area < 1200:  # Small detection
            return detection.confidence >= 0.4  # Higher confidence required
        
        return size_valid and aspect_valid and confidence_valid
    
    def _update_enhanced_deepsort(self, detections, frame):
        """Enhanced DeepSORT update with custom track handling"""
        try:
            # Convert to DeepSORT format: List of tuples ([x, y, w, h], confidence, class)
            raw_detections = []
            for det in detections:
                x1, y1, x2, y2 = det.bbox
                raw_detections.append(([x1, y1, x2-x1, y2-y1], det.confidence, det.class_id))
            
            if not raw_detections:  # No detections
                return list(self.tracks.values())
            
            # Update DeepSORT with correct API format
            deepsort_tracks = self.deepsort_tracker.update_tracks(
                raw_detections, 
                embeds=None,
                frame=frame
            )
            
            # Convert to enhanced Track objects
            tracks = []
            for track_dict in deepsort_tracks:
                if hasattr(track_dict, 'track_id') and track_dict.is_confirmed():
                    track_id = track_dict.track_id
                    bbox = track_dict.to_ltrb()
                    confidence = getattr(track_dict, 'conf', 0.5)
                    
                    if track_id in self.tracks:
                        # Update existing track
                        self.tracks[track_id].update(bbox, confidence)
                        track = self.tracks[track_id]
                    else:
                        # Create new enhanced track
                        track = EnhancedTrack(track_id, bbox, confidence)
                        self.tracks[track_id] = track
                        self.stats['total_tracks_created'] += 1
                    
                    # Predict next position
                    track.predict_next_position()
                    tracks.append(track)
                    self.stats['successful_associations'] += 1
            
            return tracks
            
        except Exception as e:
            print(f"[WARNING] DeepSORT error: {e}")
            self.stats['failed_associations'] += 1
            return self._update_enhanced_simple_tracker(detections)
    
    def _update_enhanced_simple_tracker(self, detections):
        """Enhanced simple tracker with IoU matching and quality scoring"""
        # Age existing tracks
        for track in list(self.tracks.values()):
            track.mark_missed()
            if track.is_deleted():
                del self.tracks[track.track_id]
        
        # Match detections to tracks using enhanced IoU
        matched_tracks = set()
        unmatched_detections = []
        
        for detection in detections:
            best_track = None
            best_iou = 0.3  # Threshold for matching
            
            for track in self.tracks.values():
                if track.track_id in matched_tracks:
                    continue
                    
                # Use predicted position if available
                compare_bbox = track.predicted_bbox if hasattr(track, 'predicted_bbox') else track.bbox
                iou = self._calculate_enhanced_iou(detection.bbox, compare_bbox)
                
                if iou > best_iou:
                    best_iou = iou
                    best_track = track
            
            if best_track:
                best_track.update(detection.bbox, detection.confidence)
                matched_tracks.add(best_track.track_id)
                self.stats['successful_associations'] += 1
            else:
                unmatched_detections.append(detection)
                self.stats['failed_associations'] += 1
        
        # Create new tracks for unmatched detections
        for detection in unmatched_detections:
            if detection.confidence >= self.dynamic_confidence_threshold:
                new_track = EnhancedTrack(self.next_track_id, detection.bbox, detection.confidence)
                self.tracks[self.next_track_id] = new_track
                self.next_track_id += 1
                self.stats['total_tracks_created'] += 1
        
        # Return confirmed tracks only
        return [track for track in self.tracks.values() if track.is_confirmed()]
    
    def _post_process_tracks(self, tracks, detections):
        """Post-process tracks with recovery and quality enhancement"""
        # Attempt track recovery
        recovered_tracks = self._attempt_track_recovery(tracks, detections)
        
        # Quality enhancement for existing tracks
        for track in recovered_tracks:
            if hasattr(track, 'get_track_quality'):
                quality = track.get_track_quality()
                if quality['quality_score'] > 0.7:
                    self.stats['quality_improvements'] += 1
        
        return recovered_tracks
    
    def _attempt_track_recovery(self, tracks, detections):
        """Advanced track recovery with predictive matching"""
        current_track_ids = {track.track_id for track in tracks}
        
        # Try to recover lost tracks
        for lost_id, lost_track in list(self.lost_tracks.items()):
            if lost_id in current_track_ids:
                continue  # Already recovered
                
            # Find best matching detection
            best_match = self._find_predictive_match(lost_track, detections, tracks)
            if best_match:
                # Recover the track with enhanced initialization
                recovered_track = EnhancedTrack(lost_id, best_match.bbox, best_match.confidence)
                recovered_track.state = 'confirmed'
                recovered_track.age = lost_track.age  # Maintain age
                recovered_track.trajectory = lost_track.trajectory[-10:]  # Keep recent trajectory
                
                tracks.append(recovered_track)
                self.tracks[lost_id] = recovered_track
                self.stats['track_recoveries'] += 1
                del self.lost_tracks[lost_id]
        
        return tracks
    
    def _find_predictive_match(self, lost_track, detections, existing_tracks):
        """Find best matching detection using prediction and quality metrics"""
        if not hasattr(lost_track, 'predicted_bbox'):
            return None
            
        best_detection = None
        best_score = 0.2  # Minimum matching score
        
        pred_bbox = lost_track.predicted_bbox
        
        for detection in detections:
            # Check if detection is already assigned
            assigned = any(self._calculate_enhanced_iou(detection.bbox, track.bbox) > 0.4 
                         for track in existing_tracks)
            if assigned:
                continue
                
            # Calculate matching score
            iou_score = self._calculate_enhanced_iou(detection.bbox, pred_bbox)
            confidence_score = detection.confidence
            size_consistency = self._calculate_size_consistency(detection.bbox, lost_track.bbox_history[-1] if lost_track.bbox_history else lost_track.bbox)
            
            # Combined matching score
            match_score = (iou_score * 0.5 + confidence_score * 0.3 + size_consistency * 0.2)
            
            if match_score > best_score:
                best_score = match_score
                best_detection = detection
        
        return best_detection
    
    def _calculate_enhanced_iou(self, bbox1, bbox2):
        """Enhanced IoU calculation with edge case handling"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection with bounds checking
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
            
        intersection = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
        area1 = max(0, x2_1 - x1_1) * max(0, y2_1 - y1_1)
        area2 = max(0, x2_2 - x1_2) * max(0, y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_size_consistency(self, bbox1, bbox2):
        """Calculate size consistency between bounding boxes"""
        w1, h1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
        w2, h2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]
        
        if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
            return 0.0
        
        size_ratio = min(w1/w2, w2/w1) * min(h1/h2, h2/h1)
        return size_ratio
    
    def _update_analytics(self, tracks):
        """Update comprehensive tracking analytics"""
        current_ids = {track.track_id for track in tracks}
        
        # Update track history
        for track in tracks:
            self.track_history[track.track_id] = {
                'last_seen': self.frame_count,
                'track': track,
                'quality': track.get_track_quality()
            }
        
        # Manage lost tracks
        for track_id in list(self.track_history.keys()):
            if track_id not in current_ids:
                if self.frame_count - self.track_history[track_id]['last_seen'] < 15:
                    self.lost_tracks[track_id] = self.track_history[track_id]['track']
                del self.track_history[track_id]
        
        # Clean old lost tracks
        for track_id in list(self.lost_tracks.keys()):
            if self.frame_count - self.lost_tracks[track_id].age > 30:
                del self.lost_tracks[track_id]
    
    def _age_tracks(self):
        """Age tracks when no detections are available"""
        for track in list(self.tracks.values()):
            track.mark_missed()
            if track.is_deleted():
                del self.tracks[track.track_id]
        
        return [track for track in self.tracks.values() if track.is_confirmed()]
    
    def get_enhanced_statistics(self):
        """Get comprehensive tracking statistics with performance metrics"""
        total_attempts = self.stats['successful_associations'] + self.stats['failed_associations']
        
        return {
            # Basic stats
            'total_tracks_created': self.stats['total_tracks_created'],
            'successful_associations': self.stats['successful_associations'],
            'failed_associations': self.stats['failed_associations'],
            'track_recoveries': self.stats['track_recoveries'],
            'quality_improvements': self.stats['quality_improvements'],
            
            # Current state
            'current_tracks': len(self.tracks),
            'active_history': len(self.track_history),
            'lost_tracks': len(self.lost_tracks),
            
            # Performance metrics
            'success_rate': self.stats['successful_associations'] / max(1, total_attempts),
            'recovery_rate': self.stats['track_recoveries'] / max(1, self.stats['total_tracks_created']),
            'quality_rate': self.stats['quality_improvements'] / max(1, self.stats['total_tracks_created']),
            
            # Adaptive parameters
            'adaptive_threshold': self.dynamic_confidence_threshold,
            'scene_motion': self.scene_analysis['motion_level'],
            'average_density': np.mean(self.scene_analysis['density_history']) if self.scene_analysis['density_history'] else 0
        }
    
    def draw_enhanced_visualization(self, frame, tracks):
        """Draw enhanced track visualization with quality indicators"""
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            # Quality-based color coding
            quality = track.get_track_quality()['quality_score']
            if quality > 0.7:
                color = (0, 255, 0)  # Green for high quality
                thickness = 3
            elif quality > 0.4:
                color = (0, 255, 255)  # Yellow for medium quality
                thickness = 2
            else:
                color = (0, 100, 255)  # Orange for low quality
                thickness = 2
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, track.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw track ID and quality
            label = f"ID:{track.track_id} Q:{quality:.2f}"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw trajectory (smooth version)
            if hasattr(track, 'smooth_trajectory') and len(track.smooth_trajectory) > 1:
                points = np.array(track.smooth_trajectory, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], False, color, 2)
            
            # Draw predicted position (faint)
            if hasattr(track, 'predicted_bbox'):
                px1, py1, px2, py2 = map(int, track.predicted_bbox)
                cv2.rectangle(frame, (px1, py1), (px2, py2), (128, 128, 128), 1)
        
        return frame
    
    def export_analytics_data(self):
        """Export comprehensive analytics data for analysis"""
        return {
            'statistics': self.get_enhanced_statistics(),
            'scene_analysis': self.scene_analysis,
            'track_data': {
                track_id: {
                    'quality_metrics': track_data['quality'],
                    'trajectory_length': len(track_data['track'].trajectory) if hasattr(track_data['track'], 'trajectory') else 0,
                    'age': track_data['track'].age,
                    'state': track_data['track'].state
                }
                for track_id, track_data in self.track_history.items()
            },
            'parameters': {
                'base_confidence': self.config.DEEPSORT_MIN_CONFIDENCE,
                'adaptive_confidence': self.dynamic_confidence_threshold,
                'max_age': self.config.DEEPSORT_MAX_AGE
            }
        }
    
    def reset_tracker(self):
        """Reset tracker state for new session"""
        self.tracks.clear()
        self.track_history.clear()
        self.lost_tracks.clear()
        self.next_track_id = 1
        self.frame_count = 0
        self.dynamic_confidence_threshold = self.config.DEEPSORT_MIN_CONFIDENCE
        
        # Reset statistics
        for key in self.stats:
            self.stats[key] = 0
        
        # Reset scene analysis
        self.scene_analysis['density_history'].clear()
        self.scene_analysis['motion_level'] = 'normal'
        self.scene_analysis['last_tuning'] = 0
        
        print("[INFO] Enhanced DeepSORT Tracker reset with advanced features")
    
    def cleanup(self):
        """Clean up resources with detailed reporting"""
        if hasattr(self.deepsort_tracker, 'delete'):
            self.deepsort_tracker.delete()
        
        # Print comprehensive statistics
        stats = self.get_enhanced_statistics()
        print("[INFO] Enhanced DeepSORT Tracker Cleanup")
        print(f"[INFO] Tracks Created: {stats['total_tracks_created']}")
        print(f"[OK] Success Rate: {stats['success_rate']:.1%}")
        print(f"[INFO] Recovery Rate: {stats['recovery_rate']:.1%}")
        print(f"[INFO] Quality Rate: {stats['quality_rate']:.1%}")
        
        self.reset_tracker()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            self.cleanup()
        except:
            pass

# Alias for backward compatibility
DeepSORTTracker = EnhancedDeepSORTTracker
Track = EnhancedTrack