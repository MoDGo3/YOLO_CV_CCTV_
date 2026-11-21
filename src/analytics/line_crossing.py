"""
Line Crossing Detection - IN/OUT Counting
ตรวจจับการข้ามเส้นเพื่อนับจำนวน IN/OUT พร้อม hysteresis และ cooldown
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

class CrossingEvent:
    """Class to represent a line crossing event"""
    def __init__(self, track_id: int, direction: str, timestamp: float, demographics: Dict = None):
        self.track_id = track_id
        self.direction = direction  # "IN" or "OUT"
        self.timestamp = timestamp
        self.demographics = demographics or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MQTT/logging"""
        return {
            "track_id": self.track_id,
            "direction": self.direction,
            "status": "entered" if self.direction == "IN" else "left",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.timestamp)),
            **self.demographics
        }

class LineCrossing:
    def __init__(self, config):
        self.config = config
        
        # Line configuration (store base/original coordinates from config)
        self.base_line_start = config.COUNTING_LINE["start"]
        self.base_line_end = config.COUNTING_LINE["end"]
        # Active (scaled) line coordinates used for computation/drawing
        self.line_start = self.base_line_start
        self.line_end = self.base_line_end
        self.line_thickness = config.COUNTING_LINE["thickness"]
        self.line_color = config.COUNTING_LINE["color"]
        
        # Crossing detection parameters
        self.buffer_distance = config.CROSSING_BUFFER_DISTANCE
        self.cooldown_time = config.CROSSING_COOLDOWN_TIME
        self.min_track_length = config.MIN_TRACK_LENGTH
        
        # Frame size and scaling state
        self._frame_size = None  # (width, height)
        self._last_scaled_from = None  # cache to avoid recompute
        
        # Track states for crossing detection
        self.track_states = {}  # track_id -> {"side": "top"/"bottom", "last_crossing": timestamp}
        self.track_crossings = {}  # track_id -> list of crossing points
        
        # Cooldown management
        self.last_crossing_time = {}  # track_id -> timestamp
        # External trajectories for trackers without built-in trajectory (e.g., YOLOv8 native)
        self._ext_trajectories = defaultdict(list)  # track_id -> list[(x,y)]

        print(f"[INFO] Line Crossing initialized")
        print(f"[INFO] Line (base): {self.base_line_start} -> {self.base_line_end}")
        print(f"[INFO] Cooldown: {self.cooldown_time}s")

    def update_frame_size(self, width: int, height: int):
        """Update current frame size and scale the counting line accordingly."""
        self._frame_size = (width, height)
        # Scale base line from configured input size to current frame size if different
        in_w, in_h = self.config.INPUT_WIDTH, self.config.INPUT_HEIGHT
        if in_w <= 0 or in_h <= 0:
            # Fallback: use base coordinates directly
            self.line_start = self.base_line_start
            self.line_end = self.base_line_end
            return
        if self._last_scaled_from == (width, height):
            return  # no change
        # If configured to center the counting line vertically, compute new coords
        if getattr(self.config, 'COUNTING_LINE_MODE', 'fixed') == 'center_vertical':
            margin_top = getattr(self.config, 'COUNTING_LINE_MARGIN_TOP', 100)
            margin_bottom = getattr(self.config, 'COUNTING_LINE_MARGIN_BOTTOM', 100)
            cx = width // 2
            y1 = int(margin_top)
            y2 = int(max(margin_top + 1, height - margin_bottom))
            self.line_start = (cx, y1)
            self.line_end = (cx, y2)
        else:
            sx = width / float(in_w)
            sy = height / float(in_h)
            x1, y1 = self.base_line_start
            x2, y2 = self.base_line_end
            self.line_start = (int(x1 * sx), int(y1 * sy))
            self.line_end = (int(x2 * sx), int(y2 * sy))
        self._last_scaled_from = (width, height)
        print(f"[INFO] Scaled line to frame {width}x{height}: {self.line_start} -> {self.line_end}")
        
    def check_crossings(self, tracks: List) -> List[CrossingEvent]:
        """
        Check for line crossings in current frame
        
        Args:
            tracks: List of active tracks
            
        Returns:
            List of crossing events
        """
        crossing_events = []
        current_time = time.time()
        
        for track in tracks:
            if not hasattr(track, 'is_confirmed') or not track.is_confirmed():
                continue
                
            track_id = track.track_id
            # Build/maintain trajectory history
            if hasattr(track, 'get_center'):
                center = track.get_center()
                if isinstance(center, (tuple, list)) and len(center) == 2:
                    self._ext_trajectories[track_id].append((int(center[0]), int(center[1])))
                    if len(self._ext_trajectories[track_id]) > 100:
                        self._ext_trajectories[track_id].pop(0)
            
            # Choose trajectory source: track.trajectory if present, else external
            traj = getattr(track, 'trajectory', None)
            if not traj or len(traj) < 2:
                traj = self._ext_trajectories.get(track_id, [])
            
            if len(traj) < 2:
                continue
            
            # Skip if in cooldown period
            if self._is_in_cooldown(track_id, current_time):
                continue
                
            # Previously skipped short trajectories; now allow them to pass to
            # _detect_crossing with an internal fallback to catch fast crossings.
            # We still require at least two points (already checked above).
                
            # Check for line crossing
            crossing_event = self._detect_crossing(track, current_time, traj)
            if crossing_event:
                crossing_events.append(crossing_event)
                self.last_crossing_time[track_id] = current_time
                
        # Clean up old track states
        self._cleanup_old_states(current_time)
        
        return crossing_events
        
    def _detect_crossing(self, track, current_time: float, traj: List[Tuple[int, int]]) -> Optional[CrossingEvent]:
        """Detect if track has crossed the counting line"""
        track_id = track.track_id

        # Get current and previous positions from provided trajectory
        current_pos = traj[-1]
        previous_pos = traj[-2]
        # Estimate bbox radius to account for near-line crossings even if center barely misses
        bbox_radius = 0.0
        try:
            bbox = None
            if hasattr(track, 'to_tlbr'):
                bbox = track.to_tlbr()
            if bbox is None and hasattr(track, 'bbox'):
                bbox = track.bbox
            if bbox is not None and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                w = max(0.0, float(x2) - float(x1))
                h = max(0.0, float(y2) - float(y1))
                # Use a fraction of bbox size as effective radius
                bbox_radius = 0.25 * min(w, h)
        except Exception:
            bbox_radius = 0.0

        # Check if trajectory crosses the line (segment intersection or near-line within buffer)
        crossed, direction = self._line_intersection_with_direction(previous_pos, current_pos, bbox_radius=bbox_radius)
        
        if not crossed:
            return None
            
        # Validate crossing with hysteresis; if it fails but the trajectory is still short,
        # allow a permissive early count to avoid missing fast crossings.
        if not self._validate_crossing_with_hysteresis(track, direction):
            # Fallback: permissive count for short trajectories
            if len(traj) <= max(4, self.min_track_length + 1):
                # Respect cooldown
                if self._is_in_cooldown(track_id, current_time):
                    return None
                # else allow
            else:
                return None
            
        # Create crossing event with demographics
        demographics = self._extract_demographics(track)
        
        crossing_event = CrossingEvent(
            track_id=track_id,
            direction=direction,
            timestamp=current_time,
            demographics=demographics
        )
        
        print(f"[INFO] Crossing detected: ID {track_id} -> {direction}")
        
        return crossing_event
        
    def _line_intersection_with_direction(self, p1: Tuple[int, int], p2: Tuple[int, int], bbox_radius: float = 0.0) -> Tuple[bool, Optional[str]]:
        """
        Check if line segment p1-p2 intersects with counting line and determine direction
        
        Returns:
            (crossed, direction) where direction is "IN" or "OUT"
        """
        line_p1 = self.line_start
        line_p2 = self.line_end

        # Helper: segment intersection
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        def intersect(A, B, C, D):
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

        side1 = self._get_side_of_line(p1)
        side2 = self._get_side_of_line(p2)

        # Determine if segment crosses the finite counting line or is near the line
        seg_intersects = intersect(p1, p2, line_p1, line_p2)
        # Distance from movement segment to line segment (always compute)
        seg_dist = self._segment_to_segment_distance(p1, p2, line_p1, line_p2)
        near_motion = seg_dist <= (self.buffer_distance + float(bbox_radius))

        if not (seg_intersects or side1 != side2 or near_motion):
            return False, None

        # Direction
        if side1 != side2:
            # Use side transition for direction when available
            direction = "IN" if (side1 == "left" and side2 == "right") else "OUT"
        else:
            # Fallback: use movement vs line orientation (cross product sign)
            line_vec = (line_p2[0] - line_p1[0], line_p2[1] - line_p1[1])
            move_vec = (p2[0] - p1[0], p2[1] - p1[1])
            cross_product = line_vec[0] * move_vec[1] - line_vec[1] * move_vec[0]
            direction = "IN" if cross_product > 0 else "OUT"
        return True, direction
        
    def _determine_crossing_direction(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> str:
        """
        Deprecated: Kept for backward compatibility. Not used after direction fix.
        """
        side1 = self._get_side_of_line(p1)
        side2 = self._get_side_of_line(p2)
        return "IN" if (side1 == "left" and side2 == "right") else "OUT"
        
    def _validate_crossing_with_hysteresis(self, track, direction: str) -> bool:
        """
        Validate crossing using hysteresis to prevent oscillation
        
        Args:
            track: Track object
            direction: Proposed crossing direction
            
        Returns:
            True if crossing is valid
        """
        track_id = track.track_id
        
        # Get current side of the line
        current_side = self._get_side_of_line(track.get_center())
        
        # Initialize track state if new
        if track_id not in self.track_states:
            self.track_states[track_id] = {
                "side": current_side,
                "last_crossing": None
            }
            return False  # Don't count first detection
            
        previous_side = self.track_states[track_id]["side"]
        
        # Check if there's a clear transition
        if previous_side != current_side:
            # Valid transition - update state
            self.track_states[track_id]["side"] = current_side
            self.track_states[track_id]["last_crossing"] = time.time()
            
            # Validate direction matches the side transition
            expected_direction = "IN" if (previous_side == "left" and current_side == "right") else "OUT"
            
            return direction == expected_direction
            
        return False
        
    def _get_side_of_line(self, point: Tuple[int, int]) -> str:
        """Get which side of the line a point is on"""
        x, y = point
        
        # Use line equation to determine side; handle general (non-vertical) lines
        # Calculate which side of the infinite line the point lies on using cross product
        x1, y1 = self.line_start
        x2, y2 = self.line_end
        # Vector of the line
        vx, vy = (x2 - x1), (y2 - y1)
        # Vector from line start to point
        px, py = (x - x1), (y - y1)
        # z-component of cross product (2D)
        cross = vx * py - vy * px
        # For a line from start->end, define left side as cross > 0
        return "left" if cross > 0 else "right"

    def _segment_to_segment_distance(self, A: Tuple[int, int], B: Tuple[int, int], C: Tuple[int, int], D: Tuple[int, int]) -> float:
        """Compute the minimum distance between two line segments AB and CD"""
        def clamp(a, low, high):
            return max(low, min(a, high))

        Ax, Ay = A; Bx, By = B; Cx, Cy = C; Dx, Dy = D
        # Segment AB as vector and CD as vector
        ABx, ABy = Bx - Ax, By - Ay
        CDx, CDy = Dx - Cx, Dy - Cy
        ACx, ACy = Cx - Ax, Cy - Ay

        # Handle degenerate segments
        AB_len2 = ABx * ABx + ABy * ABy
        CD_len2 = CDx * CDx + CDy * CDy
        if AB_len2 == 0 and CD_len2 == 0:
            # Both points
            return ((Ax - Cx) ** 2 + (Ay - Cy) ** 2) ** 0.5
        if AB_len2 == 0:
            # Distance from A to segment CD
            return self._point_to_segment_distance(A, C, D)
        if CD_len2 == 0:
            # Distance from C to segment AB
            return self._point_to_segment_distance(C, A, B)

        # Compute parameters of closest points
        # Based on http://geomalgorithms.com/a07-_distance.html
        denom = ABx * ABx + ABy * ABy
        numer = (ABx * ACx + ABy * ACy)
        t0 = -numer / denom
        t0 = clamp(t0, 0.0, 1.0)
        closest_on_AB = (Ax + t0 * ABx, Ay + t0 * ABy)
        return self._point_to_segment_distance(closest_on_AB, C, D)

    def _point_to_segment_distance(self, P: Tuple[float, float], A: Tuple[int, int], B: Tuple[int, int]) -> float:
        """Compute the distance from point P to segment AB"""
        Ax, Ay = A; Bx, By = B; Px, Py = P
        ABx, ABy = Bx - Ax, By - Ay
        APx, APy = Px - Ax, Py - Ay
        ab2 = ABx * ABx + ABy * ABy
        if ab2 == 0:
            return ((Px - Ax) ** 2 + (Py - Ay) ** 2) ** 0.5
        t = (APx * ABx + APy * ABy) / ab2
        t = max(0.0, min(1.0, t))
        Cx = Ax + t * ABx
        Cy = Ay + t * ABy
        return ((Px - Cx) ** 2 + (Py - Cy) ** 2) ** 0.5
        
    def _is_in_cooldown(self, track_id: int, current_time: float) -> bool:
        """Check if track is in cooldown period"""
        if track_id not in self.last_crossing_time:
            return False
            
        time_since_last = current_time - self.last_crossing_time[track_id]
        return time_since_last < self.cooldown_time
        
    def _extract_demographics(self, track) -> Dict[str, Any]:
        """Extract demographics from track for event"""
        demographics = track.demographics.copy() if hasattr(track, 'demographics') else {}
        
        # Add bounding box and center
        bbox = None
        if hasattr(track, 'to_tlbr'):
            try:
                bbox = track.to_tlbr()
            except Exception:
                bbox = None
        if bbox is None and hasattr(track, 'bbox'):
            bbox = track.bbox
        
        try:
            center = track.get_center()
        except Exception:
            center = None
        
        demographics.update({
            "bbox": bbox,
            "center": center
        })
        
        return demographics
        
    def _cleanup_old_states(self, current_time: float):
        """Clean up old track states and crossings"""
        # Remove states for tracks not seen recently
        old_tracks = []
        for track_id, state in self.track_states.items():
            if state["last_crossing"] and (current_time - state["last_crossing"]) > 60:  # 1 minute
                old_tracks.append(track_id)
                
        for track_id in old_tracks:
            self.track_states.pop(track_id, None)
            self.last_crossing_time.pop(track_id, None)
            
    def draw_counting_line(self, frame: np.ndarray) -> np.ndarray:
        """Draw the counting line on frame"""
        cv2.line(
            frame,
            self.line_start,
            self.line_end,
            self.line_color,
            self.line_thickness
        )
        
        # Draw line label
        mid_point = (
            (self.line_start[0] + self.line_end[0]) // 2,
            (self.line_start[1] + self.line_end[1]) // 2
        )
        
        cv2.putText(
            frame,
            "COUNTING LINE",
            (mid_point[0] - 50, mid_point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.line_color,
            1
        )
        
        return frame
        
    def update_line_coordinates(self, start: Tuple[int, int], end: Tuple[int, int]):
        """Update counting line coordinates"""
        # Update base line and recompute scaled line if frame size known
        self.base_line_start = start
        self.base_line_end = end
        if self._frame_size:
            self.update_frame_size(*self._frame_size)
        else:
            self.line_start = start
            self.line_end = end
        
        # Clear existing states as line position changed
        self.track_states.clear()
        self.last_crossing_time.clear()
        
        print(f"[INFO] Line updated (base): {start} -> {end}")
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get line crossing statistics"""
        return {
            "line_coordinates": {
                "start": self.line_start,
                "end": self.line_end
            },
            "active_tracks": len(self.track_states),
            "cooldown_time": self.cooldown_time,
            "buffer_distance": self.buffer_distance,
            "min_track_length": self.min_track_length
        }
        
    def reset(self):
        """Reset all crossing states"""
        self.track_states.clear()
        self.last_crossing_time.clear()
        print("[INFO] Line crossing states reset")
        
    def __del__(self):
        """Destructor"""
        pass