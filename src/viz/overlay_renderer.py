"""
Overlay Renderer - Real-time Display Graphics
แสดงผล Overlay, Bounding Box, และสถิติแบบเรียลไทม์
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
import time

class OverlayRenderer:
    def __init__(self, config):
        self.config = config
        
        # Display configuration
        self.display_config = config.get_display_config()
        self.font = self.display_config["font"]
        self.font_scale = self.display_config["font_scale"]
        self.font_thickness = self.display_config["font_thickness"]
        self.text_color = self.display_config["text_color"]
        self.bg_color = self.display_config["bg_color"]
        
        # Colors for different elements
        self.colors = {
            "person_bbox": (0, 255, 0),      # Green
            "confirmed_track": (0, 0, 255),  # Red
            "tentative_track": (255, 255, 0), # Yellow
            "line": config.COUNTING_LINE["color"],
            "text_bg": (0, 0, 0, 180),      # Semi-transparent black
            "overlay_bg": (0, 0, 0, 200)    # Dark overlay background
        }
        
        print("[INFO] Overlay Renderer initialized")

    @staticmethod
    def normalize_gender(gender_raw):
        if not gender_raw:
            return "Unknown"
        g = str(gender_raw).strip().lower()
        if g in {"male", "man", "m", "ชาย"}:
            return "Male"
        if g in {"female", "woman", "f", "หญิง"}:
            return "Female"
        return "Unknown"
        
    def map_age_group(self, age: int) -> str:
        """
        แปลงอายุเป็นกลุ่มอายุ
        
        Args:
            age: อายุเป็นตัวเลข
            
        Returns:
            กลุ่มอายุ: Child, Teen, Adult, Middle-age, Elderly
        """
        if age is None:
            return "Unknown"
            
        if age <= 12:
            return "Child"
        elif age <= 19:
            return "Teen"
        elif age <= 39:
            return "Adult"
        elif age <= 59:
            return "Middle-age"
        else:
            return "Elderly"
            
    def map_race_region(self, race: str) -> str:
        """
        แปลงเชื้อชาติเป็นภูมิภาค
        
        Args:
            race: เชื้อชาติจาก DeepFace
            
        Returns:
            ภูมิภาค: Asian หรือ Europe
        """
        if not race:
            return "Unknown"
            
        race_lower = race.lower()
        
        # Asian group
        if any(keyword in race_lower for keyword in ['asian', 'indian', 'middle eastern', 'latino']):
            return "Asian"
        # European group  
        elif any(keyword in race_lower for keyword in ['white', 'european']):
            return "Europe"
        # African group -> map to Asian for security classification
        elif any(keyword in race_lower for keyword in ['black', 'african']):
            return "Asian"  # Map to Asian region for security purposes
        else:
            return "Asian"  # Default to Asian
        
    def map_age_group(self, age: int) -> str:
        """
        แปลงอายุเป็นกลุ่มอายุ
        
        Args:
            age: อายุเป็นตัวเลข
            
        Returns:
            กลุ่มอายุ: Child, Teen, Adult, Middle-age, Elderly
        """
        if age is None:
            return "Unknown"
            
        if age <= 12:
            return "Child"
        elif age <= 19:
            return "Teen"
        elif age <= 39:
            return "Adult"
        elif age <= 59:
            return "Middle-age"
        else:
            return "Elderly"
            
    def map_race_region(self, race: str) -> str:
        """
        แปลงเชื้อชาติเป็นภูมิภาค
        
        Args:
            race: เชื้อชาติจาก DeepFace
            
        Returns:
            ภูมิภาค: Asian หรือ Europe
        """
        if not race:
            return "Unknown"
            
        race_lower = race.lower()
        
        # Asian group
        if any(keyword in race_lower for keyword in ['asian', 'indian', 'middle eastern', 'latino']):
            return "Asian"
        # European group  
        elif any(keyword in race_lower for keyword in ['white', 'european']):
            return "Europe"
        # African group -> map to Asian for security classification
        elif any(keyword in race_lower for keyword in ['black', 'african']):
            return "Asian"  # Map to Asian region for security purposes
        else:
            return "Asian"  # Default to Asian
        
    def render(self, frame: np.ndarray, tracks: List, counter_stats: Dict, fps: float) -> np.ndarray:
        """
        Render complete overlay on frame
        
        Args:
            frame: Input video frame
            tracks: List of active tracks
            counter_stats: Statistics from PeopleCounters
            fps: Current FPS
            
        Returns:
            Frame with rendered overlay
        """
        # Create working copy
        display_frame = frame.copy()
        
        # Resize frame if needed
        if display_frame.shape[:2] != (self.config.DISPLAY_HEIGHT, self.config.DISPLAY_WIDTH):
            display_frame = cv2.resize(display_frame, (self.config.DISPLAY_WIDTH, self.config.DISPLAY_HEIGHT))
        
        # Draw counting line
        display_frame = self.draw_counting_line(display_frame)
        
        # Draw track bounding boxes
        display_frame = self.draw_tracks(display_frame, tracks)
        
        # Draw center points for debugging line crossing
        if getattr(self.config, 'SHOW_TRACK_CENTER', False):
            try:
                for t in tracks:
                    if hasattr(t, 'get_center'):
                        cx, cy = t.get_center()
                        cv2.circle(display_frame, (int(cx), int(cy)), 3, (0, 255, 255), -1)
            except Exception:
                pass
        
        # Draw main overlay (top-left)
        display_frame = self.draw_main_overlay(display_frame, counter_stats, tracks)
        
        # Draw FPS (bottom-right)
        display_frame = self.draw_fps(display_frame, fps)
        
        # Draw system status (top-right)
        display_frame = self.draw_system_status(display_frame)
        
        return display_frame
        
    def draw_tracks(self, frame: np.ndarray, tracks: List) -> np.ndarray:
        """Draw enhanced bounding boxes and labels with quality indicators"""
        # Compute scale factors from model/input resolution to current display frame
        height, width = frame.shape[:2]
        scale_x = width / self.config.INPUT_WIDTH
        scale_y = height / self.config.INPUT_HEIGHT

        for track in tracks:
            if track.is_deleted():
                continue
            # Only show confirmed tracks
            if not (hasattr(track, 'is_confirmed') and track.is_confirmed()):
                continue

            # Get bounding box
            x1, y1, x2, y2 = [float(coord) for coord in track.to_tlbr()]
            
            # Check if should show face bbox only
            show_face_only = getattr(self.config, 'SHOW_FACE_BBOX_ONLY', False)
            face_bbox = None
            
            if show_face_only and hasattr(track, 'demographics') and track.demographics:
                face_bbox = track.demographics.get('face_bbox')
            
            # If face bbox available and show_face_only is True, use face bbox
            if show_face_only and face_bbox:
                # face_bbox is relative to face_region (top 60% of person ROI)
                fx1, fy1, fx2, fy2 = face_bbox
                # คำนวณ offset จาก person bbox
                person_h = y2 - y1
                # face_region คือส่วนบน 60% ของ person ROI
                # ดังนั้น face_bbox ใน face_region ต้องบวก offset ของ person bbox
                new_x1 = x1 + fx1
                new_y1 = y1 + fy1  # fy1 relative to top of face_region which starts at y1
                new_x2 = x1 + fx2
                new_y2 = y1 + fy2
                
                # ใช้ค่าใหม่
                x1, y1, x2, y2 = new_x1, new_y1, new_x2, new_y2
            elif show_face_only:
                # ถ้าไม่มี face_bbox ให้ประมาณจากส่วนบน 30% ของ person bbox
                person_h = y2 - y1
                person_w = x2 - x1
                # เอาส่วนบน 30% และตรงกลาง 60%
                margin_x = person_w * 0.2
                x1 = x1 + margin_x
                x2 = x2 - margin_x
                y2 = y1 + (person_h * 0.35)  # เอาแค่ส่วนบน 35%

            # Scale bbox from input resolution to display resolution
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            # Clip to frame bounds
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))
            # ตรวจสอบความถูกต้องของ bounding box
            if x2 <= x1 or y2 <= y1:
                continue  # ข้าม track ที่มี bbox ไม่ถูกต้อง

            # Color for confirmed track
            color = self.colors["confirmed_track"]
            thickness = 2

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # เพิ่มมุม 4 มุมเพื่อเน้นการแสดงผล
            corner_size = 15
            corner_thickness = thickness + 1
            # มุมบนซ้าย
            cv2.line(frame, (x1, y1), (x1 + corner_size, y1), color, corner_thickness)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_size), color, corner_thickness)
            # มุมบนขวา
            cv2.line(frame, (x2, y1), (x2 - corner_size, y1), color, corner_thickness)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_size), color, corner_thickness)
            # มุมล่างซ้าย
            cv2.line(frame, (x1, y2), (x1 + corner_size, y2), color, corner_thickness)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_size), color, corner_thickness)
            # มุมล่างขวา
            cv2.line(frame, (x2, y2), (x2 - corner_size, y2), color, corner_thickness)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_size), color, corner_thickness)

            # Enhanced trajectory drawing
            if hasattr(track, 'smooth_trajectory') and len(track.smooth_trajectory) > 1:
                points = np.array(track.smooth_trajectory, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], False, color, max(1, thickness-1))
                for i, point in enumerate(track.smooth_trajectory[-5:]):
                    alpha = (i + 1) / 5.0
                    point_color = tuple(int(c * alpha) for c in color)
                    cv2.circle(frame, tuple(map(int, point)), 2, point_color, -1)

            # Prepare label
            label_parts = []
            # ID only (no quality score)
            label_parts.append(f"ID:{track.track_id}")
            
            if hasattr(track, 'demographics') and track.demographics:
                demographics = track.demographics
                gender = self.normalize_gender(demographics.get('gender', 'Unknown'))
                gender_conf = demographics.get('gender_confidence', 0)
                
                # แสดงเพศด้วยตัวย่อ F, M, U พร้อม confidence %
                if gender == 'Female':
                    gender_abbr = 'F'
                elif gender == 'Male':
                    gender_abbr = 'M'
                else:
                    gender_abbr = 'U'
                
                # แสดง confidence % ถ้ามี
                if gender_conf > 0 and gender != 'Unknown':
                    label_parts.append(f"{gender_abbr} {int(gender_conf*100)}%")
                else:
                    label_parts.append(gender_abbr)
                
                # แสดงกลุ่มอายุ หรือ U
                age = demographics.get('age')
                if age is not None:
                    age_group = self.map_age_group(age)
                    label_parts.append(age_group)
                else:
                    label_parts.append('U')
                
                # แสดงภูมิภาค หรือ U
                if self.config.ETHNICITY_ENABLED and demographics.get('ethnicity'):
                    race = demographics['ethnicity']
                    region = self.map_race_region(race)
                    label_parts.append(region)
                else:
                    label_parts.append('U')
            else:
                # ไม่มีข้อมูล demographics แสดง U ทั้งหมด
                label_parts.extend(['U', 'U', 'U'])
            label_text = " | ".join(label_parts)
            label_size = cv2.getTextSize(label_text, self.font, self.font_scale, self.font_thickness)[0]
            label_x = max(0, min(x1, width - label_size[0] - 10))
            label_y = max(label_size[1] + 10, y1)
            label_bg_start = (label_x, label_y - label_size[1] - 10)
            label_bg_end = (label_x + label_size[0] + 10, label_y)
            overlay = frame.copy()
            cv2.rectangle(overlay, label_bg_start, label_bg_end, self.bg_color, -1)
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            cv2.putText(
                frame,
                label_text,
                (label_x + 5, label_y - 5),
                self.font,
                self.font_scale,
                self.text_color,
                self.font_thickness
            )
        return frame
        
    def draw_main_overlay(self, frame: np.ndarray, stats: Dict, tracks=None) -> np.ndarray:
        """Draw main statistics overlay (top-left corner)"""
        # Overlay position and size
        overlay_x = 10
        overlay_y = 10
        line_height = 25

        # Prepare simplified overlay text per user request
        in_m = stats.get('in_count', {}).get('Male', 0)
        in_f = stats.get('in_count', {}).get('Female', 0)
        in_u = stats.get('in_count', {}).get('Unknown', 0)
        out_m = stats.get('out_count', {}).get('Male', 0)
        out_f = stats.get('out_count', {}).get('Female', 0)
        out_u = stats.get('out_count', {}).get('Unknown', 0)
        # Prefer live tracked count (confirmed & not deleted) when tracks provided
        tracked_count = stats.get('current_occupancy', 0)
        try:
            if tracks:
                tc = 0
                for t in tracks:
                    if hasattr(t, 'is_deleted') and t.is_deleted():
                        continue
                    if hasattr(t, 'is_confirmed'):
                        if t.is_confirmed():
                            tc += 1
                    else:
                        tc += 1
                tracked_count = tc
        except Exception:
            tracked_count = stats.get('current_occupancy', 0)

        # Compute Asia / Europe counts from ethnicity_count keys (best-effort mapping)
        asia_count = 0
        europe_count = 0
        for k, v in stats.get('ethnicity_count', {}).items():
            key = str(k).lower()
            if any(x in key for x in ['asian', 'india', 'indian', 'south asian', 'east asian', 'southeast']):
                asia_count += v
            elif any(x in key for x in ['white', 'europe', 'european']):
                europe_count += v
            else:
                asia_count += v

        # Final overlay lines (only what's requested)
        overlay_lines = [
            f"IN  M:{in_m} F:{in_f} U:{in_u}",
            f"OUT M:{out_m} F:{out_f} U:{out_u}",
            "",
            f"person : {tracked_count}",
            f"Asia : {asia_count}  Europe : {europe_count}",
        ]

        # Calculate overlay background size
        max_text_width = 0
        for line in overlay_lines:
            text_size = cv2.getTextSize(line, self.font, self.font_scale, self.font_thickness)[0]
            max_text_width = max(max_text_width, text_size[0])

        overlay_width = max_text_width + 20
        overlay_height = len(overlay_lines) * line_height + 20

        # Draw overlay background if enabled; otherwise draw text with outline
        current_y = overlay_y + 20
        if getattr(self.config, 'SHOW_OVERLAY_BACKGROUND', True):
            overlay_bg = frame.copy()
            cv2.rectangle(
                overlay_bg,
                (overlay_x, overlay_y),
                (overlay_x + overlay_width, overlay_y + overlay_height),
                self.bg_color,
                -1
            )
            frame = cv2.addWeighted(frame, 0.7, overlay_bg, 0.3, 0)
            # Draw border
            cv2.rectangle(
                frame,
                (overlay_x, overlay_y),
                (overlay_x + overlay_width, overlay_y + overlay_height),
                self.text_color,
                1
            )
            # Draw text lines normally
            for line in overlay_lines:
                if line.strip():
                    cv2.putText(
                        frame,
                        line,
                        (overlay_x + 10, current_y),
                        self.font,
                        self.font_scale,
                        self.text_color,
                        self.font_thickness
                    )
                current_y += line_height
        else:
            # Draw text with a thin dark outline for readability on video
            outline_color = (0, 0, 0)
            outline_thickness = max(1, self.font_thickness)
            for line in overlay_lines:
                if line.strip():
                    # Draw outline (thicker)
                    cv2.putText(
                        frame,
                        line,
                        (overlay_x + 10, current_y),
                        self.font,
                        self.font_scale,
                        outline_color,
                        outline_thickness + 2
                    )
                    # Draw main text
                    cv2.putText(
                        frame,
                        line,
                        (overlay_x + 10, current_y),
                        self.font,
                        self.font_scale,
                        self.text_color,
                        self.font_thickness
                    )
                current_y += line_height

        return frame
        
    def draw_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """Draw FPS counter (bottom-right corner)"""
        height, width = frame.shape[:2]
        
        fps_text = f"FPS: {fps:.1f}"
        text_size = cv2.getTextSize(fps_text, self.font, self.font_scale, self.font_thickness)[0]
        
        # Position at bottom-right
        text_x = width - text_size[0] - 20
        text_y = height - 20
        
        # Draw background
        bg_start = (text_x - 10, text_y - text_size[1] - 10)
        bg_end = (text_x + text_size[0] + 10, text_y + 10)
        
        overlay_bg = frame.copy()
        cv2.rectangle(overlay_bg, bg_start, bg_end, self.bg_color, -1)
        frame = cv2.addWeighted(frame, 0.7, overlay_bg, 0.3, 0)
        
        # Choose color based on FPS
        fps_color = self.text_color
        if fps < 10:
            fps_color = (0, 0, 255)  # Red for low FPS
        elif fps < 20:
            fps_color = (0, 255, 255)  # Yellow for medium FPS
        else:
            fps_color = (0, 255, 0)  # Green for good FPS
            
        # Draw FPS text
        cv2.putText(
            frame,
            fps_text,
            (text_x, text_y),
            self.font,
            self.font_scale,
            fps_color,
            self.font_thickness
        )
        
        return frame
        
    def draw_system_status(self, frame: np.ndarray) -> np.ndarray:
        """Draw system status indicators (top-right corner)"""
        height, width = frame.shape[:2]
        
        # Status indicators
        status_lines = [
            f"RTSP: {'[OK] CONNECTED' if True else '[ERROR] DISCONNECTED'}",  # TODO: Get actual RTSP status
            f"MQTT: {'[OK] CONNECTED' if True else '[ERROR] DISCONNECTED'}",  # TODO: Get actual MQTT status
        ]
        
        if self.config.ETHNICITY_ENABLED:
            status_lines.append("[INFO] ETHNICITY: ENABLED")
        else:
            status_lines.append("[INFO] ETHNICITY: DISABLED")
            
        # Calculate position
        max_text_width = 0
        for line in status_lines:
            # Remove emoji for width calculation  
            clean_line = ''.join(c for c in line if ord(c) < 128)
            text_size = cv2.getTextSize(clean_line, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            max_text_width = max(max_text_width, text_size[0])
            
        status_x = width - max_text_width - 20
        status_y = 30
        line_height = 20
        
        # Draw each status line
        for i, line in enumerate(status_lines):
            # Remove emoji for OpenCV (it doesn't handle Unicode well)
            clean_line = ''.join(c for c in line if ord(c) < 128)
            
            cv2.putText(
                frame,
                clean_line,
                (status_x, status_y + i * line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                self.text_color,
                1
            )
            
        return frame
        
    def draw_counting_line(self, frame: np.ndarray) -> np.ndarray:
        """Draw the counting line"""
        # Scale line coordinates to display resolution
        height, width = frame.shape[:2]
        # If config requests center_vertical mode, compute centered vertical line
        if getattr(self.config, 'COUNTING_LINE_MODE', 'fixed') == 'center_vertical':
            margin_top = getattr(self.config, 'COUNTING_LINE_MARGIN_TOP', 100)
            margin_bottom = getattr(self.config, 'COUNTING_LINE_MARGIN_BOTTOM', 100)
            cx = width // 2
            start_x, start_y = cx, int(margin_top)
            end_x, end_y = cx, int(max(margin_top + 1, height - margin_bottom))
        else:
            scale_x = width / self.config.INPUT_WIDTH
            scale_y = height / self.config.INPUT_HEIGHT
            start_x = int(self.config.COUNTING_LINE["start"][0] * scale_x)
            start_y = int(self.config.COUNTING_LINE["start"][1] * scale_y)
            end_x = int(self.config.COUNTING_LINE["end"][0] * scale_x)
            end_y = int(self.config.COUNTING_LINE["end"][1] * scale_y)
        
        # Draw line
        cv2.line(
            frame,
            (start_x, start_y),
            (end_x, end_y),
            self.colors["line"],
            self.config.COUNTING_LINE["thickness"]
        )
        
        # Draw line label
        mid_x = (start_x + end_x) // 2
        mid_y = (start_y + end_y) // 2
        
        cv2.putText(
            frame,
            "COUNT LINE",
            (mid_x - 40, mid_y - 10),
            self.font,
            0.4,
            self.colors["line"],
            1
        )
        
        return frame
        
    def draw_debug_info(self, frame: np.ndarray, debug_data: Dict) -> np.ndarray:
        """Draw debug information overlay"""
        debug_lines = []
        
        # Add debug information
        if "detection_time" in debug_data:
            debug_lines.append(f"Detection: {debug_data['detection_time']:.3f}s")
        if "tracking_time" in debug_data:
            debug_lines.append(f"Tracking: {debug_data['tracking_time']:.3f}s")  
        if "face_analysis_time" in debug_data:
            debug_lines.append(f"Face Analysis: {debug_data['face_analysis_time']:.3f}s")
        if "total_time" in debug_data:
            debug_lines.append(f"Total: {debug_data['total_time']:.3f}s")
            
        # Draw debug overlay (bottom-left)
        if debug_lines:
            debug_y = frame.shape[0] - len(debug_lines) * 25 - 20
            
            for i, line in enumerate(debug_lines):
                cv2.putText(
                    frame,
                    line,
                    (10, debug_y + i * 25),
                    self.font,
                    0.4,
                    (255, 255, 255),
                    1
                )
                
        return frame
        
    def create_summary_frame(self, stats: Dict, width: int = 800, height: int = 600) -> np.ndarray:
        """Create a summary statistics frame"""
        # Create black frame
        summary_frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Title
        title = "PEOPLE TRACKING SUMMARY"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        title_x = (width - title_size[0]) // 2
        
        cv2.putText(summary_frame, title, (title_x, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Statistics sections
        y_pos = 100
        line_height = 30
        
        sections = [
            ("CURRENT COUNTS", [
                f"IN:  Male: {stats['in_count']['Male']}, Female: {stats['in_count']['Female']}",
                f"OUT: Male: {stats['out_count']['Male']}, Female: {stats['out_count']['Female']}",
                f"Current Occupancy: {stats['current_occupancy']}"
            ]),
            ("TOTALS", [
                f"Total IN: {stats['total_in']}",
                f"Total OUT: {stats['total_out']}",
                f"Net Change: {stats['total_in'] - stats['total_out']}"
            ])
        ]
        
        # Add group statistics if available
        if stats.get('group_count'):
            group_lines = [f"{group}: {count}" for group, count in stats['group_count'].items()]
            sections.append(("GROUPS", group_lines))
            
        # Draw sections
        for section_title, section_lines in sections:
            # Section title
            cv2.putText(summary_frame, section_title, (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_pos += line_height
            
            # Section content
            for line in section_lines:
                cv2.putText(summary_frame, f"  {line}", (70, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_pos += line_height - 5
                
            y_pos += 10  # Extra space between sections
            
        return summary_frame
        
    def _get_current_demographics(self, tracks) -> List[Dict]:
        """Extract demographics from current tracks in frame"""
        demographics = []
        for track in tracks:
            if hasattr(track, 'demographics') and track.demographics:
                demo_data = track.demographics.copy()
                demo_data['id'] = track.track_id
                demographics.append(demo_data)
        return demographics
        
    def update_config(self, new_config):
        """Update configuration dynamically"""
        self.config = new_config
        self.display_config = new_config.get_display_config()
        print("[INFO] Overlay renderer config updated")