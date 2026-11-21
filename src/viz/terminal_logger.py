"""
Terminal Logger - Real-time Event Logging
แสดงผล log แบบเรียลไทม์ใน terminal สำหรับ People Tracking events
"""

import time
import threading
from datetime import datetime
from typing import Dict, Any, List
from collections import deque

class TerminalLogger:
    def __init__(self, max_lines: int = 50):
        self.max_lines = max_lines
        self.log_buffer = deque(maxlen=max_lines)
        self.lock = threading.Lock()
        # Statistics
        self.total_events = 0
        self.events_by_direction = {"IN": 0, "OUT": 0, "TRACK": 0}
        self.events_by_gender = {"Male": 0, "Female": 0, "Unknown": 0}
        print("[INFO] Terminal Logger initialized")
    
    @staticmethod
    def _normalize_gender(gender_raw) -> str:
        """Map various gender strings to Male/Female/Unknown to keep keys consistent."""
        if not gender_raw:
            return "Unknown"
        g = str(gender_raw).strip().lower()
        if g in {"male", "man", "m", "ชาย"}:
            return "Male"
        if g in {"female", "woman", "f", "หญิง"}:
            return "Female"
        return "Unknown"
        
    def log_event(self, crossing_event):
        """
        Log a crossing event to terminal
        
        Args:
            crossing_event: CrossingEvent object
        """
        with self.lock:
            self.total_events += 1
            
            # Extract event data
            track_id = crossing_event.track_id
            direction = crossing_event.direction
            demographics = crossing_event.demographics
            timestamp = crossing_event.timestamp
            
            # Update statistics
            self.events_by_direction[direction] += 1
            raw_gender = demographics.get("gender", "Unknown")
            gender = self._normalize_gender(raw_gender)
            self.events_by_gender[gender] += 1
            
            # Format timestamp
            time_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
            
            # Build log message
            # Ensure age and track_id are formatted safely
            age_val = demographics.get('age', 'N/A')
            age_str = str(age_val) if age_val is not None else 'N/A'
            track_id_str = str(int(track_id)) if isinstance(track_id, (int, float, str)) else 'N/A'
            log_parts = [
                f"ID:{track_id_str:>3s}",
                f"| {gender:6s}",
                f"| Age:{age_str}",
            ]
            
            # Add ethnicity if available and enabled
            ethnicity = demographics.get('ethnicity')
            group = demographics.get('group')
            ethnicity_confidence = demographics.get('ethnicity_confidence', 0)
            
            if ethnicity and ethnicity_confidence >= 0.75:  # Only show if confident
                log_parts.append(f"| {ethnicity:5s}")
                if group:
                    log_parts.append(f"| {group}")
            else:
                log_parts.append(f"| {'N/A':5s}")
                
            # Add direction and status
            status = "entered" if direction == "IN" else "left"
            log_parts.extend([
                f"| {status:7s}",
                f"| {time_str}"
            ])
            
            # Create final log message
            log_message = " ".join(log_parts)
            
            # Add to buffer
            self.log_buffer.append({
                "timestamp": timestamp,
                "message": log_message,
                "event_data": {
                    "track_id": track_id,
                    "direction": direction,
                    "demographics": demographics.copy()
                }
            })
            
            # Print to console
            print(f"[INFO] {log_message}")
            
    def log_system_event(self, event_type: str, message: str, level: str = "INFO"):
        """
        Log system events (non-crossing events)
        
        Args:
            event_type: Type of system event
            message: Event message
            level: Log level (DEBUG, INFO, WARNING, ERROR)
        """
        with self.lock:
            timestamp = time.time()
            time_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
            
            # Format system message
            level_icon = {
                "DEBUG": "[DEBUG]",
                "INFO": "[INFO]",
                "WARNING": "[WARNING]", 
                "ERROR": "[ERROR]"
            }.get(level, "[INFO]")
            
            log_message = f"{level_icon} [{level}] {event_type}: {message} | {time_str}"
            
            # Add to buffer
            self.log_buffer.append({
                "timestamp": timestamp,
                "message": log_message,
                "event_data": {
                    "type": "system",
                    "event_type": event_type,
                    "level": level,
                    "message": message
                }
            })
            
            # Print to console
            print(log_message)
            
    def log_statistics_update(self, stats: Dict[str, Any]):
        """Log periodic statistics updates"""
        total_in = stats.get("total_in", 0)
        total_out = stats.get("total_out", 0)
        occupancy = stats.get("current_occupancy", 0)
        
        stats_message = f"[INFO] Stats Update: IN:{total_in} OUT:{total_out} Current:{occupancy}"
        
        self.log_system_event("STATS", stats_message, "INFO")
        
    def get_recent_logs(self, count: int = 10) -> List[Dict]:
        """Get recent log entries"""
        with self.lock:
            return list(self.log_buffer)[-count:]
            
    def get_all_logs(self) -> List[Dict]:
        """Get all log entries in buffer"""
        with self.lock:
            return list(self.log_buffer)
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get logger statistics"""
        with self.lock:
            return {
                "total_events": self.total_events,
                "events_by_direction": self.events_by_direction.copy(),
                "events_by_gender": self.events_by_gender.copy(),
                "buffer_size": len(self.log_buffer),
                "max_buffer_size": self.max_lines
            }
            
    def print_summary(self):
        """Print summary of all events"""
        with self.lock:
            print("\n" + "="*60)
            print("[INFO] TERMINAL LOGGER SUMMARY")
            print("="*60)
            print(f"Total Events Logged: {self.total_events}")
            print(f"Buffer Size: {len(self.log_buffer)}/{self.max_lines}")
            print()
            
            # Direction breakdown
            print("[INFO] Direction Breakdown:")
            for direction, count in self.events_by_direction.items():
                percentage = (count / max(self.total_events, 1)) * 100
                print(f"  {direction:4s}: {count:3d} ({percentage:5.1f}%)")
            print()
            
            # Gender breakdown
            print("[INFO] Gender Breakdown:")
            for gender, count in self.events_by_gender.items():
                percentage = (count / max(self.total_events, 1)) * 100
                print(f"  {gender:7s}: {count:3d} ({percentage:5.1f}%)")
            print("="*60)
            
    def export_logs(self, format_type: str = "text") -> str:
        """
        Export logs in specified format
        
        Args:
            format_type: "text", "csv", or "json"
            
        Returns:
            Formatted log data as string
        """
        with self.lock:
            if format_type == "text":
                return self._export_as_text()
            elif format_type == "csv":
                return self._export_as_csv()
            elif format_type == "json":
                return self._export_as_json()
            else:
                return self._export_as_text()
                
    def _export_as_text(self) -> str:
        """Export logs as plain text"""
        lines = ["PEOPLE TRACKING LOG EXPORT", "=" * 50, ""]
        
        for log_entry in self.log_buffer:
            lines.append(log_entry["message"])
            
        lines.extend(["", "=" * 50, "SUMMARY"])
        
        # Add summary statistics
        stats = self.get_statistics()
        lines.append(f"Total Events: {stats['total_events']}")
        lines.append(f"IN: {stats['events_by_direction']['IN']}, OUT: {stats['events_by_direction']['OUT']}")
        
        return "\n".join(lines)
        
    def _export_as_csv(self) -> str:
        """Export logs as CSV"""
        lines = ["timestamp,track_id,direction,gender,age,ethnicity,group,message"]
        
        for log_entry in self.log_buffer:
            event_data = log_entry.get("event_data", {})
            if event_data.get("type") == "system":
                continue  # Skip system events in CSV
                
            demographics = event_data.get("demographics", {})
            
            csv_line = ",".join([
                str(log_entry["timestamp"]),
                str(event_data.get("track_id", "")),
                event_data.get("direction", ""),
                demographics.get("gender", ""),
                str(demographics.get("age", "")),
                demographics.get("ethnicity", ""),
                demographics.get("group", ""),
                f'"{log_entry["message"]}"'
            ])
            
            lines.append(csv_line)
            
        return "\n".join(lines)
        
    def _export_as_json(self) -> str:
        """Export logs as JSON"""
        import json
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "statistics": self.get_statistics(),
            "logs": list(self.log_buffer)
        }
        
        return json.dumps(export_data, indent=2, ensure_ascii=False)
        
    def clear_logs(self):
        """Clear all logs and reset statistics"""
        with self.lock:
            self.log_buffer.clear()
            self.total_events = 0
            self.events_by_direction = {"IN": 0, "OUT": 0}
            self.events_by_gender = {"Male": 0, "Female": 0, "Unknown": 0}
            
        print("[INFO] Terminal logs cleared")
        
    def save_logs_to_file(self, filename: str = None, format_type: str = "text"):
        """Save logs to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            extension = {"text": "txt", "csv": "csv", "json": "json"}.get(format_type, "txt")
            filename = f"people_tracking_log_{timestamp}.{extension}"
            
        try:
            log_content = self.export_logs(format_type)
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(log_content)
                
            print(f"[OK] Logs saved to: {filename}")
            return filename
            
        except Exception as e:
            print(f"[ERROR] Failed to save logs: {e}")
            return None
            
    def start_periodic_stats(self, interval: int = 60):
        """Start periodic statistics logging"""
        def stats_loop():
            while True:
                time.sleep(interval)
                stats = self.get_statistics()
                self.log_system_event("PERIODIC_STATS", 
                                    f"Events: {stats['total_events']}, Buffer: {stats['buffer_size']}")
                
        stats_thread = threading.Thread(target=stats_loop, daemon=True)
        stats_thread.start()
        print(f"[INFO] Periodic stats logging started (every {interval}s)")
        
    def __str__(self) -> str:
        return f"TerminalLogger(events={self.total_events}, buffer={len(self.log_buffer)})"
        
    def __repr__(self) -> str:
        return f"TerminalLogger(total_events={self.total_events}, buffer_size={len(self.log_buffer)})"