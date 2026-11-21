"""
People Counters - IN/OUT Statistics Management
จัดการสถิติการนับจำนวนคน แยกตาม Gender และ Group
"""

import time
from typing import Dict, Any, List
from collections import defaultdict, Counter
from datetime import datetime, timedelta

class PeopleCounters:
    def __init__(self):
        # Main counters
        self.in_count = {"Male": 0, "Female": 0, "Unknown": 0}
        self.out_count = {"Male": 0, "Female": 0, "Unknown": 0}
        
        # Group counters (for ethnicity groups)
        self.group_count = defaultdict(int)  # GroupA, GroupB, etc.
        
        # Ethnicity counters (for raw ethnicity data)
        self.ethnicity_count = defaultdict(int)  # Asian, White, Black, etc.
        
        # Total counters
        self.total_in = 0
        self.total_out = 0
        self.current_occupancy = 0
        
        # Detailed history
        self.crossing_history = []  # List of all crossing events
        self.hourly_stats = defaultdict(lambda: {"in": 0, "out": 0})  # Hour -> counts
        
        # Session statistics
        self.session_start_time = time.time()
        self.last_reset_time = time.time()
        
        print("[INFO] People Counters initialized")
        
    def _normalize_gender(self, gender_raw) -> str:
        """Map various gender strings to Male/Female/Unknown to keep keys consistent."""
        if not gender_raw:
            return "Unknown"
        g = str(gender_raw).strip().lower()
        male_aliases = {"male", "man", "m", "ชาย"}
        female_aliases = {"female", "woman", "f", "หญิง"}
        if g in male_aliases:
            return "Male"
        if g in female_aliases:
            return "Female"
        return "Unknown"

    def update_count(self, crossing_event):
        """
        Update counters based on crossing event
        
        Args:
            crossing_event: CrossingEvent object
        """
        demographics = crossing_event.demographics or {}
        direction = crossing_event.direction
        
        # Extract demographics
        gender = self._normalize_gender(demographics.get("gender"))
        ethnicity = demographics.get("ethnicity")
        group = demographics.get("group")
        
        # Update main counters
        if direction == "IN":
            self.in_count[gender] = self.in_count.get(gender, 0) + 1
            self.total_in += 1
            self.current_occupancy += 1
        elif direction == "OUT":
            self.out_count[gender] = self.out_count.get(gender, 0) + 1  
            self.total_out += 1
            self.current_occupancy = max(0, self.current_occupancy - 1)
            
        # Update group counters
        if group:
            if direction == "IN":
                self.group_count[group] += 1
            # Note: We don't decrement group count on OUT to maintain total groups seen
            
        # Update ethnicity counters
        if ethnicity:
            if direction == "IN":
                self.ethnicity_count[ethnicity] += 1
            
        # Add to crossing history
        crossing_record = {
            "timestamp": crossing_event.timestamp,
            "track_id": crossing_event.track_id,
            "direction": direction,
            "gender": gender,
            "ethnicity": ethnicity,
            "group": group,
            "demographics": demographics.copy()
        }
        self.crossing_history.append(crossing_record)
        
        # Update hourly statistics
        hour_key = datetime.fromtimestamp(crossing_event.timestamp).strftime("%Y-%m-%d %H:00")
        if direction == "IN":
            self.hourly_stats[hour_key]["in"] += 1
        else:
            self.hourly_stats[hour_key]["out"] += 1
            
        # Keep history manageable (last 1000 events)
        if len(self.crossing_history) > 1000:
            self.crossing_history = self.crossing_history[-1000:]
            
        print(f"[INFO] Count updated: {direction} {gender} | Total IN: {self.total_in}, OUT: {self.total_out}")
        
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        return {
            "in_count": self.in_count.copy(),
            "out_count": self.out_count.copy(),
            "group_count": dict(self.group_count),
            "ethnicity_count": dict(self.ethnicity_count),
            "total_in": self.total_in,
            "total_out": self.total_out,
            "current_occupancy": self.current_occupancy,
            "session_duration": time.time() - self.session_start_time,
            # เพิ่มค่าแยกสำหรับ Dashboard
            "male_count": self.in_count.get("Male", 0) + self.out_count.get("Male", 0),
            "female_count": self.in_count.get("Female", 0) + self.out_count.get("Female", 0),
            "unknown_count": self.in_count.get("Unknown", 0) + self.out_count.get("Unknown", 0),
        }
        
    def get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed statistics including history"""
        stats = self.get_stats()
        
        # Add session info
        stats["session"] = {
            "start_time": datetime.fromtimestamp(self.session_start_time).isoformat(),
            "duration_hours": (time.time() - self.session_start_time) / 3600,
            "total_events": len(self.crossing_history)
        }
        
        # Add recent activity (last hour)
        recent_events = self.get_recent_events(hours=1)
        stats["recent_activity"] = {
            "last_hour_in": len([e for e in recent_events if e["direction"] == "IN"]),
            "last_hour_out": len([e for e in recent_events if e["direction"] == "OUT"]),
            "recent_groups": dict(Counter([e["group"] for e in recent_events if e["group"]]))
        }
        
        # Add peak hours
        stats["peak_hours"] = self.get_peak_hours()
        
        return stats
        
    def get_recent_events(self, hours: int = 1) -> List[Dict]:
        """Get crossing events from the last N hours"""
        cutoff_time = time.time() - (hours * 3600)
        return [event for event in self.crossing_history if event["timestamp"] >= cutoff_time]
        
    def get_peak_hours(self) -> Dict[str, Any]:
        """Calculate peak hours based on crossing activity"""
        if not self.hourly_stats:
            return {"peak_in_hour": None, "peak_out_hour": None, "peak_total_hour": None}
            
        # Find peak hours
        peak_in = max(self.hourly_stats.items(), key=lambda x: x[1]["in"])
        peak_out = max(self.hourly_stats.items(), key=lambda x: x[1]["out"])
        peak_total = max(self.hourly_stats.items(), key=lambda x: x[1]["in"] + x[1]["out"])
        
        return {
            "peak_in_hour": {"hour": peak_in[0], "count": peak_in[1]["in"]},
            "peak_out_hour": {"hour": peak_out[0], "count": peak_out[1]["out"]}, 
            "peak_total_hour": {"hour": peak_total[0], "count": peak_total[1]["in"] + peak_total[1]["out"]}
        }
        
    def get_demographic_breakdown(self) -> Dict[str, Any]:
        """Get detailed demographic breakdown"""
        # Gender breakdown
        total_gender_in = sum(self.in_count.values())
        total_gender_out = sum(self.out_count.values())
        
        gender_breakdown = {
            "in_percentages": {
                gender: (count / max(total_gender_in, 1)) * 100 
                for gender, count in self.in_count.items()
            },
            "out_percentages": {
                gender: (count / max(total_gender_out, 1)) * 100 
                for gender, count in self.out_count.items()
            }
        }
        
        # Group breakdown
        total_groups = sum(self.group_count.values())
        group_breakdown = {
            group: {
                "count": count,
                "percentage": (count / max(total_groups, 1)) * 100
            }
            for group, count in self.group_count.items()
        }
        
        return {
            "gender": gender_breakdown,
            "groups": group_breakdown,
            "total_classified": total_gender_in + total_gender_out,
            "classification_rate": {
                "gender": ((total_gender_in + total_gender_out - 
                          self.in_count.get("Unknown", 0) - self.out_count.get("Unknown", 0)) / 
                         max(total_gender_in + total_gender_out, 1)) * 100
            }
        }
        
    def get_hourly_report(self, hours: int = 24) -> Dict[str, Any]:
        """Get hourly activity report for last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        cutoff_str = cutoff_time.strftime("%Y-%m-%d %H:00")
        
        # Filter recent hours
        recent_hours = {
            hour: stats for hour, stats in self.hourly_stats.items() 
            if hour >= cutoff_str
        }
        
        if not recent_hours:
            return {"hours": {}, "summary": {"total_in": 0, "total_out": 0, "peak_hour": None}}
        
        # Calculate summary
        total_in = sum(stats["in"] for stats in recent_hours.values())
        total_out = sum(stats["out"] for stats in recent_hours.values())
        
        # Find peak hour
        peak_hour = max(recent_hours.items(), key=lambda x: x[1]["in"] + x[1]["out"])
        
        return {
            "hours": recent_hours,
            "summary": {
                "total_in": total_in,
                "total_out": total_out,
                "net_change": total_in - total_out,
                "peak_hour": {
                    "hour": peak_hour[0],
                    "in_count": peak_hour[1]["in"],
                    "out_count": peak_hour[1]["out"],
                    "total_count": peak_hour[1]["in"] + peak_hour[1]["out"]
                }
            }
        }
        
    def export_data(self, format_type: str = "dict") -> Any:
        """Export all counter data"""
        export_data = {
            "metadata": {
                "session_start": datetime.fromtimestamp(self.session_start_time).isoformat(),
                "export_time": datetime.now().isoformat(),
                "total_events": len(self.crossing_history)
            },
            "current_stats": self.get_stats(),
            "demographic_breakdown": self.get_demographic_breakdown(),
            "hourly_stats": dict(self.hourly_stats),
            "crossing_history": self.crossing_history.copy()
        }
        
        if format_type == "dict":
            return export_data
        elif format_type == "json":
            import json
            return json.dumps(export_data, indent=2, ensure_ascii=False)
        else:
            return export_data
            
    def reset_counters(self, reset_history: bool = False):
        """Reset all counters"""
        self.in_count = {"Male": 0, "Female": 0, "Unknown": 0}
        self.out_count = {"Male": 0, "Female": 0, "Unknown": 0}
        self.group_count = defaultdict(int)
        self.ethnicity_count = defaultdict(int)
        self.total_in = 0
        self.total_out = 0
        self.current_occupancy = 0
        self.last_reset_time = time.time()
        
        if reset_history:
            self.crossing_history.clear()
            self.hourly_stats.clear()
            self.session_start_time = time.time()
            
        print("[INFO] Counters reset")
        
    def reset_session(self):
        """Reset entire session (counters + history)"""
        self.reset_counters(reset_history=True)
        print("[INFO] Session reset")
        
    def get_occupancy_estimate(self) -> Dict[str, Any]:
        """Get current occupancy estimate"""
        # Simple IN - OUT calculation
        basic_occupancy = max(0, self.total_in - self.total_out)
        
        # More sophisticated estimate based on recent activity
        recent_events = self.get_recent_events(hours=2)  # Last 2 hours
        recent_in = len([e for e in recent_events if e["direction"] == "IN"])
        recent_out = len([e for e in recent_events if e["direction"] == "OUT"])
        
        return {
            "current_occupancy": self.current_occupancy,
            "basic_estimate": basic_occupancy,
            "recent_activity": {
                "in_last_2h": recent_in,
                "out_last_2h": recent_out,
                "net_change_2h": recent_in - recent_out
            },
            "confidence": "high" if len(recent_events) > 10 else "medium" if len(recent_events) > 3 else "low"
        }
        
    def __str__(self) -> str:
        """String representation of current stats"""
        stats = self.get_stats()
        return (f"IN: M:{stats['in_count']['Male']} F:{stats['in_count']['Female']} | "
                f"OUT: M:{stats['out_count']['Male']} F:{stats['out_count']['Female']} | "
                f"Occupancy: {stats['current_occupancy']}")
                
    def __repr__(self) -> str:
        return f"PeopleCounters(in={self.total_in}, out={self.total_out}, occupancy={self.current_occupancy})"
