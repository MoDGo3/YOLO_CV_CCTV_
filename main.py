# -*- coding: utf-8 -*-
"""
YOLO CV CCTV V4 - People Tracking + Demographics Analytics
High-level security project for government agencies
Support analysis: age, gender, ethnicity with IN/OUT counting
"""

import cv2
import sys
import threading
import time
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Fix encoding for Windows Command Prompt
if sys.platform.startswith('win'):
    try:
        os.system('mode con: cols=120 lines=40')
    except:
        pass

# File dialog for video selection
def select_video_file():
    """Open file dialog to select a video file"""
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        root = tk.Tk()
        root.withdraw()  # Hide main window
        root.attributes('-topmost', True)  # Bring dialog to front
        
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video Files", "*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm"),
                ("MP4 Files", "*.mp4"),
                ("AVI Files", "*.avi"),
                ("MKV Files", "*.mkv"),
                ("All Files", "*.*")
            ]
        )
        
        root.destroy()
        return file_path if file_path else None
        
    except Exception as e:
        print(f"[WARNING] File dialog error: {e}")
        print("\nEnter video file path manually:")
        print("Example: C:\\path\\to\\video.mp4")
        print("         or ./videos/video.mp4")
        
        while True:
            file_path = input("Enter file path (or press Enter to cancel): ").strip().strip('"\'')
            
            if not file_path:
                print("[INFO] Cancelled.")
                return None
                
            from pathlib import Path
            p = Path(file_path)
            if p.exists() and p.is_file():
                print(f"[OK] File found: {p.absolute()}")
                return str(p.absolute())
            else:
                print(f"[ERROR] File not found: {file_path}")
                retry = input("Try another path? (y/n): ").strip().lower()
                if retry != 'y':
                    return None


def check_dependencies():
    """Check and display available dependencies"""
    print("\n" + "="*50)
    print("[CHECK] System Dependencies")
    print("="*50)
    
    # Check deep_sort_realtime
    try:
        from deep_sort_realtime.deepsort_tracker import DeepSort
        print("[OK] deep_sort_realtime is available")
    except ImportError:
        print("[FAIL] deep_sort_realtime not available")
    
    # Check DeepFace
    try:
        from deepface import DeepFace
        print("[OK] DeepFace is available")
    except ImportError:
        print("[FAIL] DeepFace not available")
    
    # Check face_recognition
    try:
        import face_recognition
        print("[OK] face_recognition is available")
    except ImportError:
        print("[FAIL] face_recognition not available")
    
    print("="*50 + "\n")


def show_input_menu():
    """Display input source selection menu"""
    print("\n" + "="*50)
    print("[SELECT] Video Input Source")
    print("="*50)
    print("  [1] Video File (from computer)")
    print("  [2] RTSP Camera Stream")
    print("  [3] Webcam")
    print("  [4] Exit")
    print("="*50)
    
    while True:
        try:
            choice = input("Enter your choice (1-4): ").strip()
            
            if choice == '1':
                return 'video_file', None
            elif choice == '2':
                return 'rtsp', None
            elif choice == '3':
                # Ask for webcam index
                cam_input = input("   Webcam index (press Enter for 0): ").strip()
                cam_index = int(cam_input) if cam_input else 0
                return 'webcam', cam_index
            elif choice == '4':
                print("[INFO] Exiting program...")
                sys.exit(0)
            else:
                print("[WARNING] Please select 1-4")
        except ValueError:
            print("[WARNING] Please enter a number 1-4")
        except KeyboardInterrupt:
            print("\n[INFO] Exiting program...")
            sys.exit(0)

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from config import Config
from src.io.rtsp_handler import RTSPHandler
from src.io.video_file_handler import VideoFileHandler
from src.io.mqtt_client import MQTTClient
from src.detect.yolo_detector import YOLODetector
from src.track.deepsort_tracker import DeepSORTTracker
from src.track.yolov8_native_tracker import YOLOv8NativeTracker
from src.face.face_analyzer import FaceAnalyzer
from src.analytics.line_crossing import LineCrossing
from src.analytics.counters import PeopleCounters
from src.viz.overlay_renderer import OverlayRenderer
from src.viz.terminal_logger import TerminalLogger
from src.utils.fps_counter import FPSCounter

class CCTVSystem:
    def __init__(self, input_source=None, webcam_index=None):
        self.config = Config()
        self.running = False
        self.frame_count = 0
        self._last_stats_publish = 0.0
        self.current_tracked_count = 0
        self.headless_mode = False  # Auto-detect if GUI is available
        
        # Session statistics tracking
        self.session_start_time = datetime.now()
        self.session_id = self.session_start_time.strftime("%Y%m%d_%H%M%S")
        self.total_processed_frames = 0
        self.video_source_path = None  # Track video source for logging
        
        # Use provided input_source or fall back to config
        self.input_source = input_source if input_source else getattr(self.config, 'INPUT_SOURCE', 'rtsp').lower()
        self.video_handler = None
        
        if self.input_source == 'video_file':
            # Use video file input
            video_path = getattr(self.config, 'VIDEO_FILE_PATH', 'browse')
            video_loop = getattr(self.config, 'VIDEO_LOOP', True)
            
            # Always open file dialog for video_file from menu
            if input_source == 'video_file' or video_path.lower() in ['browse', 'select', 'dialog', '']:
                print("[INFO] Opening file browser...")
                video_path = select_video_file()
                if not video_path:
                    print("[ERROR] No file selected. Exiting.")
                    sys.exit(1)
            
            self.video_handler = VideoFileHandler(video_path, loop=video_loop)
            if hasattr(self.config, 'VIDEO_PLAYBACK_SPEED'):
                self.video_handler.set_playback_speed(self.config.VIDEO_PLAYBACK_SPEED)
            self.video_source_path = video_path  # Track for logging
            print(f"[OK] Using Video File: {video_path}")
        elif self.input_source == 'webcam':
            # Use webcam input - use provided index or config
            cam_index = webcam_index if webcam_index is not None else getattr(self.config, 'WEBCAM_INDEX', 0)
            self.video_handler = VideoFileHandler(str(cam_index), loop=False)
            print(f"[OK] Using Webcam: {cam_index}")
        else:
            # Default: use RTSP stream
            self.video_handler = RTSPHandler(self.config.RTSP_URL)
            print(f"[OK] Using RTSP: {self.config.RTSP_URL_MASKED}")
        
        # Keep rtsp_handler reference for compatibility
        self.rtsp_handler = self.video_handler
        
        self.mqtt_client = MQTTClient(self.config)
        self.mqtt_client.set_command_handler(self.handle_mqtt_command)
        self.yolo_detector = YOLODetector(self.config)
        
        # Initialize Enhanced DeepSORT Tracker
        self.use_native_tracking = False
        try:
            self.tracker = DeepSORTTracker(self.config)
            print("[OK] Using Enhanced DeepSORT Tracking")
        except Exception as e:
            print(f"[ERROR] DeepSORT tracking failed: {e}")
            # Fallback to YOLOv8 native if needed
            try:
                self.tracker = YOLOv8NativeTracker(self.config, tracker_type="botsort")
                self.use_native_tracking = True
                print("[INFO] Fallback to YOLOv8 Native Tracking")
            except Exception as e2:
                print(f"[ERROR] All tracking methods failed: {e2}")
                self.tracker = None
        
        self.face_analyzer = FaceAnalyzer(self.config)
        self.line_crossing = LineCrossing(self.config)
        self.counters = PeopleCounters()
        self.overlay_renderer = OverlayRenderer(self.config)
        self.terminal_logger = TerminalLogger()
        self.fps_counter = FPSCounter()
        
        print("[OK] CCTV System initialized")
        print(f"[INFO] RTSP: {self.config.RTSP_URL_MASKED}")
        print(f"[INFO] MQTT: {self.config.MQTT_BROKER}:{self.config.MQTT_PORT}")
        print(f"[INFO] Analytics: Age/Gender/Ethnicity Enabled")

    def handle_mqtt_command(self, command: Dict[str, Any], message=None):
        """Handle inbound control commands via MQTT."""
        try:
            action = str(command.get("action", command.get("type", ""))).lower()
        except Exception:
            action = ""
        if action != "reset":
            return

        scope = str(command.get("scope", command.get("target", "counters"))).lower()
        full_reset = scope in {"full", "session", "all"}
        source = command.get("source", "mqtt")
        self.reset_counters(full_reset=full_reset, source=source)

    def reset_counters(self, full_reset: bool = False, source: str = "manual"):
        """Reset counters/session and publish updated statistics."""
        try:
            if full_reset:
                self.counters.reset_session()
                print(f"[INFO] Session reset triggered by {source}")
            else:
                self.counters.reset_counters(reset_history=False)
                print(f"[INFO] Counters reset triggered by {source}")

            self.current_tracked_count = 0
            stats = self.counters.get_stats()
            stats["tracked_count"] = self.current_tracked_count
            self.mqtt_client.publish_statistics(stats)
            self.mqtt_client.publish_control(
                action="reset",
                scope="session" if full_reset else "counters",
                source=source,
            )
        except Exception as exc:
            print(f"[WARNING] Failed to reset counters ({source}): {exc}")

    def process_frame(self, frame):
        """Process single frame through the pipeline"""
        self.frame_count += 1
        
        # 1. Person Detection with Enhanced YOLOv8
        detections = self.yolo_detector.detect(frame)
        
        # 2. Enhanced Multi-Object Tracking with DeepSORT
        if self.tracker:
            if self.use_native_tracking:
                # YOLOv8 Native Tracking (fallback)
                tracks = self.tracker.update(self.yolo_detector.model, frame)
            else:
                # Enhanced DeepSORT Tracking
                tracks = self.tracker.update(detections, frame)
        else:
            # Create simple tracks from detections if no tracker available
            tracks = []
            for i, detection in enumerate(detections):
                track = type('SimpleTrack', (), {
                    'track_id': i + 1,
                    'bbox': detection.bbox,
                    'confidence': detection.confidence,
                    'age': None,
                    'gender': None,
                    'ethnicity': None,
                    'is_confirmed': lambda: True,
                    'is_deleted': lambda: False,
                    'to_tlbr': lambda: detection.bbox,
                    'time_since_update': 0,
                    'update_demographics': lambda x: None
                })()
                tracks.append(track)
        
        # Update current tracked count (confirmed or active)
        try:
            self.current_tracked_count = sum(
                1 for t in tracks
                if not getattr(t, 'is_deleted', lambda: False)() and (
                    not hasattr(t, 'is_confirmed') or t.is_confirmed()
                )
            )
        except Exception:
            self.current_tracked_count = len(tracks)

        # 3. Face Analysis for valid tracks
        for track in tracks:
            # Check track validity (different methods for different trackers)
            is_valid = True
            if hasattr(track, 'is_confirmed') and hasattr(track, 'is_deleted'):
                is_valid = track.is_confirmed() and not track.is_deleted()
            
            if is_valid:
                # Extract person ROI with bounds checking
                if hasattr(track, 'to_tlbr'):
                    x1, y1, x2, y2 = map(int, track.to_tlbr())
                else:
                    x1, y1, x2, y2 = map(int, track.bbox)
                
                # Ensure coordinates are within frame bounds
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Extract person ROI
                if x2 > x1 and y2 > y1:
                    person_roi = frame[y1:y2, x1:x2]
                    
                    # Face analysis (only if enabled and every N frames for performance)
                    face_enabled = getattr(self.config, 'FACE_ANALYSIS_ENABLED', True)
                    if (face_enabled and
                        track.time_since_update == 0 and 
                        person_roi.shape[0] > 30 and person_roi.shape[1] > 30 and
                        self.frame_count % self.config.PROCESS_FACE_EVERY_N_FRAMES == 0):
                        
                        face_data = self.face_analyzer.analyze(person_roi, track.track_id)
                        if face_data and face_data.is_valid():
                            track.update_demographics(face_data)
        
        # 4. Line Crossing Detection
        crossing_events = self.line_crossing.check_crossings(tracks)
        
        # 5. Update counters and send MQTT
        for event in crossing_events:
            # Update counters with the CrossingEvent object
            self.counters.update_count(event)

            # Build MQTT event payload (dict) and add tracked_count
            try:
                event_dict = event.to_dict()
            except Exception:
                # If already a dict
                event_dict = dict(event) if isinstance(event, dict) else {
                    "track_id": getattr(event, 'track_id', None),
                    "direction": getattr(event, 'direction', None),
                    "timestamp": getattr(event, 'timestamp', None),
                }
            event_dict["tracked_count"] = self.current_tracked_count
            self.mqtt_client.publish_event(event_dict)

            # Log original event object for terminal output (never let logging crash the system)
            try:
                self.terminal_logger.log_event(event)
            except Exception as log_err:
                print(f"[WARNING] Terminal log error (non-fatal): {log_err}")

        # Log every confirmed track in this frame (not just crossings)
        for track in tracks:
            try:
                if hasattr(track, 'is_deleted') and not track.is_deleted():
                    if hasattr(track, 'is_confirmed') and track.is_confirmed():
                        demo = getattr(track, 'demographics', {})
                        tid = getattr(track, 'track_id', None)
                        event = type('TrackLog', (), {
                            'track_id': tid,
                            'direction': 'TRACK',
                            'timestamp': time.time(),
                            'demographics': demo
                        })()  
                        self.terminal_logger.log_event(event)
            except Exception as log_err:
                print(f"[WARNING] Track log error (non-fatal): {log_err}")
        
        # 6. Enhanced DeepSORT visualization (handled in overlay_renderer)
        
        # 7. Render overlay
        display_frame = self.overlay_renderer.render(
            frame, tracks, self.counters.get_stats(), self.fps_counter.get_fps()
        )
        
        return display_frame
    
    def _start_auto_benchmark(self):
        """Start auto-benchmark thread if enabled"""
        if not getattr(self.config, 'AUTO_BENCHMARK_ENABLED', False):
            return
        
        interval_minutes = getattr(self.config, 'AUTO_BENCHMARK_INTERVAL_MINUTES', 5)
        print(f"[INFO] Auto-benchmark enabled: every {interval_minutes} minutes")
        
        self._benchmark_thread = threading.Thread(
            target=self._run_auto_benchmark,
            args=(interval_minutes,),
            daemon=True
        )
        self._benchmark_thread.start()
    
    def _run_auto_benchmark(self, interval_minutes):
        """Run benchmark periodically in background"""
        import subprocess
        
        interval_seconds = interval_minutes * 60
        last_run = time.time()
        
        while self.running:
            time.sleep(10)  # Check every 10 seconds
            
            if time.time() - last_run >= interval_seconds:
                try:
                    print(f"\n[INFO] Running auto-benchmark...")
                    
                    # Collect live metrics from current session
                    live_metrics = {
                        'timestamp': datetime.now().isoformat(),
                        'fps': self.fps_counter.get_fps(),
                        'total_frames': self.total_processed_frames,
                        'session_duration_seconds': (datetime.now() - self.session_start_time).total_seconds(),
                        'in_count': self.counters.get_stats().get('in_count', 0),
                        'out_count': self.counters.get_stats().get('out_count', 0),
                        'tracked_count': self.current_tracked_count,
                    }
                    
                    # Save live metrics to file
                    metrics_file = Path('benchmark/results/live_metrics.json')
                    metrics_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(metrics_file, 'w', encoding='utf-8') as f:
                        json.dump(live_metrics, f, ensure_ascii=False, indent=2)
                    
                    # Append to history CSV
                    history_file = Path('benchmark/results/live_metrics_history.csv')
                    import csv
                    file_exists = history_file.exists()
                    with open(history_file, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=live_metrics.keys())
                        if not file_exists:
                            writer.writeheader()
                        writer.writerow(live_metrics)
                    
                    # Publish to MQTT
                    self.mqtt_client.publish_event({
                        'type': 'auto_benchmark',
                        'data': live_metrics
                    })
                    
                    print(f"[OK] Auto-benchmark complete: FPS={live_metrics['fps']:.1f}")
                    last_run = time.time()
                    
                except Exception as e:
                    print(f"[WARNING] Auto-benchmark error: {e}")

    def run(self):
        """Main processing loop"""
        self.running = True
        print("[START] Starting CCTV processing...")
        
        # Start MQTT client
        self.mqtt_client.connect()
        
        # Start auto-benchmark thread if enabled
        self._start_auto_benchmark()
        
        try:
            frame_counter = 0
            while self.running:
                frame = self.rtsp_handler.get_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                # Process frame
                start_time = time.time()
                # Update line scaler with actual frame size before processing
                try:
                    h, w = frame.shape[:2]
                    self.line_crossing.update_frame_size(w, h)
                except Exception:
                    pass
                display_frame = self.process_frame(frame)
                self.fps_counter.update(time.time() - start_time)

                # Periodically publish system statistics (retained)
                now = time.time()
                if now - self._last_stats_publish > 5.0:  # every 5 seconds
                    try:
                        stats = self.counters.get_stats()
                        stats["tracked_count"] = self.current_tracked_count
                        self.mqtt_client.publish_statistics(stats)
                    except Exception as e:
                        print(f"⚠️ Failed to publish stats: {e}")
                    self._last_stats_publish = now
                
                frame_counter += 1
                
                # Display result (skip GUI if not available)
                if not self.headless_mode:
                    try:
                        window_name = 'CCTV Security System'
                        cv2.imshow(window_name, display_frame)
                        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                            print("[INFO] Window closed by user")
                            self.running = False
                            break
                        # Handle keyboard input
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                        elif key == ord('s'):
                            self.show_settings()
                        # Reset counters: 'r' -> reset counters, 'R' -> reset full session (history)
                        elif key == ord('r'):
                            self.reset_counters(full_reset=False, source="keyboard")
                        elif key == ord('R'):
                            self.reset_counters(full_reset=True, source="keyboard-full")
                        # Video playback controls (only for video file input)
                        elif key == ord(' '):  # Space = pause/resume
                            if hasattr(self.video_handler, 'toggle_pause'):
                                self.video_handler.toggle_pause()
                        elif key == ord('+') or key == ord('='):  # Speed up
                            if hasattr(self.video_handler, 'set_playback_speed'):
                                current_speed = getattr(self.video_handler, 'playback_speed', 1.0)
                                self.video_handler.set_playback_speed(current_speed + 0.25)
                        elif key == ord('-'):  # Slow down
                            if hasattr(self.video_handler, 'set_playback_speed'):
                                current_speed = getattr(self.video_handler, 'playback_speed', 1.0)
                                self.video_handler.set_playback_speed(current_speed - 0.25)
                    except cv2.error as e:
                        print(f"[WARNING] GUI not available, switching to headless mode: {e}")
                        self.headless_mode = True
                    except Exception as e:
                        print(f"[WARNING] Display error, switching to headless mode: {e}")
                        self.headless_mode = True
                
                # In headless mode, show status every 100 frames
                if self.headless_mode and frame_counter % 100 == 0:
                    stats = self.counters.get_stats()
                    print(
                        "[PROC] Processing frame "
                        f"{frame_counter} | FPS: {self.fps_counter.get_fps():.1f} "
                        f"| IN: {stats['total_in']} | OUT: {stats['total_out']} "
                        f"| Occupancy: {stats['current_occupancy']}"
                    )
                
                # Exit condition for headless mode (Ctrl+C)
                if self.headless_mode:
                    time.sleep(0.01)  # Small delay to prevent CPU overload
                
        except KeyboardInterrupt:
            print("\n[INFO] Stopping system...")
        finally:
            self.cleanup()

    def show_settings(self):
        """Show settings window"""
        print("[INFO] Settings - Press 's' to open settings, 'r' to reset counters, 'R' to reset session")
        # TODO: Implement settings GUI

    def cleanup(self):
        """Cleanup resources and save session statistics"""
        self.running = False
        
        # Save session statistics before cleanup
        self.save_session_statistics()
        
        self.rtsp_handler.release()
        self.mqtt_client.disconnect()
        
        # Safe cleanup of OpenCV windows
        try:
            cv2.destroyAllWindows()
        except cv2.error as e:
            print(f"[WARNING] OpenCV GUI cleanup skipped: {e}")
        except Exception as e:
            print(f"[WARNING] Window cleanup error (non-critical): {e}")
        
        print("[OK] System shutdown complete")
    
    def save_session_statistics(self):
        """บันทึกสถิติการใช้งานทุกครั้งที่รันระบบ / Save session statistics to log file"""
        try:
            session_end_time = datetime.now()
            duration = (session_end_time - self.session_start_time).total_seconds()
            
            # Get final counters
            stats = self.counters.get_stats()
            avg_fps = self.fps_counter.get_fps()
            
            # Build session log entry
            session_data = {
                "session_id": self.session_id,
                "start_time": self.session_start_time.isoformat(),
                "end_time": session_end_time.isoformat(),
                "duration_seconds": round(duration, 2),
                "input_source": self.input_source,
                "video_source": self.video_source_path or (self.config.RTSP_URL_MASKED if self.input_source == 'rtsp' else f"webcam_{getattr(self.config, 'WEBCAM_INDEX', 0)}"),
                "total_frames_processed": self.frame_count,
                "average_fps": round(avg_fps, 2),
                "total_in": stats.get('total_in', 0),
                "total_out": stats.get('total_out', 0),
                "final_occupancy": stats.get('current_occupancy', 0),
                "peak_occupancy": stats.get('peak_occupancy', 0),
            }
            
            # Ensure logs directory exists
            logs_dir = Path("data/logs")
            logs_dir.mkdir(parents=True, exist_ok=True)
            
            # Save to session log file (JSON)
            log_file = logs_dir / f"session_{self.session_id}.json"
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
            
            # Append to history CSV
            history_file = logs_dir / "session_history.csv"
            write_header = not history_file.exists()
            
            with open(history_file, 'a', encoding='utf-8') as f:
                if write_header:
                    f.write("session_id,start_time,end_time,duration_seconds,input_source,total_frames,avg_fps,total_in,total_out,final_occupancy,peak_occupancy\n")
                f.write(f"{session_data['session_id']},{session_data['start_time']},{session_data['end_time']},{session_data['duration_seconds']},{session_data['input_source']},{session_data['total_frames_processed']},{session_data['average_fps']},{session_data['total_in']},{session_data['total_out']},{session_data['final_occupancy']},{session_data['peak_occupancy']}\n")
            
            print(f"[INFO] Session statistics saved: {log_file}")
            print(f"   Duration: {duration:.1f}s | Frames: {self.frame_count} | Avg FPS: {avg_fps:.1f}")
            print(f"   IN: {stats.get('total_in', 0)} | OUT: {stats.get('total_out', 0)} | Peak: {stats.get('peak_occupancy', 0)}")
            
        except Exception as e:
            print(f"[WARNING] Failed to save session statistics: {e}")

if __name__ == "__main__":
    try:
        # Check dependencies first
        check_dependencies()
        
        # Show input source selection menu FIRST
        input_source, extra_param = show_input_menu()
        
        # Start FairFace Benchmark in background AFTER menu selection
        print("\n[INFO] Starting FairFace Benchmark in background...")
        try:
            benchmark_thread = threading.Thread(
                target=lambda: subprocess.run(
                    [sys.executable, "benchmark/run_benchmark_simple.py"],
                    cwd=str(Path(__file__).parent),
                    capture_output=False
                ),
                daemon=True,
                name="FairFace-Benchmark"
            )
            benchmark_thread.start()
            print("[OK] FairFace Benchmark started in background thread")
        except Exception as e:
            print(f"[WARNING] Could not start FairFace Benchmark: {e}")
        
        # Initialize system with selected input source
        webcam_idx = extra_param if input_source == 'webcam' else None
        system = CCTVSystem(input_source=input_source, webcam_index=webcam_idx)
        system.run()
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        sys.exit(1)
