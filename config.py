"""
Configuration file for CCTV Security System
โครงการความมั่นคงระดับสูง - การตั้งค่าระบบ
"""

import os
from pathlib import Path
from urllib.parse import quote

# โหลดค่าลับ (รหัสผ่านกล้อง / MQTT) จากไฟล์ .env ข้างไฟล์นี้
# ไฟล์ .env ถูก .gitignore ไว้ จึงไม่ขึ้น GitHub — ดูตัวอย่างใน .env.example
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    print("[WARNING] python-dotenv not installed - reading secrets from OS environment only")


def _env_bool(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def mask_url(url):
    """ซ่อนรหัสผ่านใน URL เช่น rtsp://admin:secret@host -> rtsp://admin:***@host"""
    import re
    return re.sub(r"(://[^:/@]+:)[^/]*@", r"\1***@", url or "")


class Config:
    # === RTSP Camera Settings ===
    # HIKVISION Stream Channels:
    # Channel 101 = Main Stream (1920x1080, 25fps, H.265) - High quality
    # Channel 102 = Sub Stream (640x480, 25fps, H.264) - Lower bandwidth
    
    # ข้อมูลกล้องอ่านจาก .env (RTSP_USER, RTSP_PASSWORD, RTSP_HOST, RTSP_PORT, RTSP_CHANNEL)
    # RTSP_CHANNEL: 101 = Main Stream ความละเอียดสูงสุด, 102 = Sub Stream ประหยัดแบนด์วิดท์
    RTSP_USER = os.getenv("RTSP_USER", "admin")
    RTSP_PASSWORD = os.getenv("RTSP_PASSWORD", "")
    RTSP_HOST = os.getenv("RTSP_HOST", "192.168.1.2")
    RTSP_PORT = int(os.getenv("RTSP_PORT", "554"))
    RTSP_CHANNEL = os.getenv("RTSP_CHANNEL", "101")
    # ตั้ง RTSP_URL ใน .env ตรงๆ ได้ถ้าต้องการ (จะใช้แทนค่าด้านบน)
    # quote() เข้ารหัสอักขระพิเศษในรหัสผ่าน (เช่น @ : /) ไม่ให้ URL เพี้ยน
    RTSP_URL = os.getenv("RTSP_URL") or (
        f"rtsp://{quote(RTSP_USER, safe='')}:{quote(RTSP_PASSWORD, safe='')}"
        f"@{RTSP_HOST}:{RTSP_PORT}/Streaming/Channels/{RTSP_CHANNEL}"
    )
    # ใช้ตัวนี้เวลาพิมพ์ออกหน้าจอ/บันทึก log (ซ่อนรหัสผ่าน)
    RTSP_URL_MASKED = mask_url(RTSP_URL)
    
    RTSP_BUFFER_SIZE = 1  # ลด buffer สำหรับ real-time
    RTSP_TIMEOUT = 10
    RTSP_RECONNECT_DELAY = 5  # seconds
    RTSP_MAX_RECONNECT_ATTEMPTS = 10
    
    # === Video Input Source ===
    # Input source mode: 'rtsp', 'video_file', 'webcam'
    INPUT_SOURCE = "rtsp"  # เปลี่ยนเป็น 'video_file' เพื่อใช้ไฟล์วิดีโอ
    
    # Video file settings (ใช้เมื่อ INPUT_SOURCE = 'video_file')
    # ใส่ "browse" เพื่อเปิด file dialog เลือกไฟล์
    VIDEO_FILE_PATH = "browse"  # หรือใส่ path เช่น "videos/test.mp4"
    VIDEO_LOOP = True  # Loop video when finished
    VIDEO_PLAYBACK_SPEED = 1.0  # 1.0 = normal, 0.5 = slow, 2.0 = fast
    
    # Webcam settings (ใช้เมื่อ INPUT_SOURCE = 'webcam')
    WEBCAM_INDEX = 0  # Webcam index (0 = default camera)
    
    # === Video Processing ===
    TARGET_FPS = 25
    PROCESS_EVERY_N_FRAMES = 15  # วิเคราะห์ทุกเฟรม (แม่นยำสุด แต่ FPS ต่ำ)
    
    # ปรับตาม Main Stream (1920x1080) จากการตั้งค่ากล้อง
    INPUT_WIDTH = 1920
    INPUT_HEIGHT = 1080
    
    # สำหรับ Sub Stream (หาก FPS ต่ำเกินไป)
    # INPUT_WIDTH = 640
    # INPUT_HEIGHT = 480
    
    # === YOLO Detection Settings ===
    YOLO_MODEL = "models/yolo/yolov8n.pt"  # yolov8n = เร็วสุด, yolov8s/m = แม่นกว่า
    YOLO_CONFIDENCE = 0.50  # เพิ่มเป็น 0.50 เพื่อลด false positive
    YOLO_IOU_THRESHOLD = 0.70  # เพิ่มเป็น 0.70 เพื่อลด box ซ้ำซ้อนมากขึ้น
    YOLO_DEVICE = "cuda"  # ใช้ GPU NVIDIA
    YOLO_HALF_PRECISION = True  # FP16 เพิ่มความเร็ว 2x บน GPU
    YOLO_BATCH_SIZE = 1  # Batch size สำหรับ real-time
    YOLO_MAX_DET = 50  # จำกัด detection สูงสุด (เพิ่มความเร็ว)
    
    # === GPU Optimization ===
    USE_TENSORRT = False  # เปิดเป็น True ถ้ามี TensorRT (เร็วขึ้น 3-5x)
    DEEPFACE_BACKEND = "retinaface"  # retinaface = แม่นกว่า ssd มาก, รองรับใบหน้าหลายเชื้อชาติ
    DEEPFACE_MODEL = "Facenet512"    # Facenet512 = bias น้อยกว่า VGG-Face, แม่นกว่าทั้งเพศและเชื้อชาติ
    # หมายเหตุ: GENDER/ETHNICITY_CONFIDENCE_THRESHOLD กำหนดไว้ครั้งเดียวในหัวข้อ Face Analysis ด้านล่าง
    
    # === Tracking Settings ===
    # Tracking method: 'yolov8_native', 'deepsort', 'auto'
    TRACKING_METHOD = "auto"  # auto = prefer yolov8_native, fallback to deepsort
    YOLOV8_TRACKER_TYPE = "botsort"  # 'botsort' or 'bytetrack'
    YOLOV8_TRACKER_CONFIG = "configs/botsort_custom.yaml"
    # Visual debug: show predicted bounding box from tracker
    SHOW_PREDICTED_BBOX = False
    
    # === Enhanced DeepSORT Tracking Settings ===
    DEEPSORT_MODEL_PATH = "models/deepsort/deep_sort.pb"
    DEEPSORT_MAX_DIST = 0.3  # เพิ่มเป็น 0.3 เพื่อให้ track ไม่หลุดง่าย
    DEEPSORT_MIN_CONFIDENCE = 0.50  # เพิ่มเป็น 0.50 ให้ตรงกับ YOLO
    DEEPSORT_NMS_MAX_OVERLAP = 0.5  # ลดเป็น 0.5 เพื่อลด box ซ้ำมากขึ้น
    DEEPSORT_MAX_IOU_DISTANCE = 0.7  # เพิ่มเป็น 0.7 เพื่อ match ได้ง่ายขึ้น
    DEEPSORT_MAX_AGE = 50  # ลดเป็น 50 เพื่อลบ track เก่าเร็วขึ้น
    DEEPSORT_N_INIT = 3  # เพิ่มเป็น 3 เพื่อ confirm track ช้าลง (ลด false ID)
    DEEPSORT_NN_BUDGET = 100
    
    # Advanced Enhanced Features
    ENABLE_FALLBACK_TRACKER = True
    ADAPTIVE_CONFIDENCE_THRESHOLD = True
    TRACK_QUALITY_SCORING = True
    ENHANCED_VELOCITY_PREDICTION = True
    ADVANCED_TRACK_RECOVERY = True
    
    # Enhanced DeepSORT Advanced Parameters
    ENHANCED_MAX_TRAJECTORY_LENGTH = 50  # เพิ่มจาก 30
    ENHANCED_BBOX_SMOOTHING = True
    ENHANCED_QUALITY_THRESHOLD = 0.3
    ENHANCED_RECOVERY_WINDOW = 15  # frames
    ENHANCED_LOST_TRACK_RETENTION = 30  # frames
    
    # Adaptive Algorithm Settings
    ADAPTIVE_SCENE_ANALYSIS = True
    ADAPTIVE_DENSITY_THRESHOLDS = {
        'sparse': 3,    # < 3 detections = sparse scene
        'crowded': 15   # > 15 detections = crowded scene
    }
    ADAPTIVE_MOTION_ANALYSIS = True
    ADAPTIVE_CONFIDENCE_RANGE = {
        'min_adjustment': -0.15,
        'max_adjustment': 0.2
    }
    
    # === Face Analysis Settings ===
    FACE_ANALYSIS_ENABLED = True  # เปิด Face Analysis (Age/Gender/Ethnicity)
    FACE_MODEL_PATH = "models/face/"
    FACE_MIN_SIZE = 40  # ลดเป็น 40 เพื่อวิเคราะห์ได้มากขึ้น
    FACE_CONFIDENCE_THRESHOLD = 0.6  # ลดเป็น 0.6 เพื่อวิเคราะห์ได้มากขึ้น
    PROCESS_FACE_EVERY_N_FRAMES = 5  # วิเคราะห์ทุก 5 เฟรม (แม่นยำสุด)
    SHOW_FACE_BBOX_ONLY = True  # แสดงกรอบแค่ใบหน้าเท่านั้น (ไม่แสดงกรอบตัวคน)
    # True = ถ้าไม่มี DeepFace ให้สุ่มอายุ/เพศ/เชื้อชาติ (ใช้ทดสอบหน้าจอเท่านั้น ห้ามเปิดตอนเก็บข้อมูลจริง)
    FACE_ALLOW_MOCK = False
    
    # Age/Gender Settings
    AGE_GENDER_MODEL = "models/face/age_gender/fairface_4race.onnx"
    AGE_CONFIDENCE_THRESHOLD = 0.6  # ลดเป็น 0.6 เพื่อให้วิเคราะห์บ่อยขึ้น
    GENDER_CONFIDENCE_THRESHOLD = 0.6  # ลดเป็น 0.6 เพื่อให้วิเคราะห์บ่อยขึ้น
    
    # === Ethnicity Classification (Security Feature) ===
    ETHNICITY_ENABLED = True  # เปิด ethnicity detection
    ETHNICITY_MODEL = "models/face/ethnicity/fairface_race.onnx"
    ETHNICITY_CONFIDENCE_THRESHOLD = 0.5
    ETHNICITY_ABSTAIN_THRESHOLD = 0.6  # Don't show if confidence < this
    
    # Ethnicity Groups Mapping for Security Classification
    ETHNICITY_GROUPS = {
        "Asian": "GroupA",
        "White": "GroupB", 
        "Black": "GroupC",
        "Indian": "GroupD",
        "Others": "GroupE"
    }
    
    # === Auto Benchmark Settings ===
    AUTO_BENCHMARK_ENABLED = True  # เปิด/ปิด auto benchmark
    AUTO_BENCHMARK_INTERVAL_MINUTES = 1  # รัน benchmark ทุก X นาที
    AUTO_BENCHMARK_SAVE_HISTORY = True  # เก็บประวัติ benchmark
    
    # === Line Crossing Settings ===
    # Virtual counting line coordinates (can be adjusted via GUI)
    COUNTING_LINE = {
        "start": (960, 100),   # x = 1920//2 = 960, y = 100 (จากบน)
        "end":   (960, 980),   # x = 960, y = 980 (จากบน)
        "thickness": 3,
        "color": (0, 255, 0)
    }

    # Counting line rendering mode: 'fixed' uses COUNTING_LINE coords;
    # 'center_vertical' forces a vertical line centered on the frame and
    # spanning from margin_top to height - margin_bottom.
    COUNTING_LINE_MODE = 'center_vertical'  # 'fixed' or 'center_vertical'
    COUNTING_LINE_MARGIN_TOP = 100
    COUNTING_LINE_MARGIN_BOTTOM = 100
    
    # Crossing detection parameters
    CROSSING_BUFFER_DISTANCE = 80  # pixels (tolerant for low FPS / fast crossings)
    CROSSING_COOLDOWN_TIME = 3.0   # seconds between counts for same ID
    MIN_TRACK_LENGTH = 3           # minimum track points before counting
    
    # === MQTT Settings ===
    MQTT_ENABLED = True
    # ค่าการเชื่อมต่อ MQTT อ่านจาก .env (ห้ามใช้ broker สาธารณะ เพราะข้อความมีข้อมูลเพศ/อายุ/เชื้อชาติ)
    MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")  # Local mosquitto broker
    MQTT_USE_TLS = _env_bool("MQTT_USE_TLS", False)
    MQTT_PORT = int(os.getenv("MQTT_PORT", "8883" if MQTT_USE_TLS else "1883"))
    MQTT_USERNAME = os.getenv("MQTT_USERNAME") or None
    MQTT_PASSWORD = os.getenv("MQTT_PASSWORD") or None
    MQTT_CA_CERTS = os.getenv("MQTT_CA_CERTS") or None  # path ไฟล์ CA (.crt) ถ้า broker ใช้ cert ที่เซ็นเอง
    MQTT_TLS_INSECURE = _env_bool("MQTT_TLS_INSECURE", False)  # True = ไม่ตรวจชื่อ host ใน cert (ใช้ทดสอบเท่านั้น)
    MQTT_TOPIC = "/cctv/people_tracking"
    MQTT_QOS = 1
    MQTT_RETAIN = False
    MQTT_KEEPALIVE = 60
    # Optional plain-text summary label publish
    MQTT_SEND_SUMMARY_LABEL = True
    MQTT_SUMMARY_TOPIC = MQTT_TOPIC + "/label"
    # Compatibility: also publish to topics without leading '/'
    MQTT_COMPAT_PUBLISH_NO_LEADING_SLASH = True
    # Command/control topic for remote actions (e.g., reset counters)
    MQTT_COMMAND_TOPIC = MQTT_TOPIC + "/commands"
    
    # === Display Settings ===
    DISPLAY_WINDOW_NAME = "CCTV Security System"
    DISPLAY_WIDTH = 1280
    DISPLAY_HEIGHT = 720
    
    # Overlay settings
    FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.6
    FONT_THICKNESS = 2
    TEXT_COLOR = (255, 255, 255)  # White
    BACKGROUND_COLOR = (0, 0, 0)  # Black
    # Show semi-transparent background behind main overlay (top-left)
    SHOW_OVERLAY_BACKGROUND = False
    # Debug/visualization toggles
    SHOW_TRACK_CENTER = True  # draw small dot at track center for debugging line crossing
    # Toggle specific overlay sections
    SHOW_CURRENT_PEOPLE = False  # Hide "CURRENT PEOPLE" list in overlay
    
    # Bounding box colors
    BBOX_COLORS = {
        "person": (0, 255, 0),      # Green
        "tracked": (255, 0, 0),     # Blue  
        "confirmed": (0, 0, 255)    # Red
    }
    
    # === Logging Settings ===
    LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
    LOG_TO_FILE = True
    LOG_DIR = "data/logs"
    TERMINAL_LOG_ENABLED = True
    
    # === Performance Settings ===
    MAX_BATCH_SIZE = 4
    GPU_MEMORY_FRACTION = 0.8
    ENABLE_TENSORRT = False
    ENABLE_ONNX = True
    
    # === Paths ===
    BASE_DIR = Path(__file__).parent
    MODELS_DIR = BASE_DIR / "models"
    DATA_DIR = BASE_DIR / "data" 
    CONFIGS_DIR = BASE_DIR / "configs"
    
    @classmethod
    def validate_paths(cls):
        """Validate that required paths exist"""
        required_dirs = [
            cls.MODELS_DIR,
            cls.DATA_DIR,
            cls.CONFIGS_DIR,
            cls.DATA_DIR / "logs"
        ]
        
        for dir_path in required_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            
    @classmethod  
    def load_from_file(cls, config_path):
        """Load configuration from YAML file (optional)"""
        # TODO: Implement YAML config loading
        pass
        
    def __init__(self):
        self.validate_paths()
        
        # Runtime settings that can be modified
        self.current_fps = 0
        self.processing_enabled = True
        self.face_analysis_enabled = True
        self.ethnicity_display_enabled = self.ETHNICITY_ENABLED
        
    def get_display_config(self):
        """Get display configuration"""
        return {
            "width": self.DISPLAY_WIDTH,
            "height": self.DISPLAY_HEIGHT, 
            "font": self.FONT,
            "font_scale": self.FONT_SCALE,
            "font_thickness": self.FONT_THICKNESS,
            "text_color": self.TEXT_COLOR,
            "bg_color": self.BACKGROUND_COLOR
        }
        
    def get_mqtt_config(self):
        """Get MQTT configuration"""
        return {
            "enabled": self.MQTT_ENABLED,
            "broker": self.MQTT_BROKER,
            "port": self.MQTT_PORT,
            "topic": self.MQTT_TOPIC,
            "qos": self.MQTT_QOS,
            "retain": self.MQTT_RETAIN,
            "keepalive": self.MQTT_KEEPALIVE
        }
