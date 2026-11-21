# 📊 การวิเคราะห์โค้ดทั้งหมด - YOLO CV CCTV System

## 🎯 สรุปภาพรวมระบบ

**YOLO CV CCTV V4** เป็นระบบตรวจจับและติดตามบุคคลแบบเรียลไทม์ พร้อมวิเคราะห์ลักษณะประชากรศาสตร์ (Demographics) โดยใช้ AI และ Deep Learning เพื่อนับจำนวนคนเข้า-ออก และวิเคราะห์ Age, Gender, Ethnicity

**ระดับโครงการ**: โครงการระดับสูงสำหรับหน่วยงานรัฐบาล (High-level security project for government agencies)

---

## 🏗️ สถาปัตยกรรมระบบ (System Architecture)

### 1. โครงสร้างไดเรกทอรี

```
YOLO_CV_CCTV/
├── config.py                    # ไฟล์ตั้งค่าหลักทั้งระบบ
├── main.py                      # จุดเริ่มต้นโปรแกรม
├── requirements.txt             # แพ็กเกจที่ต้องติดตั้ง
├── README.md                    # คู่มือโครงการ
│
├── src/                         # โค้ดหลักของระบบ
│   ├── detect/                  # โมดูลตรวจจับ (YOLO)
│   │   └── yolo_detector.py
│   ├── track/                   # โมดูลติดตาม (DeepSORT)
│   │   ├── deepsort_tracker.py
│   │   └── yolov8_native_tracker.py
│   ├── face/                    # โมดูลวิเคราะห์ใบหน้า
│   │   └── face_analyzer.py
│   ├── analytics/               # โมดูลวิเคราะห์
│   │   ├── line_crossing.py
│   │   └── counters.py
│   ├── viz/                     # โมดูลแสดงผล
│   │   ├── overlay_renderer.py
│   │   └── terminal_logger.py
│   ├── io/                      # โมดูล I/O
│   │   ├── rtsp_handler.py
│   │   ├── video_file_handler.py
│   │   └── mqtt_client.py
│   ├── utils/                   # โมดูลเสริม
│   │   └── fps_counter.py
│   └── web/                     # Dashboard API
│       └── dashboard.py
│
├── models/                      # โฟลเดอร์เก็บโมเดล AI
│   ├── yolo/                    # YOLOv8 models
│   ├── deepsort/                # DeepSORT models
│   └── face/                    # Face analysis models
│
├── benchmark/                   # ระบบทดสอบประสิทธิภาพ
│   ├── run_benchmark.py
│   └── results/
│
├── data/                        # โฟลเดอร์เก็บข้อมูล
│   └── logs/                    # Session logs
│
├── configs/                     # ไฟล์ตั้งค่าเพิ่มเติม
│   └── botsort_custom.yaml
│
└── streamlit_dashboard.py       # Dashboard แสดงผลแบบ Web
```

### 2. ไลบรารีหลักที่ใช้

| ไลบรารี | เวอร์ชัน | หน้าที่ |
|---------|----------|---------|
| **ultralytics** | Latest | YOLOv8 object detection |
| **opencv-python** | 4.8.0+ | ประมวลผลภาพและวิดีโอ |
| **DeepFace** | Latest | วิเคราะห์ Age/Gender/Ethnicity |
| **deep-sort-realtime** | Latest | Multi-object tracking |
| **paho-mqtt** | Latest | ส่งข้อมูลแบบ Real-time |
| **streamlit** | Latest | Web Dashboard |
| **FastAPI** | Latest | REST API |
| **torch** | Latest | GPU acceleration (CUDA) |

---

## 📋 วิเคราะห์ไฟล์หลัก

### 1. `config.py` - ไฟล์ตั้งค่าหลัก (271 บรรทัด)

**หน้าที่**: จัดการการตั้งค่าทั้งหมดของระบบ

#### ส่วนสำคัญ:

| หมวดหมู่ | พารามิเตอร์สำคัญ | ค่าเริ่มต้น | คำอธิบาย |
|----------|------------------|-------------|----------|
| **Video Input** | `INPUT_SOURCE` | `"rtsp"` | แหล่งวิดีโอ: rtsp/video_file/webcam |
| | `RTSP_URL` | `rtsp://admin:...@192.168.1.2:554/...` | URL กล้อง HIKVISION |
| | `VIDEO_FILE_PATH` | `"browse"` | เลือกไฟล์วิดีโอ |
| **YOLO Detection** | `YOLO_MODEL` | `"yolov8n.pt"` | โมเดล YOLOv8 (n=เร็ว, s/m=แม่น) |
| | `YOLO_CONFIDENCE` | `0.50` | Threshold ตรวจจับ |
| | `YOLO_DEVICE` | `"cuda"` | ใช้ GPU |
| | `YOLO_HALF_PRECISION` | `True` | FP16 เพิ่มความเร็ว 2x |
| **Tracking** | `TRACKING_METHOD` | `"auto"` | yolov8_native / deepsort |
| | `DEEPSORT_MAX_AGE` | `50` | เวลาจำ track (frames) |
| | `DEEPSORT_N_INIT` | `3` | จำนวนเฟรมยืนยัน track |
| **Face Analysis** | `FACE_ANALYSIS_ENABLED` | `True` | เปิด/ปิด face analysis |
| | `PROCESS_FACE_EVERY_N_FRAMES` | `5` | วิเคราะห์ทุก N เฟรม |
| | `AGE_GENDER_MODEL` | `fairface_4race.onnx` | โมเดล Age/Gender |
| **Ethnicity** | `ETHNICITY_ENABLED` | `True` | เปิด/ปิด ethnicity |
| | `ETHNICITY_MODEL` | `fairface_race.onnx` | โมเดล Ethnicity |
| | `ETHNICITY_GROUPS` | `{Asian, White, Black, Indian, Others}` | กลุ่มเชื้อชาติ |
| **Line Crossing** | `COUNTING_LINE_MODE` | `"center_vertical"` | โหมดเส้นนับ |
| | `CROSSING_BUFFER_DISTANCE` | `80` | ระยะ buffer (pixels) |
| | `CROSSING_COOLDOWN_TIME` | `3.0` | เวลารอระหว่างการนับ (วินาที) |
| **MQTT** | `MQTT_ENABLED` | `True` | เปิด/ปิด MQTT |
| | `MQTT_BROKER` | `"test.mosquitto.org"` | MQTT Broker |
| | `MQTT_TOPIC` | `"/cctv/people_tracking"` | MQTT Topic |

**จุดเด่น**:
- รองรับ 3 แหล่ง input: RTSP Camera, Video File, Webcam
- ปรับแต่ง performance ง่าย (FPS, GPU, Batch Size)
- มี Auto Benchmark ทุก X นาที
- รองรับ MQTT command/control

---

### 2. `main.py` - โปรแกรมหลัก (685 บรรทัด)

**หน้าที่**: จัดการ pipeline ทั้งหมด และประสานงานระหว่างโมดูล

#### โครงสร้างหลัก:

```python
class CCTVSystem:
    def __init__(self, input_source, webcam_index)
    def process_frame(self, frame)  # Pipeline หลัก
    def run(self)                   # Main loop
    def cleanup(self)               # ทำความสะอาด
```

#### Pipeline การทำงาน (ใน `process_frame`):

```
1. YOLO Detection
   └─> ตรวจจับคน (Person) ในเฟรม
   
2. Multi-Object Tracking
   └─> ติดตาม ID ของแต่ละคน (DeepSORT หรือ YOLOv8 Native)
   
3. Face Analysis
   └─> วิเคราะห์ Age, Gender, Ethnicity (ทุก N เฟรม)
   
4. Line Crossing Detection
   └─> ตรวจจับการข้ามเส้น → นับ IN/OUT
   
5. Update Counters & Send MQTT
   └─> อัปเดตสถิติและส่ง MQTT
   
6. Render Overlay
   └─> วาด Bounding Box, Label, สถิติ
```

**จุดเด่น**:
- รองรับ headless mode (ไม่มี GUI)
- บันทึก session statistics อัตโนมัติ
- รองรับ video playback controls (pause, speed)
- Auto-benchmark ทุก N นาที (background thread)
- จัดการ MQTT command (reset counters)

**Keyboard Controls**:
- `q` - ออกจากโปรแกรม
- `r` - Reset counters
- `R` - Reset session (full reset)
- `Space` - Pause/Resume (video file)
- `+/-` - ปรับความเร็ววิดีโอ

---

### 3. `src/detect/yolo_detector.py` - ตรวจจับบุคคล (371 บรรทัด)

**หน้าที่**: ตรวจจับบุคคลด้วย YOLOv8

#### คลาสหลัก:

```python
class Detection:
    """แทนผลการตรวจจับ 1 คน"""
    bbox: [x1, y1, x2, y2]
    confidence: float
    class_id: int
    
class YOLODetector:
    """ตัวตรวจจับหลัก"""
    def detect(self, frame) -> List[Detection]
    def detect_batch(self, frames) -> List[List[Detection]]
```

**จุดเด่น**:
- รองรับ GPU (CUDA) + Half Precision (FP16)
- Model warmup สำหรับประสิทธิภาพ
- Batch processing
- Performance statistics tracking

**การทำงาน**:
1. โหลด YOLOv8 model (`yolov8n.pt`)
2. Inference บน GPU (ถ้ามี)
3. กรองผลลัพธ์ด้วย confidence threshold
4. คืนค่า `Detection` objects

---

### 4. `src/track/deepsort_tracker.py` - ติดตามบุคคล (498 บรรทัด)

**หน้าที่**: ติดตามบุคคลข้ามเฟรม พร้อม ID ที่คงที่

#### คลาสหลัก:

```python
class Track:
    """แทนการติดตาม 1 คน"""
    track_id: int
    bbox: [x1, y1, x2, y2]
    confidence: float
    trajectory: List[Tuple[int, int]]  # ประวัติการเคลื่อนที่
    demographics: Dict                  # Age, Gender, Ethnicity
    time_since_update: int
    hits: int                          # จำนวนครั้งที่มองเห็น
    
class DeepSORTTracker:
    """ตัวติดตามหลัก"""
    def update(self, detections, frame) -> List[Track]
```

**จุดเด่น**:
- รองรับ 2 โหมด: DeepSORT RealTime กับ Simple IoU Tracker
- Fallback อัตโนมัติถ้าไม่มี DeepSORT
- Enhanced features:
  - Velocity prediction (ทำนายตำแหน่งถัดไป)
  - Trajectory smoothing
  - Quality scoring
  - Track recovery
- Adaptive algorithms:
  - Scene analysis (sparse/crowded)
  - Motion analysis
  - Confidence adjustment

**การทำงาน**:
1. รับ detections จาก YOLO
2. Match กับ tracks เดิม (ด้วย IoU หรือ appearance features)
3. สร้าง track ใหม่สำหรับ detection ใหม่
4. ลบ track เก่าที่หายไป
5. คืนค่า active tracks

---

### 5. `src/face/face_analyzer.py` - วิเคราะห์ใบหน้า (435 บรรทัด)

**หน้าที่**: วิเคราะห์ Age, Gender, Ethnicity ด้วย DeepFace

#### คลาสหลัก:

```python
class FaceAnalysisResult:
    """ผลการวิเคราะห์ใบหน้า"""
    age: int
    gender: str                  # "Man" or "Woman"
    ethnicity: str               # "asian", "white", "black", "indian"
    group: str                   # "GroupA", "GroupB", ...
    gender_confidence: float
    ethnicity_confidence: float
    
class FaceAnalyzer:
    """ตัววิเคราะห์หลัก"""
    def analyze(self, person_roi, track_id) -> FaceAnalysisResult
```

**จุดเด่น**:
- ใช้ DeepFace (backend: ssd)
- Caching ผลการวิเคราะห์เพื่อความเสถียร
- Stabilization (เฉลี่ยผลหลายครั้ง)
- รองรับ confidence threshold
- แมป ethnicity เป็น security groups

**Ethnicity Groups**:
```python
ETHNICITY_GROUPS = {
    "Asian": "GroupA",
    "White": "GroupB", 
    "Black": "GroupC",
    "Indian": "GroupD",
    "Others": "GroupE"
}
```

**การทำงาน**:
1. ตัดส่วน face region จาก person ROI (ส่วนบน 60%)
2. วิเคราะห์ด้วย DeepFace
3. Cache ผลลัพธ์สำหรับ track นี้
4. Stabilize โดยเฉลี่ยผลหลายครั้ง
5. คืนค่า `FaceAnalysisResult`

---

### 6. `src/analytics/line_crossing.py` - ตรวจจับการข้ามเส้น (469 บรรทัด)

**หน้าที่**: ตรวจจับการข้ามเส้น เพื่อนับ IN/OUT

#### คลาสหลัก:

```python
class CrossingEvent:
    """เหตุการณ์การข้ามเส้น"""
    track_id: int
    direction: str              # "IN" or "OUT"
    timestamp: float
    demographics: Dict
    
class LineCrossing:
    """ตัวตรวจจับการข้ามเส้น"""
    def check_crossings(self, tracks) -> List[CrossingEvent]
```

**อลกอริทึม**:
1. **Line Intersection Detection**: ตรวจสอบว่าเส้นทาง (trajectory) ข้ามเส้นนับหรือไม่
2. **Direction Detection**: ใช้ cross product เพื่อหาทิศทาง
3. **Hysteresis**: ป้องกันการนับซ้ำจากการแกว่ง
4. **Cooldown**: รอ N วินาทีก่อนนับซ้ำสำหรับ track เดียวกัน
5. **Buffer Distance**: ระยะ tolerance สำหรับ low FPS

**จุดเด่น**:
- รองรับ 2 โหมด: `fixed` (พิกัดคงที่) หรือ `center_vertical` (กลางเฟรม)
- Auto-scale ตามขนาดเฟรม
- Hysteresis + Cooldown เพื่อความแม่นยำ
- รองรับ fast crossing (low FPS)

---

### 7. `src/analytics/counters.py` - นับสถิติ

**หน้าที่**: เก็บสถิติการนับ IN/OUT แยกตาม Gender, Ethnicity

```python
class PeopleCounters:
    def update_count(self, event: CrossingEvent)
    def get_stats(self) -> Dict
```

**สถิติที่เก็บ**:
- `in_count`: {Male, Female, Unknown}
- `out_count`: {Male, Female, Unknown}
- `ethnicity_count`: {Asian, White, Black, Indian}
- `current_occupancy`: จำนวนคนในพื้นที่ปัจจุบัน
- `peak_occupancy`: จำนวนคนสูงสุด

---

### 8. `src/viz/overlay_renderer.py` - แสดงผลกราฟิก (700 บรรทัด)

**หน้าที่**: วาด overlay บนวิดีโอ

```python
class OverlayRenderer:
    def render(self, frame, tracks, stats, fps) -> np.ndarray
```

**สิ่งที่แสดง**:
1. **Bounding Boxes พร้อม Labels**
   - ID:xxx | F/M/U | Age Group | Asia/Europe/U
   - Confidence % สำหรับ Gender
   - มุมเน้น 4 มุม

2. **Main Overlay (บนซ้าย)**
   ```
   IN  M:10 F:8 U:2
   OUT M:7  F:5 U:1
   
   person : 5
   Asia : 15  Europe : 3
   ```

3. **Counting Line**
   - เส้นสีเขียว + ข้อความ "COUNT LINE"

4. **FPS Counter (ล่างขวา)**
   - สีเขียว (FPS ≥ 20)
   - สีเหลือง (10 ≤ FPS < 20)
   - สีแดง (FPS < 10)

5. **System Status (บนขวา)**
   - RTSP: CONNECTED
   - MQTT: CONNECTED
   - ETHNICITY: ENABLED

**Mode พิเศษ**:
- `SHOW_FACE_BBOX_ONLY`: แสดงแค่กรอบใบหน้า (ไม่แสดงกรอบตัวคน)

---

### 9. `src/io/mqtt_client.py` - ส่งข้อมูล MQTT (459 บรรทัด)

**หน้าที่**: ส่งข้อมูลแบบ Real-time ไปยัง IoT Dashboard

```python
class MQTTClient:
    def publish_event(self, event_data: Dict)
    def publish_statistics(self, stats: Dict)
    def publish_control(self, action: str, **details)
```

**Message Format**:
```json
{
  "timestamp": "2025-02-01T02:20:00",
  "camera_id": "CCTV_001",
  "event_type": "person_tracking",
  "data": {
    "person_id": 123,
    "tracked_count": 5,
    "demographics": {
      "gender": "Male",
      "age": 25,
      "ethnicity": "asian",
      "group": "GroupA",
      "confidence_scores": {
        "gender": 0.95,
        "age": 0.87,
        "ethnicity": 0.82
      }
    },
    "movement": {
      "status": "entered",
      "direction": "IN",
      "location": {
        "bbox": [100, 200, 300, 500],
        "center": [200, 350]
      }
    }
  },
  "label": "ID:123 | Male | Adult | Asian",
  "summary": {
    "id": 123,
    "gender": "Male",
    "age_group": "Adult",
    "region": "Asian",
    "direction": "IN",
    "status": "entered"
  }
}
```

**Topics**:
- `/cctv/people_tracking` - Events
- `/cctv/people_tracking/stats` - Statistics (retained)
- `/cctv/people_tracking/label` - Summary labels
- `/cctv/people_tracking/commands` - Control commands (subscribe)

**Commands รองรับ**:
```json
{
  "action": "reset",
  "scope": "counters" | "session" | "full"
}
```

---

### 10. `streamlit_dashboard.py` - Dashboard แบบ Web (960 บรรทัด)

**หน้าที่**: แสดงผลแบบ Real-time และ Benchmark

#### Tab 1: Real-time Analytics
- **Metrics Cards**: IN, OUT, Asia, Europe, Male, Female, Unknown
- **Bar Chart**: การกระจายข้อมูล
- **Summary Tables**: ภาพรวมทิศทาง, ภูมิภาค, เพศ
- **Event Log**: บันทึกเหตุการณ์ล่าสุด (JSON)

#### Tab 2: Benchmark Results
- **Summary Metrics** (5 การ์ดหลัก):
  - Speed (FPS %)
  - Gender Accuracy (%)
  - Ethnicity Accuracy (%)
  - mAP (YOLOv8n reference)
  - Overall Score (%)

- **Inference Speed**:
  - Preprocess, Inference, Postprocess, Total (ms)
  - FPS + Real-time status

- **Gender Accuracy**:
  - Precision, Recall, F1-Score แยกตาม class

- **Ethnicity Accuracy**:
  - Precision, Recall, F1-Score แยกตาม class

- **FairFace Per-Race Analysis**:
  - Accuracy แยกตาม race
  - Fairness Gap analysis
  - Chart visualization

**Features**:
- Dark/Light theme toggle
- Auto-refresh ทุก N วินาที
- รองรับ FastAPI backend
- Custom CSS สวยงาม

---

## 🔬 ระบบ Benchmark

### ไฟล์หลัก: `benchmark/run_benchmark.py`

**หน้าที่**: วัดประสิทธิภาพแบบครบวงจร

#### ส่วนที่ทดสอบ:

1. **Inference Speed**
   - Preprocess time (ms)
   - Inference time (ms)
   - Postprocess time (ms)
   - Total time (ms)
   - FPS

2. **Gender Classification**
   - Accuracy (%)
   - Precision, Recall, F1-Score แยกตาม class (Man/Woman)
   - Confusion Matrix

3. **Ethnicity Classification**
   - Accuracy (%)
   - Precision, Recall, F1-Score แยกตาม class
   - Per-Race Accuracy (FairFace Dataset)
   - Fairness Gap analysis

4. **Environmental Tests**
   - Normal Light accuracy
   - Low Light accuracy

### FairFace Dataset

**ขนาด**: 5,000 รูป (Medium dataset)
**รูปแบบชื่อไฟล์**: `{Race}_{Gender}_{AgeGroup}_{ID}.jpg`
**ตัวอย่าง**: `Asian_Male_20-29_001234.jpg`

**Races**: Black, White, Indian, Latino, East Asian, Southeast Asian, Middle Eastern

---

## ⚙️ วิธีการทำงานทั้งระบบ

### การเริ่มต้น (Startup Sequence)

```
1. โหลด Config
   └─> ตรวจสอบ paths, validate settings
   
2. เลือก Input Source
   └─> RTSP / Video File / Webcam
   
3. Initialize Components
   ├─> YOLO Detector (GPU + warmup)
   ├─> DeepSORT Tracker
   ├─> Face Analyzer
   ├─> Line Crossing
   ├─> Counters
   ├─> MQTT Client
   └─> Overlay Renderer
   
4. Connect MQTT
   └─> Subscribe to command topics
   
5. Start Auto-Benchmark (background thread)
   
6. Start FairFace Benchmark (background thread)
   
7. Run Main Loop
```

### Main Loop (Real-time Processing)

```python
while running:
    # 1. อ่านเฟรมจาก input
    frame = video_handler.get_frame()
    
    # 2. ประมวลผลเฟรม
    display_frame = process_frame(frame)
    # ↑ ทำ detection → tracking → face analysis → line crossing
    
    # 3. อัปเดต FPS
    fps_counter.update()
    
    # 4. Publish MQTT statistics (ทุก 5 วินาที)
    if time_elapsed > 5:
        mqtt_client.publish_statistics(stats)
    
    # 5. แสดงผล
    cv2.imshow(window, display_frame)
    
    # 6. Handle keyboard input
    key = cv2.waitKey(1)
```

### Frame Processing Pipeline

```
Input Frame (1920x1080)
    ↓
┌─────────────────────────────────────────┐
│ 1. YOLO Detection                        │
│    - Detect persons                      │
│    - Filter by confidence > 0.50         │
│    - Output: List[Detection]             │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 2. DeepSORT Tracking                     │
│    - Match detections to existing tracks │
│    - Create new tracks                   │
│    - Delete lost tracks                  │
│    - Output: List[Track]                 │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 3. Face Analysis (every 5 frames)       │
│    - Extract person ROI                  │
│    - Detect face                         │
│    - Analyze: Age, Gender, Ethnicity     │
│    - Update track.demographics           │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 4. Line Crossing Detection               │
│    - Check trajectory intersection       │
│    - Determine direction (IN/OUT)        │
│    - Validate with hysteresis            │
│    - Output: List[CrossingEvent]         │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 5. Update Counters & Send MQTT          │
│    - Update IN/OUT counts                │
│    - Update demographics stats           │
│    - Publish to MQTT                     │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 6. Render Overlay                        │
│    - Draw bounding boxes                 │
│    - Draw counting line                  │
│    - Draw statistics                     │
│    - Draw FPS                            │
│    - Output: Display Frame               │
└─────────────────────────────────────────┘
```

---

## 📊 ประสิทธิภาพและค่า Benchmark

### ความเร็ว (Inference Speed)

| Component | Time (ms) | Notes |
|-----------|-----------|-------|
| Preprocess | 2-5 ms | Resize, normalize |
| YOLO Inference | 15-25 ms | YOLOv8n on GPU |
| Postprocess | 3-7 ms | NMS, filtering |
| **Total** | **20-37 ms** | **~27-50 FPS** |
| Face Analysis | 50-100 ms | ทุก 5 เฟรม |
| Tracking | 3-8 ms | DeepSORT |

**Real-time Status**:
- ✅ Excellent: FPS ≥ 30
- ✅ Good: FPS ≥ 20
- ⚠️ Acceptable: FPS ≥ 15
- ❌ Slow: FPS < 15

### ความแม่นยำ (Accuracy)

#### Person Detection (YOLOv8n)
- **mAP50**: 37.3% (COCO dataset)
- **mAP50-95**: 24.3%

#### Gender Classification
- **Overall Accuracy**: 95-97%
- **Man**: Precision 97.0%, Recall 98.0%, F1 97.5%
- **Woman**: Precision 97.4%, Recall 96.7%, F1 97.0%

#### Ethnicity Classification
- **Overall Accuracy**: 80-85%
- **Asian**: 85-90%
- **White**: 90-95%
- **Black**: 75-80%
- **Indian**: 70-75%

**Fairness Gap**: 10-20% (ระหว่าง best และ worst performing race)

### ปัจจัยที่ส่งผลต่อความแม่นยำ

| ปัจจัย | ผลกระทบ | แนะนำ |
|--------|---------|-------|
| **แสง** | สูง | > 150 lux |
| **มุมใบหน้า** | สูงมาก | 0-30° |
| **ระยะห่าง** | สูง | < 2 เมตร |
| **ขนาดใบหน้า** | สูงมาก | > 100 pixels |
| **การปิดกั้น** | สูง | หลีกเลี่ยง mask, แว่น |

---

## 🎯 จุดเด่นและนวัตกรรม

### 1. Enhanced Tracking System
- **Adaptive Algorithms**: ปรับพารามิเตอร์ตามความหนาแน่นของฉาก
- **Velocity Prediction**: ทำนายตำแหน่งถัดไป
- **Quality Scoring**: ประเมินคุณภาพของ track
- **Track Recovery**: กู้คืน track ที่หายไปชั่วคราว

### 2. Demographics Analysis
- **FairFace Models**: ใช้โมเดลที่ออกแบบมาเพื่อความยุติธรรม
- **Stabilization**: เฉลี่ยผลหลายครั้งเพื่อความเสถียร
- **Confidence Thresholding**: กรองผลที่มี confidence ต่ำ
- **Security Groups**: จัดกลุ่มตามความต้องการด้านความปลอดภัย

### 3. Line Crossing Algorithm
- **Hysteresis**: ป้องกันการนับซ้ำ
- **Cooldown**: รอระหว่างการนับ
- **Buffer Distance**: รองรับ low FPS และ fast crossing
- **Direction Detection**: ใช้ cross product หาทิศทาง

### 4. Real-time Communication
- **MQTT Integration**: ส่งข้อมูลแบบ Real-time
- **Bidirectional Control**: รองรับคำสั่งจากภายนอก
- **Multi-topic Publishing**: Events, Stats, Labels
- **Retained Messages**: เก็บ stats ล่าสุด

### 5. Comprehensive Benchmarking
- **FairFace Integration**: ทดสอบกับ dataset มาตรฐาน
- **Per-Race Analysis**: วิเคราะห์ความยุติธรรม
- **Auto-benchmark**: รันอัตโนมัติทุก N นาที
- **History Tracking**: บันทึกประวัติ CSV

### 6. Flexible Input Modes
- **RTSP Camera**: รองรับกล้อง IP
- **Video File**: อ่านไฟล์วิดีโอ + Playback controls
- **Webcam**: ใช้กล้อง USB
- **Auto File Browser**: เลือกไฟล์ผ่าน GUI

---

## 🚀 การใช้งาน

### ติดตั้ง

```bash
# 1. สร้าง virtual environment
python -m venv .venv
.venv\Scripts\activate

# 2. ติดตั้งแพ็กเกจ
pip install -r requirements.txt

# 3. ตรวจสอบ dependencies
python main.py  # จะแสดงผล dependency check
```

### รันระบบหลัก

```bash
python main.py
```

เมนูจะให้เลือก:
1. Video File (เลือกไฟล์)
2. RTSP Camera
3. Webcam
4. Exit

### รัน Dashboard

```bash
# Streamlit Dashboard
streamlit run streamlit_dashboard.py
# → เปิด http://localhost:8501

# FastAPI Dashboard
python -m uvicorn src.web.dashboard:app --port 8080
# → เปิด http://localhost:8080
```

### รัน Benchmark

```bash
python benchmark/run_benchmark.py
```

---

## 📈 การปรับแต่งประสิทธิภาพ

### เพิ่มความเร็ว (FPS)

1. **ลด resolution**
   ```python
   INPUT_WIDTH = 640   # จาก 1920
   INPUT_HEIGHT = 480  # จาก 1080
   ```

2. **เพิ่ม process interval**
   ```python
   PROCESS_EVERY_N_FRAMES = 3  # วิเคราะห์ทุก 3 เฟรม
   PROCESS_FACE_EVERY_N_FRAMES = 10  # วิเคราะห์ใบหน้าทุก 10 เฟรม
   ```

3. **ใช้ YOLOv8n + FP16**
   ```python
   YOLO_MODEL = "yolov8n.pt"  # เร็วที่สุด
   YOLO_HALF_PRECISION = True  # เร็วขึ้น 2x
   ```

4. **ปิด face analysis**
   ```python
   FACE_ANALYSIS_ENABLED = False
   ```

### เพิ่มความแม่นยำ

1. **ใช้ YOLOv8 ที่ใหญ่กว่า**
   ```python
   YOLO_MODEL = "yolov8m.pt"  # แม่นกว่า yolov8n
   ```

2. **ลด confidence threshold**
   ```python
   YOLO_CONFIDENCE = 0.35  # จาก 0.50
   ```

3. **เพิ่มเวลาติดตาม**
   ```python
   DEEPSORT_MAX_AGE = 100  # จาก 50
   DEEPSORT_N_INIT = 5     # จาก 3
   ```

4. **ปรับ face analysis**
   ```python
   FACE_MIN_SIZE = 30              # จาก 40
   FACE_CONFIDENCE_THRESHOLD = 0.5  # จาก 0.6
   ```

---

## 🔧 การแก้ไขปัญหา

### ปัญหาที่พบบ่อย

#### 1. FPS ต่ำ

**สาเหตุ**:
- GPU ไม่ทำงาน (ใช้ CPU)
- Resolution สูงเกินไป
- Face analysis ช้า

**แก้ไข**:
```python
# ตรวจสอบ GPU
YOLO_DEVICE = "cuda"  # ต้องมี CUDA
YOLO_HALF_PRECISION = True

# ลด resolution
INPUT_WIDTH = 640
INPUT_HEIGHT = 480

# เพิ่ม interval
PROCESS_EVERY_N_FRAMES = 3
PROCESS_FACE_EVERY_N_FRAMES = 15
```

#### 2. Tracking ไม่เสถียร (ID เปลี่ยนบ่อย)

**สาเหตุ**:
- `MAX_AGE` ต่ำเกินไป
- `N_INIT` สูงเกินไป
- FPS ต่ำ

**แก้ไข**:
```python
DEEPSORT_MAX_AGE = 100      # เพิ่มจาก 50
DEEPSORT_N_INIT = 2         # ลดจาก 3
DEEPSORT_MAX_DIST = 0.4     # เพิ่มจาก 0.3
```

#### 3. Face Analysis ไม่แม่นยำ

**สาเหตุ**:
- แสงไม่พอ
- ใบหน้าเล็กเกินไป
- มุมไม่ตรง

**แก้ไข**:
- เพิ่มแสง (> 150 lux)
- ใช้กล้องที่ใกล้กว่า (< 2 เมตร)
- ปรับมุมกล้องให้ตรง

#### 4. MQTT ไม่ทำงาน

**สาเหตุ**:
- Broker ไม่พร้อม
- Network ไม่เชื่อมต่อ

**แก้ไข**:
```python
# ใช้ public broker สำหรับทดสอบ
MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883

# หรือเช็ค connection
import paho.mqtt.client as mqtt
client = mqtt.Client()
client.connect("test.mosquitto.org", 1883)
```

---

## 📚 เอกสารอ้างอิง

### ไฟล์เอกสาร

1. **README.md** - คู่มือการใช้งานหลัก
2. **RESEARCH_ACCURACY_ANALYSIS.md** - งานวิจัยความแม่นยำ
   - อธิบาย F1-Score, Confusion Matrix
   - วิเคราะห์ TP/FP/FN
   - ปัจจัยที่ส่งผลต่อความแม่นยำ
   - การปรับปรุงระบบ

### เทคโนโลยีที่ใช้

- **YOLOv8**: https://docs.ultralytics.com/
- **DeepFace**: https://github.com/serengil/deepface
- **DeepSORT**: https://github.com/nwojke/deep_sort
- **FairFace**: https://github.com/joojs/fairface

---

## ✅ สรุป

### จุดแข็ง

✅ **ครบวงจร**: Detection → Tracking → Demographics → Analytics → Visualization  
✅ **Real-time**: FPS 20-50 บน GPU  
✅ **ความแม่นยำสูง**: Gender 95%+, Ethnicity 80%+  
✅ **Flexible**: รองรับ 3 แหล่ง input (RTSP/Video/Webcam)  
✅ **IoT Ready**: MQTT integration  
✅ **Research Grade**: Comprehensive benchmarking  
✅ **User Friendly**: Interactive menu, Video playback controls  
✅ **Production Ready**: Auto-benchmark, Session logging, Error handling

### จุดอ่อน

⚠️ **แสง**: ต้องการแสงที่เพียงพอ (> 100 lux)  
⚠️ **ระยะ**: ทำงานดีที่สุดที่ < 2 เมตร  
⚠️ **มุม**: ต้องการใบหน้าตรงกล้อง (0-30°)  
⚠️ **Bias**: โมเดล biased ไปทาง White/Asian  
⚠️ **Privacy**: วิเคราะห์ลักษณะส่วนบุคคล (ต้องระวังกฎหมาย)

### การใช้งานที่เหมาะสม

1. ✅ **ระบบความปลอดภัย**: นับคนเข้า-ออกอาคาร
2. ✅ **ห้างสรรพสินค้า**: วิเคราะห์ลูกค้า
3. ✅ **สถานีขนส่ง**: ตรวจสอบความหนาแน่น
4. ✅ **โรงงาน**: ตรวจสอบพนักงาน
5. ✅ **งานวิจัย**: ศึกษาพฤติกรรมผู้คน

### การพัฒนาต่อ

🔮 **ในอนาคต**:
- Fine-tune โมเดลสำหรับสภาพแสงต่างๆ
- เพิ่ม Re-identification (ติดตามข้ามกล้อง)
- รองรับ Multi-camera
- Dashboard แสดงผลแบบ 3D
- Cloud integration

---

**เวอร์ชัน**: YOLO CV CCTV V4  
**ผู้พัฒนา**: Security System Team  
**วันที่วิเคราะห์**: 2025-02-01  
**สถานะ**: ✅ Production Ready
