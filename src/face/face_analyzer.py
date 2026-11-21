"""
Face Analyzer - Demographics + Ethnicity Classification
วิเคราะห์อายุ, เพศ, เชื้อชาติด้วย DeepFace (ไม่ใช้ InsightFace)
"""

import cv2
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import time
import threading
from collections import defaultdict, deque

# Try to import face analysis libraries
DEEPFACE_AVAILABLE = False
FACE_RECOGNITION_AVAILABLE = False

# Try DeepFace first
try:
    # -------------------------------------------------------------
    # GPU Memory Fixes for Limited VRAM (e.g., 4GB RTX 3050)
    # -------------------------------------------------------------
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress messy TF logs
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'  # Fix fragmentation
    
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("[OK] FaceAnalyzer: TF GPU memory growth & async allocator enabled")
        except RuntimeError as e:
            pass
    # -------------------------------------------------------------

    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    print("[OK] DeepFace available")
except ImportError:
    print("[WARNING] DeepFace not available")

# Try face_recognition as fallback
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
    print("[OK] face_recognition available")
except ImportError:
    print("[WARNING] face_recognition not available")

if not DEEPFACE_AVAILABLE and not FACE_RECOGNITION_AVAILABLE:
    print("[WARNING] No face analysis libraries available - face analysis will be disabled")
    print("[INFO] To enable real face analysis, install either:")
    print("   - DeepFace: pip install deepface")
    print("   - face_recognition: pip install face_recognition")

class FaceAnalysisResult:
    """Class to hold face analysis results"""
    def __init__(self):
        self.age = None
        self.gender = None
        self.ethnicity = None
        self.group = None
        
        # Confidence scores
        self.age_confidence = 0.0
        self.gender_confidence = 0.0
        self.ethnicity_confidence = 0.0
        
        # แยกเก็บค่า confidence ของแต่ละเพศ
        self.man_confidence = 0.0
        self.woman_confidence = 0.0
        
        # Face detection info
        self.face_bbox = None
        self.face_detected = False
        
    def is_valid(self) -> bool:
        """Check if result has valid data"""
        return self.face_detected and self.gender is not None

class FaceAnalyzer:
    def __init__(self, config):
        self.config = config
        self.face_detection_enabled = True
        self.demographics_enabled = True
        self.ethnicity_enabled = config.ETHNICITY_ENABLED
        
        # Face analysis cache to stabilize results
        self.analysis_cache = {}  # track_id -> deque of results
        self.cache_size = 3  # ลดเป็น 3 เพื่อให้ update เร็วขึ้นและแม่นยำขึ้น
        
        # Performance tracking
        self.analysis_count = 0
        self.successful_analysis = 0
        self.cache_hits = 0
        self.error_count = 0          # จำนวนครั้งที่ DeepFace error (ไม่นับกรณีไม่เจอหน้า)
        self._reported_errors = set()  # ชนิด error ที่แจ้งไปแล้ว
        
        # Threading for async analysis
        self.analysis_lock = threading.Lock()
        
        print(f"[INFO] Face Analyzer initialized")
        print(f"[INFO] Demographics enabled: {self.demographics_enabled}")
        print(f"[INFO] Ethnicity enabled: {self.ethnicity_enabled}")
        
        if DEEPFACE_AVAILABLE:
            self._warmup_models()
        elif getattr(config, 'FACE_ALLOW_MOCK', False):
            print("[WARNING] DeepFace not available - FACE_ALLOW_MOCK is on: demographics are RANDOM test data")
        else:
            print("[ERROR] DeepFace not available - face analysis DISABLED (no age/gender/ethnicity). "
                  "Install with: pip install deepface tf-keras")
            
    def _warmup_models(self):
        """Warm up DeepFace models"""
        try:
            print("[INFO] Warming up DeepFace models...")
            
            # Create a small test image
            test_img = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
            
            # Test face analysis (this will download models if needed)
            actions = ['age', 'gender']
            if self.ethnicity_enabled:
                actions.append('race')
            
            model_name = getattr(self.config, 'DEEPFACE_MODEL', 'Facenet512')
            DeepFace.analyze(
                img_path=test_img, 
                actions=actions,
                model_name=model_name,
                enforce_detection=False,
                silent=True
            )
            print("[OK] DeepFace models warmed up")
            
        except Exception as e:
            print(f"[WARNING] DeepFace warmup failed (will work on first real analysis): {e}")
            
    def analyze(self, person_roi: np.ndarray, track_id: int) -> Optional[FaceAnalysisResult]:
        """
        Analyze face in person ROI
        
        Args:
            person_roi: Region of interest containing person
            track_id: Track ID for caching
            
        Returns:
            FaceAnalysisResult or None if no face found
        """
        if not self.demographics_enabled:
            return None
            
        # ไม่มี DeepFace = วิเคราะห์ไม่ได้ ห้ามส่งผลสุ่มเข้าสถิติจริง
        # (เปิด FACE_ALLOW_MOCK = True ใน config เฉพาะตอนทดสอบ UI เท่านั้น)
        if not DEEPFACE_AVAILABLE:
            if getattr(self.config, 'FACE_ALLOW_MOCK', False):
                self.analysis_count += 1
                self.successful_analysis += 1
                return self._mock_analysis()
            return None
            
        self.analysis_count += 1
        
        try:
            # Detect and analyze face
            result = self._analyze_with_deepface(person_roi, track_id)
            
            if result and result.is_valid():
                self.successful_analysis += 1
                
                # Add to cache for stability
                self._update_cache(track_id, result)
                
                # Return stabilized result
                return self._get_stabilized_result(track_id)
            
            return None
            
        except Exception as e:
            print(f"[ERROR] Face analysis error: {e}")
            return None
            
    def _analyze_with_deepface(self, person_roi: np.ndarray, track_id: int) -> Optional[FaceAnalysisResult]:
        """Analyze face using DeepFace"""
        
        # Minimum face size check
        if person_roi.shape[0] < self.config.FACE_MIN_SIZE or person_roi.shape[1] < self.config.FACE_MIN_SIZE:
            return None
        
        # ตัดเฉพาะส่วนบน 60% ของ ROI (ส่วนที่มีใบหน้า)
        h, w = person_roi.shape[:2]
        face_region = person_roi[0:int(h * 0.6), :]  # เอาแค่ส่วนบน 60%
        
        # ถ้าส่วนหน้าเล็กเกินไป ใช้ ROI เดิม
        if face_region.shape[0] < 30 or face_region.shape[1] < 30:
            face_region = person_roi
        
        # Resize to 224x224 ก่อนส่งให้ DeepFace (เซฟ VRAM, ทำงานเร็วขึ้น)
        # DeepFace ก็จะย่อขนาดนี้อยู่แล้ว ให้ CPU ทำแทนเพื่อลดภาระ GPU
        face_region = cv2.resize(face_region, (224, 224))
            
        try:
            # Prepare analysis actions
            actions = ['age', 'gender']
            if self.ethnicity_enabled:
                actions.append('race')
            
            # รับ backend และ model จาก config
            detector_backend = getattr(self.config, 'DEEPFACE_BACKEND', 'retinaface')
            model_name = getattr(self.config, 'DEEPFACE_MODEL', 'Facenet512')
            
            # ลอง enforce_detection=True พร้อม retinaface + Facenet512 (แม่นที่สุด)
            try:
                analysis_result = DeepFace.analyze(
                    img_path=face_region,
                    actions=actions,
                    model_name=model_name,
                    enforce_detection=True,  # บังคับตรวจจับใบหน้าก่อน
                    detector_backend=detector_backend,
                    silent=True
                )
            except ValueError:
                # DeepFace โยน ValueError เมื่อไม่เจอใบหน้า -> skip ปกติ (ไม่ทำ fallback มั่ว)
                return None
            except Exception as e:
                # error อื่น (เช่น GPU หน่วยความจำเต็ม, โหลดโมเดลไม่ได้) ต้องแสดงให้เห็น
                # แจ้งแค่ครั้งแรกของแต่ละชนิด error เพื่อไม่ให้ terminal รก
                err_key = type(e).__name__
                if err_key not in self._reported_errors:
                    self._reported_errors.add(err_key)
                    print(f"[ERROR] DeepFace analyze failed ({err_key}): {e}")
                self.error_count += 1
                return None
            
            # DeepFace returns a list, take first result
            if isinstance(analysis_result, list):
                analysis_result = analysis_result[0]
                
            # Create our result object
            result = FaceAnalysisResult()
            
            # Extract age
            if 'age' in analysis_result:
                result.age = int(analysis_result['age'])
                result.age_confidence = 0.8  # DeepFace doesn't provide age confidence
                
            # Extract gender
            if 'gender' in analysis_result:
                gender_data = analysis_result['gender']
                if isinstance(gender_data, dict):
                    # เก็บค่า confidence ของทั้งสองเพศ
                    result.man_confidence = gender_data.get('Man', 0) / 100.0
                    result.woman_confidence = gender_data.get('Woman', 0) / 100.0
                    
                    # Find gender with highest confidence
                    max_gender = max(gender_data, key=gender_data.get)
                    confidence = gender_data[max_gender] / 100.0  # Convert percentage
                    
                    # Only accept if confidence is high enough
                    gender_threshold = getattr(self.config, 'GENDER_CONFIDENCE_THRESHOLD', 0.6)
                    if confidence >= gender_threshold:
                        result.gender = max_gender
                        result.gender_confidence = confidence
                    else:
                        result.gender = None  # Reject low confidence prediction
                        result.gender_confidence = 0.0
                else:
                    result.gender = str(gender_data)
                    result.gender_confidence = 0.8
                    
            # Extract ethnicity/race
            if self.ethnicity_enabled and 'race' in analysis_result:
                race_data = analysis_result['race']
                if isinstance(race_data, dict):
                    # Find race with highest confidence
                    max_race = max(race_data, key=race_data.get)
                    confidence = race_data[max_race] / 100.0
                    
                    # กรอง ethnicity ที่ไม่มั่นใจออก (ป้องกันเดามั่ว)
                    ethnicity_threshold = getattr(self.config, 'ETHNICITY_CONFIDENCE_THRESHOLD', 0.35)
                    if confidence < ethnicity_threshold:
                        result.ethnicity = None  # ไม่มั่นใจ ไม่ทาย
                        result.ethnicity_confidence = 0.0
                    else:
                        # ทำความสะอาดชื่อเชื้อชาติ
                        if 'white' in max_race.lower():
                            result.ethnicity = 'White'
                        elif 'asian' in max_race.lower():
                            result.ethnicity = 'Asian'
                        elif 'black' in max_race.lower():
                            result.ethnicity = 'Black'
                        elif 'indian' in max_race.lower():
                            result.ethnicity = 'Indian'
                        else:
                            result.ethnicity = max_race.title()
                        
                        result.ethnicity_confidence = confidence
                    
                    # Map to security groups
                    result.group = self._map_ethnicity_to_group(result.ethnicity)
                else:
                    result.ethnicity = str(race_data).title()
                    result.ethnicity_confidence = 0.8
                    result.group = self._map_ethnicity_to_group(result.ethnicity)
                    
            # Check if we have a face region
            if 'region' in analysis_result:
                region = analysis_result['region']
                result.face_bbox = [region['x'], region['y'], 
                                  region['x'] + region['w'], region['y'] + region['h']]
                result.face_detected = True
            else:
                result.face_detected = True  # Assume detected if analysis succeeded
                
            return result
            
        except Exception as e:
            # แสดง error เฉพาะครั้งแรก
            if not hasattr(self, '_error_shown'):
                print(f"[WARNING] DeepFace error: {e}")
                self._error_shown = True
            return None
            
    def _map_ethnicity_to_group(self, ethnicity: str) -> str:
        """Map ethnicity to security group"""
        if not ethnicity:
            return "Unknown"
            
        ethnicity_lower = ethnicity.lower()
        
        # Map based on config
        for ethnic_key, group in self.config.ETHNICITY_GROUPS.items():
            if ethnic_key.lower() in ethnicity_lower or ethnicity_lower in ethnic_key.lower():
                return group
                
        # Default mappings for common DeepFace outputs
        ethnicity_mapping = {
            'asian': 'GroupA',
            'white': 'GroupB',
            'black or african american': 'GroupC',
            'black': 'GroupC',
            'indian': 'GroupD',
            'middle eastern': 'GroupE',
            'latino hispanic': 'GroupE'
        }
        
        return ethnicity_mapping.get(ethnicity_lower, 'Unknown')
        
    def _update_cache(self, track_id: int, result: FaceAnalysisResult):
        """Update analysis cache for track stability"""
        with self.analysis_lock:
            if track_id not in self.analysis_cache:
                self.analysis_cache[track_id] = deque(maxlen=self.cache_size)
                
            self.analysis_cache[track_id].append(result)
            
    def _get_stabilized_result(self, track_id: int) -> Optional[FaceAnalysisResult]:
        """Get stabilized result from cache"""
        with self.analysis_lock:
            if track_id not in self.analysis_cache or len(self.analysis_cache[track_id]) == 0:
                return None
                
            cache = self.analysis_cache[track_id]
            
            # Create stabilized result
            stabilized = FaceAnalysisResult()
            
            # Average age
            ages = [r.age for r in cache if r.age is not None]
            if ages:
                stabilized.age = int(np.mean(ages))
                stabilized.age_confidence = np.mean([r.age_confidence for r in cache if r.age_confidence])
                
            # Most common gender
            genders = [r.gender for r in cache if r.gender is not None]
            if genders:
                gender_counts = defaultdict(int)
                gender_confidences = defaultdict(list)
                
                for i, gender in enumerate(genders):
                    gender_counts[gender] += 1
                    gender_confidences[gender].append(cache[i].gender_confidence)
                    
                # Choose most frequent gender
                stabilized.gender = max(gender_counts, key=gender_counts.get)
                stabilized.gender_confidence = np.mean(gender_confidences[stabilized.gender])
                
            # Most common ethnicity
            if self.ethnicity_enabled:
                ethnicities = [r.ethnicity for r in cache if r.ethnicity is not None]
                if ethnicities:
                    ethnicity_counts = defaultdict(int)
                    ethnicity_confidences = defaultdict(list)
                    
                    for i, ethnicity in enumerate(ethnicities):
                        if cache[i].ethnicity:  # Only count non-None values
                            ethnicity_counts[ethnicity] += 1
                            ethnicity_confidences[ethnicity].append(cache[i].ethnicity_confidence)
                            
                    if ethnicity_counts:
                        # Choose most frequent ethnicity
                        stabilized.ethnicity = max(ethnicity_counts, key=ethnicity_counts.get)
                        stabilized.ethnicity_confidence = np.mean(ethnicity_confidences[stabilized.ethnicity])
                        stabilized.group = self._map_ethnicity_to_group(stabilized.ethnicity)
                        
            # Use latest face detection info
            latest = cache[-1]
            stabilized.face_detected = latest.face_detected
            stabilized.face_bbox = latest.face_bbox
            
            self.cache_hits += 1
            return stabilized
            
    def _mock_analysis(self) -> FaceAnalysisResult:
        """Mock analysis for testing without DeepFace"""
        result = FaceAnalysisResult()
        
        # Generate mock data
        result.age = np.random.randint(20, 60)
        result.gender = np.random.choice(['Male', 'Female'])
        result.ethnicity = np.random.choice(['Asian', 'European'])
        result.group = self._map_ethnicity_to_group(result.ethnicity)
        
        # Mock confidences
        result.age_confidence = 0.8
        result.gender_confidence = 0.9
        result.ethnicity_confidence = 0.7
        result.face_detected = True
        
        return result
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get analyzer statistics"""
        success_rate = (self.successful_analysis / max(self.analysis_count, 1)) * 100
        
        return {
            "analysis_count": self.analysis_count,
            "successful_analysis": self.successful_analysis,
            "success_rate": success_rate,
            "cache_hits": self.cache_hits,
            "error_count": self.error_count,
            "cached_tracks": len(self.analysis_cache),
            "deepface_available": DEEPFACE_AVAILABLE,
            "demographics_enabled": self.demographics_enabled,
            "ethnicity_enabled": self.ethnicity_enabled,
            "face_min_size": self.config.FACE_MIN_SIZE
        }
        
    def clear_cache(self, track_id: Optional[int] = None):
        """Clear analysis cache"""
        with self.analysis_lock:
            if track_id is not None:
                self.analysis_cache.pop(track_id, None)
            else:
                self.analysis_cache.clear()
                
    def cleanup_old_cache(self, active_track_ids: List[int]):
        """Remove cache entries for inactive tracks"""
        with self.analysis_lock:
            # Remove tracks that are no longer active
            inactive_tracks = set(self.analysis_cache.keys()) - set(active_track_ids)
            for track_id in inactive_tracks:
                self.analysis_cache.pop(track_id, None)
                
    def update_config(self, new_config):
        """Update configuration dynamically"""
        self.config = new_config
        self.ethnicity_enabled = new_config.ETHNICITY_ENABLED
        print(f"[INFO] Face analyzer config updated - Ethnicity: {self.ethnicity_enabled}")
        
    def set_demographics_enabled(self, enabled: bool):
        """Enable/disable demographics analysis"""
        self.demographics_enabled = enabled
        print(f"[INFO] Demographics analysis: {'enabled' if enabled else 'disabled'}")
        
    def set_ethnicity_enabled(self, enabled: bool):
        """Enable/disable ethnicity analysis"""
        self.ethnicity_enabled = enabled and self.config.ETHNICITY_ENABLED
        print(f"[INFO] Ethnicity analysis: {'enabled' if self.ethnicity_enabled else 'disabled'}")
        
    def __del__(self):
        """Destructor"""
        self.analysis_cache.clear()