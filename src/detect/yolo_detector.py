"""
YOLO Detector - Person Detection using YOLOv8
ตรวจจับบุคคลด้วย Ultralytics YOLOv8 พร้อมการปรับแต่งประสิทธิภาพ
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Optional
import time
import torch

class Detection:
    """Class to represent a detection"""
    def __init__(self, bbox: List[float], confidence: float, class_id: int):
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.confidence = confidence
        self.class_id = class_id
        
    def get_center(self) -> Tuple[int, int]:
        """Get center point of bounding box"""
        x1, y1, x2, y2 = self.bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
        
    def get_area(self) -> float:
        """Get area of bounding box"""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

class YOLODetector:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.device = self._get_device()
        self.frame_count = 0
        self.detection_count = 0
        
        # Performance tracking
        self.inference_times = []
        self.max_time_samples = 100
        
        print(f"[INFO] Initializing YOLOv8 Detector")
        print(f"[INFO] Device: {self.device}")
        print(f"[INFO] Model: {config.YOLO_MODEL}")
        
        self.load_model()
        
    def _get_device(self) -> str:
        """Determine the best device to use"""
        if self.config.YOLO_DEVICE == "cuda" and torch.cuda.is_available():
            device = f"cuda:0"
            print(f"[INFO] GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        elif self.config.YOLO_DEVICE == "mps" and torch.backends.mps.is_available():
            device = "mps"
            print("[INFO] Apple MPS detected")
        else:
            device = "cpu"
            print("[INFO] Using CPU")
            
        return device
        
    def load_model(self):
        """Load YOLOv8 model"""
        try:
            # Load model
            self.model = YOLO(self.config.YOLO_MODEL)
            
            # Move model to device
            if self.device != "cpu":
                self.model.to(self.device)
            
            # Enable half precision if supported and requested
            if self.config.YOLO_HALF_PRECISION and self.device.startswith("cuda"):
                self.model.half()
                print("[INFO] Half precision enabled")
            
            # Warm up model with dummy input
            self._warmup_model()
            
            print("[OK] YOLOv8 model loaded successfully")
            
        except Exception as e:
            print(f"[ERROR] Failed to load YOLOv8 model: {e}")
            raise
            
    def _warmup_model(self):
        """Warm up model with dummy input for better performance"""
        try:
            print("[INFO] Warming up model...")
            dummy_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # Run a few inference passes
            for _ in range(3):
                _ = self.model.predict(
                    source=dummy_input,
                    conf=self.config.YOLO_CONFIDENCE,
                    iou=self.config.YOLO_IOU_THRESHOLD,
                    classes=[0],  # Person class only
                    verbose=False
                )
            print("[OK] Model warmup completed")
            
        except Exception as e:
            print(f"[WARNING] Model warmup failed: {e}")
            
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect persons in frame
        
        Args:
            frame: Input image as numpy array
            
        Returns:
            List of Detection objects for detected persons
        """
        if self.model is None:
            return []
            
        self.frame_count += 1
        start_time = time.time()
        
        try:
            # Run YOLO inference with optimized parameters
            results = self.model.predict(
                source=frame,
                conf=self.config.YOLO_CONFIDENCE,
                iou=self.config.YOLO_IOU_THRESHOLD,
                classes=[0],  # Class 0 = person in COCO dataset
                verbose=False,
                save=False,
                show=False,
                agnostic_nms=True,  # Class-agnostic NMS
                max_det=10,         # Maximum 10 detections per image
                retina_masks=False,
                embed=None
            )
            
            # Process results
            detections = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                
                # Extract detection data
                for i in range(len(boxes)):
                    # Get bounding box coordinates (xyxy format)
                    bbox = boxes.xyxy[i].cpu().numpy().tolist()
                    
                    # ปรับปรุง bounding box ตามมาตรฐาน YOLO detection
                    x1, y1, x2, y2 = bbox
                    frame_h, frame_w = frame.shape[:2]
                    
                    # คำนวณขนาดและอัตราส่วนของ bounding box
                    width = x2 - x1
                    height = y2 - y1
                    aspect_ratio = width / height if height > 0 else 1.0
                    
                    # การขยายแบบอัจฉริยะ (Conservative Smart Expansion)
                    detection_area = width * height
                    frame_area = frame_w * frame_h
                    area_ratio = detection_area / frame_area
                    
                    # กำหนด expansion factor ที่ระมัดระวังสูงสุด
                    if area_ratio > 0.3:  # Detection ใหญ่มาก - ไม่ขยายเลย
                        expand_factor = 0.00
                    elif area_ratio > 0.2:  # Detection ใหญ่ - ขยายน้อยมาก
                        expand_factor = 0.001  # 0.1%
                    elif area_ratio > 0.1:  # Detection ปานกลางใหญ่ - ขยายน้อย
                        expand_factor = 0.003  # 0.3%
                    elif area_ratio > 0.05:  # Detection ปานกลาง - ขยายเล็กน้อย
                        expand_factor = 0.005  # 0.5%
                    elif area_ratio > 0.02:  # Detection เล็ก - ขยายปานกลาง
                        expand_factor = 0.008  # 0.8%
                    else:  # Detection เล็กมาก - ขยายน้อยลง
                        expand_factor = 0.010  # 1.0%
                    
                    # การขยายแบบเรียบง่าย (ขยายเท่าๆกันทุกทิศทาง)
                    expand_w = width * expand_factor
                    expand_h = height * expand_factor
                    
                    # ปรับปรุงขอบเขตด้วยการตรวจสอบ
                    new_x1 = max(0, x1 - expand_w)
                    new_y1 = max(0, y1 - expand_h)
                    new_x2 = min(frame_w, x2 + expand_w)
                    new_y2 = min(frame_h, y2 + expand_h)
                    
                    # ตรวจสอบความถูกต้องของขนาด
                    final_width = new_x2 - new_x1
                    final_height = new_y2 - new_y1
                    
                    # ตรวจสอบขนาดต่ำสุดและสูงสุด
                    if final_width < 20 or final_height < 40:  # ขนาดต่ำสุด
                        continue  # ข้าม detection นี้
                        
                    # ตรวจสอบขนาดสูงสุด - ป้องกัน bbox ที่ใหญ่เกินไป (ลดขนาดลงมาก)
                    max_width = frame_w * 0.3    # ไม่เกิน 30% ของความกว้างจอ (576px)
                    max_height = frame_h * 0.4   # ไม่เกิน 40% ของความสูงจอ (432px)
                    
                    if final_width > max_width or final_height > max_height:
                        # ลด bbox ให้อยู่ในขอบเขตที่เหมาะสม
                        if final_width > max_width:
                            center_x = (new_x1 + new_x2) / 2
                            new_x1 = center_x - max_width / 2
                            new_x2 = center_x + max_width / 2
                            new_x1 = max(0, new_x1)
                            new_x2 = min(frame_w, new_x2)
                            
                        if final_height > max_height:
                            center_y = (new_y1 + new_y2) / 2
                            new_y1 = center_y - max_height / 2
                            new_y2 = center_y + max_height / 2
                            new_y1 = max(0, new_y1)
                            new_y2 = min(frame_h, new_y2)
                        
                        # อัปเดตขนาดหลังจากการปรับแต่ง
                        final_width = new_x2 - new_x1
                        final_height = new_y2 - new_y1
                        
                    bbox = [new_x1, new_y1, new_x2, new_y2]
                    
                    # Get confidence score
                    confidence = float(boxes.conf[i].cpu().numpy())
                    
                    # Get class ID (should be 0 for person)
                    class_id = int(boxes.cls[i].cpu().numpy())
                    
                    # Enhanced filtering criteria - ลดเกณฑ์เพื่อจับ detection ได้มากขึ้น
                    if (confidence >= 0.25 and  # ลดจาก YOLO_CONFIDENCE เป็น 0.25
                        class_id == 0 and 
                        final_width >= 20 and final_height >= 40):  # ลดขนาดขั้นต่ำ
                        
                        # คำนวณ aspect ratio สำหรับการกรอง
                        final_aspect_ratio = final_width / final_height
                        
                        # ขยาย aspect ratio range เพื่อรองรับท่าทางต่างๆ (0.15 - 1.5)
                        if 0.15 <= final_aspect_ratio <= 1.5:
                            # เพิ่ม quality score based on size and aspect ratio
                            size_score = min(1.0, (final_width * final_height) / (100 * 200))  # Normalize
                            aspect_score = 1.0 - abs(final_aspect_ratio - 0.5) / 0.7  # Optimal around 0.5
                            quality_score = (confidence + size_score + aspect_score) / 3.0
                            
                            detection = Detection(bbox, quality_score, class_id)
                            detections.append(detection)
                        
            self.detection_count += len(detections)
            
            # Track inference time
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            if len(self.inference_times) > self.max_time_samples:
                self.inference_times.pop(0)
                
            return detections
            
        except Exception as e:
            print(f"[ERROR] Detection error: {e}")
            return []
            
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """
        Detect persons in multiple frames (batch processing)
        
        Args:
            frames: List of input images
            
        Returns:
            List of detection lists for each frame
        """
        if self.model is None or not frames:
            return []
            
        try:
            # Run batch inference
            results = self.model.predict(
                source=frames,
                conf=self.config.YOLO_CONFIDENCE,
                iou=self.config.YOLO_IOU_THRESHOLD,
                classes=[0],  # Person class only
                verbose=False,
                save=False,
                show=False
            )
            
            # Process results for each frame
            all_detections = []
            for result in results:
                frame_detections = []
                if result.boxes is not None:
                    boxes = result.boxes
                    
                    for i in range(len(boxes)):
                        bbox = boxes.xyxy[i].cpu().numpy().tolist()
                        confidence = float(boxes.conf[i].cpu().numpy())
                        class_id = int(boxes.cls[i].cpu().numpy())
                        
                        if confidence >= self.config.YOLO_CONFIDENCE and class_id == 0:
                            detection = Detection(bbox, confidence, class_id)
                            frame_detections.append(detection)
                            
                all_detections.append(frame_detections)
                
            return all_detections
            
        except Exception as e:
            print(f"[ERROR] Batch detection error: {e}")
            return [[] for _ in frames]
            
    def get_statistics(self) -> dict:
        """Get detector performance statistics"""
        avg_inference_time = 0
        if self.inference_times:
            avg_inference_time = sum(self.inference_times) / len(self.inference_times)
            
        return {
            "frames_processed": self.frame_count,
            "total_detections": self.detection_count,
            "avg_detections_per_frame": self.detection_count / max(self.frame_count, 1),
            "avg_inference_time": avg_inference_time,
            "avg_fps": 1.0 / avg_inference_time if avg_inference_time > 0 else 0,
            "device": self.device,
            "model_path": self.config.YOLO_MODEL,
            "confidence_threshold": self.config.YOLO_CONFIDENCE
        }
        
    def update_confidence_threshold(self, new_threshold: float):
        """Update confidence threshold dynamically"""
        if 0.0 <= new_threshold <= 1.0:
            self.config.YOLO_CONFIDENCE = new_threshold
            print(f"[INFO] Updated confidence threshold to {new_threshold}")
        else:
            print(f"[WARNING] Invalid confidence threshold: {new_threshold}")
            
    def visualize_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """
        Draw detection bounding boxes on frame for debugging
        
        Args:
            frame: Input image
            detections: List of detections to draw
            
        Returns:
            Frame with drawn bounding boxes
        """
        result_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = [int(coord) for coord in detection.bbox]
            
            # Draw bounding box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence score
            conf_text = f"Person: {detection.confidence:.2f}"
            cv2.putText(result_frame, conf_text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                       
        return result_frame
        
    def release(self):
        """Release resources"""
        if self.model is not None:
            # Clear CUDA cache if using GPU
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
            
            self.model = None
            print("[INFO] YOLOv8 detector resources released")
            
    def __del__(self):
        """Destructor"""
        self.release()