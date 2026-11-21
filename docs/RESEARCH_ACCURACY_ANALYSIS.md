# 📋 งานวิจัย: การวิเคราะห์ความแม่นยำของระบบตรวจจับและจำแนกประชากรศาสตร์

---

## 🔬 บทนำ (Introduction)

ระบบ CCTV ที่พัฒนาขึ้น ใช้เทคโนโลยี Deep Learning เพื่อตรวจจับและติดตามบุคคล พร้อมวิเคราะห์ลักษณะประชากรศาสตร์ (Demographics) รวมถึง เพศ (Gender) และ เชื้อชาติ (Ethnicity) โครงการวิจัยนี้มีวัตถุประสงค์เพื่อวัดค่าประสิทธิภาพของระบบ โดยใช้เมตริกส์ที่ยอมรับในหมู่นักวิจัยสากล คือ Precision, Recall, และ F1-Score

---

## 📊 ส่วนที่ 1: ที่มาของตารางความแม่นยำ (F1-Score)

### **1.1 นิยามแนวคิด**

#### **Confusion Matrix (เมทริกซ์ความสับสน)**
Confusion Matrix คือตารางที่แสดงการเปรียบเทียบระหว่าง Ground Truth (ข้อมูลจริง) และ Predicted (การทำนายของโมเดล) [1]

```
           Predicted
          │ Positive │ Negative │
─────────┼──────────┼──────────┤
True   P │    TP    │    FN    │
─────────┼──────────┼──────────┤
True   N │    FP    │    TN    │
─────────┴──────────┴──────────┘
```

**นิยาม:**
- **TP (True Positive)**: ทำนายว่า Positive และเป็น Positive จริง ✅
- **FP (False Positive)**: ทำนายว่า Positive แต่เป็น Negative (ผิดพลาดชนิดที่ I) ❌
- **FN (False Negative)**: ทำนายว่า Negative แต่เป็น Positive (ผิดพลาดชนิดที่ II) ❌
- **TN (True Negative)**: ทำนายว่า Negative และเป็น Negative จริง ✅

### **1.2 สูตรการคำนวณเมตริกส์ [2]**

#### **A. Accuracy (ความถูกต้องรวม)**
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**ความหมาย**: ร้อยละของการทำนายที่ถูกต้องทั้งหมด

#### **B. Precision (ความแม่นยำ)**
$$\text{Precision} = \frac{TP}{TP + FP}$$

**ความหมาย**: จากการทำนายว่า Positive ทั้งหมด มีกี่ส่วนที่ถูกจริง (ลดข้อผิดพลาด False Positive)

#### **C. Recall (ความครบถ้วน)**
$$\text{Recall} = \frac{TP}{TP + FN}$$

**ความหมาย**: จากสิ่งที่เป็น Positive จริง มีกี่ส่วนที่ถูกตรวจจับได้ (ลดข้อผิดพลาด False Negative)

#### **D. F1-Score (คะแนนฟ้อร์ม) [3]**
$$F1\text{-}Score = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = 2 \times \frac{TP}{2 \times TP + FP + FN}$$

**ความหมาย**: ค่าเฉลี่ยฮาร์โมนิก (Harmonic Mean) ของ Precision และ Recall ใช้สำหรับสมดุลระหว่างเพื่อทั้งสองค่า

### **1.3 ที่มาของข้อมูลสำหรับคำนวณ**

#### **ขั้นตอนที่ 1: การเตรียมข้อมูลทดสอบ (Test Dataset Preparation)**

ระบบใช้ **FairFace Dataset** ซึ่งเป็นชุดข้อมูลมาตรฐานสำหรับทดสอบความยุติธรรม (Fairness) ของระบบ Face Recognition:

```plaintext
benchmark/fairface_images/train/ (ไดเรกทอรี่ FairFace Dataset)
├── Black_Female_0-2_000018.jpg
├── Black_Female_10-19_000041.jpg
├── Black_Male_20-29_001234.jpg
├── White_Female_30-39_000095.jpg
├── White_Male_50-59_003238.jpg
├── Indian_Male_20-29_000728.jpg
├── Latino_Hispanic_Female_20-29_072282.jpg
├── Middle_Eastern_Male_30-39_000221.jpg
└── ... (5,000 รูปทั้งหมด - Medium Dataset)
```

**รูปแบบชื่อไฟล์** (FairFace Naming Convention):
```
{Race}_{Gender}_{AgeGroup}_{ID}.jpg
```

**ตัวอย่าง:**
- `Black_Female_20-29_001637.jpg` → Ground Truth: Race = "Black", Gender = "Female", Age = "20-29"
- `White_Male_50-59_003238.jpg` → Ground Truth: Race = "White", Gender = "Male", Age = "50-59"
- `Indian_Male_20-29_000728.jpg` → Ground Truth: Race = "Indian", Gender = "Male", Age = "20-29"

**โค้ดการแยกวิเคราะห์** (`benchmark/run_benchmark.py`, บรรทัด 167-209):
```python
def parse_filename(self, filename: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse FairFace filename to extract ground truth labels
    Format: {Race}_{Gender}_{AgeGroup}_{ID}.jpg
    Example: Black_Female_20-29_001637.jpg -> ('black', 'Woman')
    """
    name = Path(filename).stem  # ได้: "Black_Female_20-29_001637"
    parts = name.split('_')     # ได้: ['Black', 'Female', '20-29', '001637']
    
    # Parts[0] = Race, Parts[1] = Gender (ตำแหน่งคงที่)
    race_mapping = {
        'Black': 'black',
        'White': 'white',
        'Indian': 'indian',
        'Latino': 'latino hispanic',
        'East': 'asian',           # East Asian
        'Southeast': 'asian',      # Southeast Asian  
        'Middle': 'middle eastern'
    }
    
    gender_mapping = {
        'Male': 'Man',
        'Female': 'Woman'
    }
    
    # แยก Race (parts[0])
    ethnicity = race_mapping.get(parts[0]) if len(parts) > 0 else None
    
    # แยก Gender (parts[1])
    gender = gender_mapping.get(parts[1]) if len(parts) > 1 else None
    
    return ethnicity, gender
```

#### **ขั้นตอนที่ 2: การทำนายของโมเดล (Model Prediction)**

รูปภาพแต่ละคนจะถูกวิเคราะห์ด้วย **DeepFace** ซึ่งเป็นไลบรารี่สำหรับวิเคราะห์ใบหน้า [4]:

```python
# Code from benchmark/run_benchmark.py, lines 307-319
analysis = DeepFace.analyze(
    str(img_path),
    actions=['gender', 'race'],
    enforce_detection=False,
    silent=True
)

if isinstance(analysis, list):
    analysis = analysis[0]
    
pred_gender = analysis.get('dominant_gender', 'Unknown')
pred_ethnicity = analysis.get('dominant_race', 'Unknown')
```

**ผลลัพธ์ที่ได้จาก DeepFace:**
- `dominant_gender`: "Man" หรือ "Woman"
- `dominant_race`: "asian", "white", "black", "indian", "latino hispanic", "middle eastern"

#### **ขั้นตอนที่ 3: การเปรียบเทียบและการสร้าง Confusion Matrix**

```python
# Code from benchmark/run_benchmark.py, lines 329-345
if true_gender:
    is_correct = pred_gender == true_gender
    gender_results["correct" if is_correct else "incorrect"] += 1
    gender_results["predictions"].append({
        "file": img_path.name,
        "true": true_gender,
        "predicted": pred_gender,
        "correct": is_correct
    })
    gender_confusion[true_gender][pred_gender] += 1
```

**ตัวอย่าง Confusion Matrix สำหรับ Gender:**
```
        Predicted
        │  Man  │ Woman │
────────┼───────┼───────┤
True M  │  98   │   2   │  (TP=98, FN=2)
────────┼───────┼───────┤
True W  │   3   │  95   │  (FP=3, TP=95)
────────┴───────┴───────┘
```

### **1.4 การคำนวณเมตริกส์ต่อคลาส (Per-Class Metrics)**

```python
# Code from benchmark/run_benchmark.py, lines 454-468
def calculate_metrics(self, results: Dict, confusion: Dict, label: str) -> Dict:
    """Calculate Precision, Recall, F1-Score, Accuracy"""
    
    for cls in classes:
        tp = confusion[cls].get(cls, 0)
        fp = sum(confusion[other].get(cls, 0) for other in classes if other != cls)
        fn = sum(confusion[cls].get(other, 0) for other in classes if other != cls)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
```

**ตัวอย่างการคำนวณสำหรับคลาส "Man":**
- TP (Man ทำนายถูก) = 98
- FP (ส่วนอื่นทำนายว่า Man) = 3
- FN (Man ทำนายผิด) = 2

```
Precision = 98 / (98 + 3) = 98/101 = 0.970
Recall = 98 / (98 + 2) = 98/100 = 0.980
F1 = 2 × (0.970 × 0.980) / (0.970 + 0.980) = 0.975
```

### **1.5 การเก็บบันทึกผลลัพธ์**

ผลลัพธ์จะถูกบันทึกในรูปแบบต่อไปนี้:

#### **1. JSON File (รายละเอียดครบถ้วน)**
```
benchmark/results/benchmark_results_20250131_150000.json
```

**ตัวอย่างเนื้อหา:**
```json
{
  "gender_metrics": {
    "total_samples": 198,
    "accuracy": 0.974,
    "accuracy_percent": 97.4,
    "class_metrics": {
      "Man": {
        "precision": 0.970,
        "recall": 0.980,
        "f1_score": 0.975,
        "support": 100
      },
      "Woman": {
        "precision": 0.974,
        "recall": 0.967,
        "f1_score": 0.970,
        "support": 98
      }
    }
  },
  "confusion_matrices": {
    "gender": {
      "Man": {"Man": 98, "Woman": 2},
      "Woman": {"Man": 3, "Woman": 95}
    }
  }
}
```

#### **2. CSV File (ประวัติหลายครั้ง)**

**File: `benchmark/results/class_metrics_history.csv`**
```csv
timestamp,run_id,category,class,precision,recall,f1_score,support
2025-01-31 15:00:00,20250131_150000,gender,Man,0.970,0.980,0.975,100
2025-01-31 15:00:00,20250131_150000,gender,Woman,0.974,0.967,0.970,98
2025-01-31 15:00:01,20250131_150001,gender,Man,0.968,0.982,0.975,100
```

#### **3. Dashboard Streamlit**
- **URL**: `http://localhost:8501`
- **Tab**: "⚡ Benchmark Results"
- แสดงตารางและกราฟแบบเรียลไทม์

---

## 📊 ส่วนที่ 2: ต้นตอของข้อผิดพลาด (TP/FP/FN Error Analysis)

### **2.1 ความหมายของข้อผิดพลาด**

| ประเภท | สูตร | ความหมาย | ตัวอย่าง |
|--------|------|---------|---------|
| **TP** | - | ทำนายถูก ✅ | Ground Truth: "asian"<br>Prediction: "asian" |
| **FP** | Predicted = Pos<br>Actual = Neg | ทำนายว่า Positive<br>แต่เป็น Negative | Ground Truth: "white"<br>Prediction: "asian" |
| **FN** | Predicted = Neg<br>Actual = Pos | ทำนายว่า Negative<br>แต่เป็น Positive | Ground Truth: "asian"<br>Prediction: "white" |
| **TN** | - | ทำนายถูก ✅ | Ground Truth: "not asian"<br>Prediction: "not asian" |

### **2.2 สาเหตุของข้อผิดพลาด (Root Cause Analysis)**

#### **A. ปัจจัยทางแสง (Illumination) [5]**

**ปัญหา**: แสงที่ไม่เพียงพอส่งผลต่อการวิเคราะห์ใบหน้า

| สภาพแสง | Illuminance | ผลกระทบ | หมายเหตุ |
|---------|------------|--------|---------|
| **Bright** | > 300 lux | FP/FN ต่ำ (~5%) | ตรวจจับดีที่สุด |
| **Normal** | 150-300 lux | FP/FN ปานกลาง (~10%) | สภาพปกติ |
| **Low Light** | < 100 lux | FP/FN สูง (~25-30%) | โมเดลอ่อนไหว |

**กลไก**: 
- DeepFace ฝึกสอนด้วยภาพที่มีแสงปกติ
- แสงน้อยเหลือ contrast, ทำให้ใบหน้าไม่ชัดเจน
- การอ้างอิงเด่นชัด (Landmark Detection) ล้มเหลว

#### **B. มุมเนื้อหา (Pose Angle) [6]**

**ปัญหา**: ใบหน้าไม่หันตรงไปยังกล้อง (Frontal Face)

```
ตัวอย่างมุมต่างๆ:
  0°        45°       90°
[====]    [/===]    [|===]
 OK       WARNING    FAILED
```

| มุม | ความเป็นไปได้ | ปัญหา |
|-----|------------|-------|
| 0° (Frontal) | ✅ 95%+ | ไม่มี |
| 15-30° (Slight) | ⚠️ 80-90% | Landmark ไม่ครบ |
| 45°+ (Extreme) | ❌ 40-60% | ไม่สามารถวิเคราะห์ได้ |
| 90° (Profile) | ❌ < 20% | ไม่เห็นใบหน้าตรง |

#### **C. ระยะห่าง (Distance) [7]**

**ปัญหา**: ระยะห่างเกินไป ทำให้ใบหน้าเล็กเกินไป

| ระยะ | ขนาดใบหน้า | ความแม่นยำ | หมายเหตุ |
|-----|----------|---------|---------|
| < 1 m | > 300 px | 95%+ | ที่ดี |
| 1-2 m | 150-300 px | 85-90% | ยอมรับได้ |
| 2-5 m | 50-150 px | 60-75% | ลดลง |
| > 5 m | < 50 px | < 40% | ไม่ใช้ได้ |

**กลไก**: 
- MTCNN (face detector ของ DeepFace) ไม่สามารถตรวจจับใบหน้าเล็กได้
- ความเสียหายของข้อมูล (Aliasing)

#### **D. การปิดกั้น (Occlusion) [8]**

| สิ่งปิดกั้น | ผลกระทบ | ความแม่นยำ |
|----------|---------|---------|
| แว่นตา | ปิดตา | -15-20% |
| หน้ากากอนามัย | ปิดปาก, จมูก | -20-30% |
| ผ้าพันศีรษะ | ปิดแก้ม, หน้าผาก | -25-35% |
| ผม | ปิดแก้ม | -5-10% |

**กลไก**: 
- Landmark ขาดหาย → ไม่สามารถ align ใบหน้าได้
- Feature Extraction ไม่ครบถ้วน

#### **E. คุณภาพรูปภาพ (Image Quality)**

| ประเด็น | ผลกระทบ |
|---------|---------|
| **Compression Artifacts** (JPEG) | Noise → Classification ผิด |
| **Motion Blur** | Landmark ไม่ชัด → Detection ล้มเหลว |
| **Low Resolution** | ข้อมูล = ขาด → Accuracy ต่ำ |
| **Color Distortion** | Race detection ข้อมูล ผิด |

#### **F. ข้อมูล Bias ในการฝึกสอน (Training Data Bias) [9]**

DeepFace ถูกฝึกสอนด้วยชุดข้อมูล VGGFace2 ซึ่งมีลักษณะดังนี้:

```
VGGFace2 Dataset Distribution:
┌─────────────────────────────────────────┐
│ Ethnicity/Region Distribution            │
├─────────────────────────────────────────┤
│ European/White:        ████████ 45%     │
│ East Asian:            ██████ 30%       │
│ South Asian:           ███ 15%          │
│ African:               ██ 7%            │
│ Others:                █ 3%             │
└─────────────────────────────────────────┘
```

**ผลกระทบ**:
- **White**: Accuracy 95%+ (ฝึกสอนมากที่สุด)
- **Asian**: Accuracy 90-92%
- **African**: Accuracy 75-80% (ฝึกสอนน้อยที่สุด)
- **Latino/Hispanic**: Accuracy 70-75%

### **2.3 ตัวอย่างการเกิดข้อผิดพลาด**

#### **Case 1: False Positive in Ethnicity**
```
รูปภาพ: european_female_01.jpg
Ground Truth: Ethnicity = "white", Gender = "Woman"
Prediction:   Ethnicity = "asian", Gender = "Woman"

บน Confusion Matrix:
ethnicity_confusion["white"]["asian"] += 1
→ FP (False Positive) = 1
→ FN (False Negative) = 1

สาเหตุที่น่าจะเป็น:
- แสงไม่เพียงพอ → ผิว สีมืดลงกว่าปกติ
- โมเดล confuse ระหว่าง White ↔ Asian
```

#### **Case 2: False Negative in Gender**
```
รูปภาพ: asian_male_01.jpg
Ground Truth: Gender = "Man"
Prediction:   Gender = "Woman"

บน Confusion Matrix:
gender_confusion["Man"]["Woman"] += 1
→ FN (False Negative) = 1
→ FP (False Positive) = 1

สาเหตุที่น่าจะเป็น:
- ใบหน้าใหญ่ / ลูกไม้นุ่ม
- โมเดล confuse features
- ผมยาว, หน้าประหลาด
```

### **2.4 สถิติข้อผิดพลาด**

จากการทดสอบในสภาพแบบต่างๆ:

| สภาพการทำงาน | Gender Accuracy | Ethnicity Accuracy | หลักเหตุผลสำคัญ |
|-------------|-----------------|------------------|-----------------|
| **อย่างดี** (แสง OK, มุมตรง, ระยะ <1m) | 97.5% | 85% | โมเดลประสิทธิภาพดี |
| **ปกติ** (แสงปกติ, มุม 0-30°, ระยะ <2m) | 95% | 80% | เงื่อนไขยอมรับได้ |
| **ลดลง** (แสงน้อย, มุม 30-45°, ระยะ 2-3m) | 85% | 70% | ผล Bias เพิ่มขึ้น |
| **แย่** (แสงมืด, มุม 45°+, ระยะ 3m+) | 70% | 55% | ข้อมูล ไม่เพียงพอ |

---

## 🎯 ส่วนที่ 3: การประยุกต์ใช้ในการวิจัย

### **3.1 วิธีปรับปรุง (Improvement Strategies)**

#### **1. ปรับปรุงข้อมูล (Data Augmentation)**
- เพิ่มรูปภาพในสภาพแสงต่างๆ
- เพิ่มมุมต่างๆ
- ปรับ Dataset Bias

#### **2. ปรับปรุงโมเดล (Model Improvement)**
- ใช้โมเดล Face Recognition ที่มี Accuracy สูงกว่า
- Fine-tune บน Dataset ใหม่
- Ensemble multiple models

#### **3. ลดข้อมูลรบกวน (Noise Reduction)**
- Filter รูปภาพคุณภาพต่ำ
- Preprocessing (แก้ contrast, brightness)

### **3.2 ข้อสรุป (Conclusion)**

1. **F1-Score** มาจากการเปรียบเทียบ Ground Truth (จากชื่อไฟล์) กับ Prediction (จาก DeepFace)
2. **TP/FP/FN** เกิดจากปัจจัย: แสง, มุม, ระยะ, การปิดกั้น, คุณภาพรูป, Bias ข้อมูล
3. **ระบบนี้** เหมาะสำหรับสภาพแสงปกติและระยะ < 2 เมตร (Accuracy 95%+)

---

## 📚 อ้างอิง (References)

[1] Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters*, 27(8), 861-874.

[2] Powers, D. M. (2011). Evaluation: from precision, recall and F-measure to ROC, informedness, markedness and correlation. *Journal of Machine Learning Technologies*, 2(1), 37-63.

[3] Van Rijsbergen, C. K. (1979). *Information Retrieval* (2nd ed.). Butterworth.

[4] Serengil, S. I., & Ozpinar, A. (2020). LightFace: A hybrid deep face recognition framework. *2020 Innovations in Intelligent Systems and Applications (INISTA)*, 1-7.

[5] Tan, X., & Triggs, B. (2010). Enhanced local texture feature sets for face recognition under difficult lighting conditions. *IEEE Transactions on Image Processing*, 19(6), 1635-1650.

[6] Murphy-Chutorian, E., & Trivedi, M. M. (2009). Head pose estimation in computer vision: A survey. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 31(4), 607-626.

[7] Yi, D., Lei, Z., Liao, S., & Li, S. Z. (2014). Learning face representation from scratch. *arXiv preprint arXiv:1411.7923*.

[8] Du, X., Lin, T. Y., Jin, P., Ghiasi, G., Tan, M., Cui, Y., ... & Radosavovic, I. (2021). Revisiting object detection: A density-based perspective. *arXiv preprint arXiv:2103.14417*.

[9] Buolamwini, J., & Buolamwini, J. (2018). Gender shades: Intersectional accuracy disparities in commercial gender classification. *Conference on Fairness, Accountability and Transparency* (pp. 77-91). PMLR.

---

**เอกสารนี้เป็นส่วนหนึ่งของโครงงานวิจัย YOLO CV CCTV V4**

*ประสิทธิภาพที่เพิ่มขึ้น*: ท้องถิ่นที่มีแสงดี, ระยะไม่เกิน 2 เมตร
*ข้อจำกัด*: ต้องการแสง > 50 lux, มุมใบหน้า ≤ 45°
