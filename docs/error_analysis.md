# 🔍 สาเหตุของความผิดพลาด - Error Analysis

## 📊 ความหมาย TP, FP, FN, TN

### Confusion Matrix ตัวอย่าง: Gender Classification (Man vs Woman)

|  | **Predicted** | |
|---|---|---|
| **Actual** | **Man** | **Woman** |
| **Man** | 2,460 ✅ | 151 ❌ |
| **Woman** | 1,373 ❌ | 1,016 ✅ |

**อธิบาย**:
- แถว = ความจริง (Ground Truth)
- คอลัมน์ = การทำนาย (Prediction)
- เส้นทแยงมุม (2,460 และ 1,016) = ทำนายถูก ✅
- นอกเส้นทแยงมุม (151 และ 1,373) = ทำนายผิด ❌


| ชนิด | ภาษาไทย | ความหมาย | จำนวน | ตัวอย่าง |
|------|---------|----------|-------|----------|
| **TP** (True Positive) | ทำนายถูก เป็นบวก | ทำนาย Man → จริงๆ เป็น Man | 2,460 | ✅ ชายจริง ทายว่าชาย |
| **FP** (False Positive) | ทำนายผิด เป็นบวก | ทำนาย Man → แต่จริงๆ เป็น Woman | 1,373 | ❌ หญิงจริง ทายว่าชาย |
| **FN** (False Negative) | ทำนายผิด เป็นลบ | ทำนาย Woman → แต่จริงๆ เป็น Man | 151 | ❌ ชายจริง ทายว่าหญิง |
| **TN** (True Negative) | ทำนายถูก เป็นลบ | ทำนาย Woman → จริงๆ เป็น Woman | 1,016 | ✅ หญิงจริง ทายว่าหญิง |

---

## 🤔 สาเหตุของความผิดพลาด

### 1. False Positive (FP) = 1,373 คน
**ผิด**: ทาย Man แต่จริงๆ เป็น Woman

#### สาเหตุที่เป็นไปได้:

| สาเหตุ | คำอธิบาย | ตัวอย่าง |
|--------|----------|----------|
| 🎭 **ทรงผมสั้น** | ผู้หญิงผมสั้น → โมเดลคิดว่าเป็นผู้ชาย | ผู้หญิงตัดผมทรงผู้ชาย |
| 👔 **แต่งกายเป็นทางการ** | ผู้หญิงใส่สูท → โมเดลสับสน | CEO หญิงใส่สูทดำ |
| 💪 **ใบหน้าคม** | ใบหน้าโครงหน้าคม (Masculine features) | นักกีฬาหญิง |
| 🎨 **ไม่แต่งหน้า** | ไม่มี makeup → ลักษณะดูเป็นกลาง | ไลฟ์สไตล์ธรรมชาติ |
| 📸 **มุมกล้อง/แสง** | มุมถ่ายไม่ดี แสงน้อย | ภาพเงา/ด้านข้าง |
| 🎯 **Dataset Bias** | โมเดลเทรนด้วยภาพ "ผู้หญิงทั่วไป" | เห็นผิดจากที่เคยเรียนรู้ |

---

### 2. False Negative (FN) = 151 คน
**ผิด**: ทาย Woman แต่จริงๆ เป็น Man

#### สาเหตุที่เป็นไปได้:

| สาเหตุ | คำอธิบาย | ตัวอย่าง |
|--------|----------|----------|
| 💇 **ทรงผมยาว** | ผู้ชายผมยาว → โมเดลคิดว่าเป็นผู้หญิง | ร็อคสตาร์, ศิลปิน |
| 👗 **แต่งกายสีสัน** | ผู้ชายใส่เสื้อสีสดใส | แฟชั่นทันสมัย |
| 😊 **ใบหน้านุ่มนวล** | ใบหน้าโครงหน้านุ่ม (Feminine features) | ผู้ชายเอเชียบางคน |
| 🧴 **ผิวเนียน/ดูแลดี** | ผิวขาวเนียน ไม่มีหนวด | K-pop idol |
| 👶 **อายุน้อย** | วัยรุ่นชาย ยังไม่มีลักษณะผู้ชายชัดเจน | นักเรียนชาย 15-18 ปี |
| 🎯 **Dataset Bias** | โมเดลเทรนด้วยภาพ "ผู้ชายทั่วไป" | เห็นผิดจากที่เคยเรียนรู้ |

---

## 🌍 Ethnicity Classification - ทำไมผิดมากกว่า?

### Accuracy: 50.8% (ต่ำกว่า Gender มาก)

#### สาเหตุหลัก:

| ปัญหา | คำอธิบาย | ผลกระทบ |
|-------|----------|---------|
| 🧬 **Mixed Race** | คนผสมเชื้อชาติ | โมเดลสับสน ไม่รู้จะเลือกอะไร |
| 🎨 **Skin Tone Overlap** | โทนผิวซ้อนทับกัน | Latino ≈ Indian ≈ Asian บางคน |
| 📊 **Dataset Imbalance** | โมเดลเทรนด้วยข้อมูลไม่สมดุล | Bias ไปทาง White/Asian |
| 💡 **Lighting** | แสงทำให้สีผิวดูต่าง | แสงเหลือง → ผิวดูเหลือง |
| 🎭 **Facial Features Vary** | ลักษณะใบหน้าหลากหลายภายในเชื้อชาติเดียวกัน | Asian มีหลายแบบ (East, Southeast) |
| 🌍 **Within-Race Diversity** | ความหลากหลายภายในเชื้อชาติเดียวกัน | คนผิวดำจากแอฟริกา ≠ แคริบเบียน |

---

## 📉 ตัวอย่างเฉพาะ: Latino Hispanic (Worst Performance)

```
Latino Hispanic Performance:
- Precision: 0.531 (53.1%)
- Recall: 0.379 (37.9%)
- F1: 0.442 (44.2%) ← ต่ำที่สุด
```

### Confusion Matrix (Latino Hispanic):

| **จริง: Latino** | **จำนวน** | **%** | **สถานะ** |
|------------------|-----------|-------|----------|
| ทายว่า **Latino** | 221 | 38% | ✅ ถูกต้อง |
| ทายว่า **White** | 206 | 35% | ❌ ผิด (คล้ายกัน) |
| ทายว่า **Asian** | 153 | 26% | ❌ ผิด (บางคนคล้าย) |
| ทายว่า **Black** | 119 | 20% | ❌ ผิด |
| ทายว่า **Indian** | 37 | 6% | ❌ ผิด |
| **รวม** | **583** | **100%** | |


### สาเหตุที่ Latino ผิดพลาดมาก:

1. 🎨 **โทนผิวหลากหลาย**: Latino มีทั้งผิวขาว ผิวคล้ำ ผิวปานกลาง
2. 🧬 **Mixed Ancestry**: ผสมระหว่าง European + Indigenous + African
3. 🌍 **Geographic Diversity**: Latino จาก Mexico ≠ Argentina ≠ Cuba
4. 📊 **Under-representation**: Dataset อาจมี Latino น้อยกว่าเชื้อชาติอื่น

---

## 🎯 สรุป: สาเหตุหลักของความผิดพลาด

### 1. ปัจจัยด้านรูปภาพ (Image Quality)
- ✗ แสง (Lighting)
- ✗ มุมกล้อง (Camera Angle)
- ✗ ความละเอียด (Resolution)
- ✗ การบดบัง (Occlusion)

### 2. ปัจจัยด้านบุคคล (Subject Factors)
- ✗ ทรงผม (Hairstyle)
- ✗ การแต่งกาย (Clothing)
- ✗ การแต่งหน้า (Makeup)
- ✗ โครงหน้า (Facial Structure)
- ✗ อายุ (Age)

### 3. ปัจจัยด้านโมเดล (Model Limitations)
- ✗ Dataset Bias (เทรนด้วยข้อมูลไม่สมดุล)
- ✗ Fixed Categories (แบ่งแค่ 2 เพศ, 4-7 เชื้อชาติ)
- ✗ Cannot handle edge cases (Mixed race, Non-binary)

### 4. ปัจจัยด้านความหลากหลาย (Diversity)
- ✗ Within-group variation สูง
- ✗ Between-group overlap สูง
- ✗ Cultural/Regional differences

---

## 💡 แนวทางแก้ไข

### ระยะสั้น (Immediate):
1. ✅ ปรับ confidence threshold
2. ✅ Add pre-processing (lighting normalization)
3. ✅ Filter low-quality images

### ระยะกลาง (Medium-term):
1. ✅ Fine-tune model with more diverse data
2. ✅ Add data augmentation (hair, lighting, angle)
3. ✅ Ensemble multiple models

### ระยะยาว (Long-term):
1. ✅ Collect balanced dataset
2. ✅ Train separate models for edge cases
3. ✅ Add "Uncertain" category for ambiguous cases

---

## 📚 อ้างอิง

1. **FairFace Dataset**: Karkkainen & Joo, 2021
2. **DeepFace**: Serengil & Ozpinar, 2020
3. **Fairness in ML**: Buolamwini & Gebru, 2018

---

**สรุป**: ความผิดพลาดเกิดจากการที่โมเดล AI ไม่สามารถแยกแยะความแตกต่างที่ซับซ้อนได้ เมื่อเจอกรณีที่ "ไม่ชัดเจน" หรือ "ไม่เหมือนกับที่เคยเห็นตอนเทรน"
