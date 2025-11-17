# Week 13: Object Detection Foundations
## Comprehensive Lecture Notes

**Course:** 21CSE558T - Deep Neural Network Architectures
**Module:** 5 - Object Localization and Detection
**Week:** 13 of 15
**Date:** November 2025
**Instructor:** Prof. Ramesh Babu

---

## Table of Contents

1. [What is Object Detection?](#part-1-what-is-object-detection)
2. [Bounding Boxes](#part-2-bounding-boxes)
3. [Intersection over Union (IOU)](#part-3-intersection-over-union-iou)
4. [Evaluation Metrics](#part-4-evaluation-metrics)
5. [Classical Approaches](#part-5-classical-approaches)
6. [Preview of Modern Methods](#part-6-preview-of-modern-methods)
7. [Summary](#part-7-summary)
8. [Practice Questions](#part-8-practice-questions)

---

## Part 1: What is Object Detection?

### 1.1 Introduction

Object detection is one of the fundamental problems in computer vision. Unlike simple image classification, object detection requires the model to not only identify **what** objects are present in an image, but also **where** they are located.

**Three Related Tasks:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    Computer Vision Tasks                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. IMAGE CLASSIFICATION                                          │
│     Input:  [Image]                                               │
│     Output: "Cat"                                                 │
│     Task:   What is in this image?                                │
│                                                                   │
│  2. OBJECT LOCALIZATION                                           │
│     Input:  [Image]                                               │
│     Output: "Cat" + [x, y, w, h]                                  │
│     Task:   Where is THE object? (single object)                  │
│                                                                   │
│  3. OBJECT DETECTION ← THIS WEEK                                  │
│     Input:  [Image]                                               │
│     Output: [("Cat", [x1,y1,w1,h1]), ("Dog", [x2,y2,w2,h2]), ...] │
│     Task:   Where are ALL the objects? (multiple objects)         │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Formal Definition

**Object Detection** is the task of:
1. **Localizing** all objects in an image (finding their positions)
2. **Classifying** each detected object (identifying what they are)

**Input:**
- Image: I ∈ ℝ^(H×W×3) (height × width × 3 color channels)

**Output:**
- Set of detections: {(c₁, b₁, s₁), (c₂, b₂, s₂), ..., (cₙ, bₙ, sₙ)}
  - cᵢ: Class label (e.g., "person", "car", "dog")
  - bᵢ: Bounding box coordinates [x, y, w, h]
  - sᵢ: Confidence score ∈ [0, 1]
  - n: Number of detected objects (variable!)

### 1.3 Real-World Applications

Object detection has transformed numerous industries:

**1. Autonomous Vehicles** 🚗
- Detect pedestrians, vehicles, traffic signs, lane markings
- Safety-critical: Must be real-time (30+ FPS) and highly accurate
- Examples: Tesla Autopilot, Waymo, Cruise

**2. Medical Imaging** 🏥
- Tumor detection in CT/MRI scans
- Lesion detection in skin images
- Cell counting in microscopy
- Organ localization for surgical planning

**3. Surveillance and Security** 📹
- Person detection and tracking
- Anomaly detection (unattended baggage, intrusion)
- Crowd counting and management
- Face detection for access control

**4. Retail and E-commerce** 🛒
- Product recognition on shelves
- Automated checkout (Amazon Go)
- Visual search (find similar products)
- Inventory management

**5. Wildlife Monitoring** 🦁
- Animal detection in camera trap images
- Species identification
- Population counting
- Behavior analysis

**6. Agriculture** 🌾
- Crop disease detection
- Fruit/vegetable counting for yield estimation
- Weed detection for precision spraying
- Livestock monitoring

**7. Industrial Inspection** 🏭
- Defect detection in manufacturing
- Quality control on assembly lines
- Safety equipment detection (helmets, vests)
- Automated visual inspection

**8. Sports Analytics** ⚽
- Player tracking and position analysis
- Ball detection and trajectory prediction
- Automated highlight generation
- Performance metrics

### 1.4 The Challenge

Object detection is significantly harder than classification because:

**Challenge 1: Variable Number of Objects**
- Classification: Always output 1 label
- Detection: Output 0, 1, 5, 100, ... objects (unknown in advance!)
- Network must handle variable-length outputs

**Challenge 2: Multiple Scales**
- Objects appear at different sizes
  - Near objects: Large (hundreds of pixels)
  - Far objects: Small (tens of pixels or less)
- Need to detect both simultaneously

**Challenge 3: Occlusion**
- Objects partially hidden by other objects
- Only part of object visible
- Must infer complete object from partial view

**Challenge 4: Cluttered Scenes**
- Many objects close together
- Overlapping bounding boxes
- Which pixels belong to which object?

**Challenge 5: Computational Efficiency**
- Real-time applications need 30+ FPS
- High-resolution images (1920×1080 or higher)
- Limited computational resources (mobile devices, embedded systems)

**Challenge 6: Class Imbalance**
- Background (no object) is majority of image
- Objects occupy small portion of pixels
- Hard to train: overwhelming negatives

### 1.5 Historical Context

**Before Deep Learning (Pre-2012):**
- Hand-crafted features: HOG (Histogram of Oriented Gradients), SIFT, SURF
- Sliding window classifiers (SVM, AdaBoost)
- Slow: Seconds per image
- Low accuracy: 30-40% mAP on PASCAL VOC

**Deep Learning Era (2012-Present):**
- 2012: AlexNet wins ImageNet → CNNs for classification
- 2014: R-CNN → First successful CNN-based detector (53.3% mAP)
- 2015: Fast R-CNN, Faster R-CNN → Speed improvements
- 2016: YOLO v1, SSD → Real-time detection (30+ FPS)
- 2018-2024: YOLOv3, v4, v5, v8 → 60+ FPS, 90+% accuracy

**Current State:**
- Real-time detection: 30-100+ FPS
- High accuracy: 80-95% mAP on COCO dataset
- Mobile deployment: Models running on smartphones
- 1000+ classes: Can detect nearly any object

### 1.6 Problem Formulation

**Given:**
- Training dataset: {(I₁, A₁), (I₂, A₂), ..., (Iₘ, Aₘ)}
  - Iᵢ: Image
  - Aᵢ: Annotations (ground truth bounding boxes + labels)

**Learn:**
- Function f: Image → Detections
- f(I) = {(c₁, b₁, s₁), ..., (cₙ, bₙ, sₙ)}

**Optimize:**
- Localization accuracy: Bounding boxes match ground truth
- Classification accuracy: Labels are correct
- Speed: Fast inference (real-time if needed)
- Generalization: Work on unseen images

---

## Part 2: Bounding Boxes

### 2.1 What is a Bounding Box?

A **bounding box** is a rectangle that tightly encloses an object in an image. It's the simplest way to represent object location.

**Properties:**
- Axis-aligned (not rotated)
- Defined by 4 numbers (coordinates)
- Tight fit: Minimal area while containing entire object
- 2D approximation (objects are 3D, but boxes are 2D rectangles)

**Limitations:**
- Cannot represent object rotation
- Cannot represent irregular shapes
- Overlapping objects create ambiguity

**Alternatives (More Advanced):**
- Rotated bounding boxes (5 parameters: x, y, w, h, θ)
- Segmentation masks (pixel-level precision)
- Keypoints (for articulated objects like humans)
- 3D bounding boxes (for autonomous driving)

### 2.2 Coordinate Formats

There are **three main formats** for representing bounding boxes:

**Format 1: XYXY (Corner Format)**
```
[x_min, y_min, x_max, y_max]
```
- (x_min, y_min): Top-left corner
- (x_max, y_max): Bottom-right corner
- Used by: Pascal VOC, many research papers
- Advantage: Easy to calculate area and intersection

**Example:**
```
Box: [50, 100, 200, 300]
├─ Top-left corner: (50, 100)
├─ Bottom-right corner: (200, 300)
├─ Width: 200 - 50 = 150 pixels
└─ Height: 300 - 100 = 200 pixels
```

**Format 2: XYWH (Center Format)**
```
[x_center, y_center, width, height]
```
- (x_center, y_center): Center point of box
- (width, height): Box dimensions
- Used by: YOLO, some modern detectors
- Advantage: Symmetric, easier for some loss functions

**Example:**
```
Box: [125, 200, 150, 200]
├─ Center: (125, 200)
├─ Width: 150 pixels
├─ Height: 200 pixels
├─ Top-left: (125 - 150/2, 200 - 200/2) = (50, 100)
└─ Bottom-right: (125 + 150/2, 200 + 200/2) = (200, 300)
```

**Format 3: COCO (Top-Left Width-Height)**
```
[x_top_left, y_top_left, width, height]
```
- (x_top_left, y_top_left): Top-left corner
- (width, height): Box dimensions
- Used by: MS COCO dataset (largest object detection benchmark)
- Advantage: Mix of absolute position and dimensions

**Example:**
```
Box: [50, 100, 150, 200]
├─ Top-left: (50, 100)
├─ Width: 150 pixels
├─ Height: 200 pixels
└─ Bottom-right: (50 + 150, 100 + 200) = (200, 300)
```

### 2.3 Format Conversion

Converting between formats:

**XYXY → XYWH:**
```python
def xyxy_to_xywh(box):
    x_min, y_min, x_max, y_max = box
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    return [x_center, y_center, width, height]
```

**XYWH → XYXY:**
```python
def xywh_to_xyxy(box):
    x_center, y_center, width, height = box
    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2
    return [x_min, y_min, x_max, y_max]
```

**XYXY → COCO:**
```python
def xyxy_to_coco(box):
    x_min, y_min, x_max, y_max = box
    width = x_max - x_min
    height = y_max - y_min
    return [x_min, y_min, width, height]
```

**COCO → XYXY:**
```python
def coco_to_xyxy(box):
    x, y, width, height = box
    x_min = x
    y_min = y
    x_max = x + width
    y_max = y + height
    return [x_min, y_min, x_max, y_max]
```

### 2.4 Normalized Coordinates

**Problem:** Bounding boxes in pixel coordinates are resolution-dependent.
- Box [100, 200, 300, 400] on 640×480 image
- Same box on 1920×1440 image would be [300, 600, 900, 1200]
- Makes it hard to transfer models between different resolutions

**Solution:** Normalize coordinates to [0, 1] range.

**Normalization:**
```python
def normalize_box(box, image_width, image_height):
    x_min, y_min, x_max, y_max = box  # XYXY format
    x_min_norm = x_min / image_width
    y_min_norm = y_min / image_height
    x_max_norm = x_max / image_width
    y_max_norm = y_max / image_height
    return [x_min_norm, y_min_norm, x_max_norm, y_max_norm]
```

**Denormalization:**
```python
def denormalize_box(box_norm, image_width, image_height):
    x_min_norm, y_min_norm, x_max_norm, y_max_norm = box_norm
    x_min = x_min_norm * image_width
    y_min = y_min_norm * image_height
    x_max = x_max_norm * image_width
    y_max = y_max_norm * image_height
    return [x_min, y_min, x_max, y_max]
```

**Advantages:**
- Resolution independence
- Easier to train models on multiple image sizes
- Numerical stability (values in [0, 1])
- Used by YOLO and many modern detectors

### 2.5 Ground Truth Annotations

Datasets must be **annotated** (labeled) with ground truth bounding boxes.

**Annotation Process:**
1. Load image
2. Draw tight bounding box around each object
3. Assign class label
4. Save coordinates in specific format
5. Repeat for all images in dataset

**Popular Annotation Tools:**
- **labelImg** (Python, simple, PASCAL VOC/YOLO formats)
- **CVAT** (Web-based, team collaboration, COCO format)
- **LabelMe** (MIT, polygon annotation)
- **VGG Image Annotator (VIA)** (In-browser, no install needed)
- **RectLabel** (macOS, user-friendly)

**Dataset Formats:**

**PASCAL VOC XML:**
```xml
<annotation>
  <object>
    <name>cat</name>
    <bndbox>
      <xmin>100</xmin>
      <ymin>200</ymin>
      <xmax>300</xmax>
      <ymax>400</ymax>
    </bndbox>
  </object>
</annotation>
```

**COCO JSON:**
```json
{
  "annotations": [
    {
      "id": 1,
      "image_id": 123,
      "category_id": 3,
      "bbox": [100, 200, 200, 200],
      "area": 40000,
      "iscrowd": 0
    }
  ]
}
```

**YOLO TXT:**
```
# class_id x_center y_center width height (all normalized)
3 0.312 0.521 0.234 0.187
```

---

## Part 3: Intersection over Union (IOU)

### 3.1 The Core Metric

**Intersection over Union (IOU)**, also called **Jaccard Index**, is the fundamental metric for evaluating bounding box predictions.

**Definition:**
```
IOU = Area of Intersection / Area of Union
```

**Mathematical Formula:**
```
IOU(A, B) = |A ∩ B| / |A ∪ B|

Where:
- A: Predicted bounding box
- B: Ground truth bounding box
- A ∩ B: Intersection (overlap area)
- A ∪ B: Union (combined area)
```

**Properties:**
- Range: [0, 1]
  - IOU = 0: No overlap (boxes don't touch)
  - IOU = 1: Perfect match (boxes identical)
  - IOU = 0.5: Moderate overlap
  - IOU = 0.8: Strong overlap
- Symmetric: IOU(A, B) = IOU(B, A)
- Scale-invariant: Works for any box size

### 3.2 Calculation Steps

Given two boxes in XYXY format:
- Box A: [x1_min, y1_min, x1_max, y1_max]
- Box B: [x2_min, y2_min, x2_max, y2_max]

**Step 1: Calculate Intersection Coordinates**
```python
x_min_inter = max(x1_min, x2_min)
y_min_inter = max(y1_min, y2_min)
x_max_inter = min(x1_max, x2_max)
y_max_inter = min(y1_max, y2_max)
```

**Step 2: Calculate Intersection Area**
```python
inter_width = max(0, x_max_inter - x_min_inter)
inter_height = max(0, y_max_inter - y_min_inter)
intersection_area = inter_width * inter_height
```

Note: `max(0, ...)` handles non-overlapping boxes (intersection = 0)

**Step 3: Calculate Individual Box Areas**
```python
area_A = (x1_max - x1_min) * (y1_max - y1_min)
area_B = (x2_max - x2_min) * (y2_max - y2_min)
```

**Step 4: Calculate Union Area**
```python
union_area = area_A + area_B - intersection_area
```

Subtract intersection to avoid double-counting!

**Step 5: Calculate IOU**
```python
iou = intersection_area / union_area if union_area > 0 else 0
```

**Complete Implementation:**
```python
def calculate_iou(box1, box2):
    """
    Calculate IOU between two boxes (XYXY format)

    Args:
        box1: [x_min, y_min, x_max, y_max]
        box2: [x_min, y_min, x_max, y_max]

    Returns:
        iou: float in [0, 1]
    """
    # Intersection coordinates
    x_min_inter = max(box1[0], box2[0])
    y_min_inter = max(box1[1], box2[1])
    x_max_inter = min(box1[2], box2[2])
    y_max_inter = min(box1[3], box2[3])

    # Intersection area
    inter_w = max(0, x_max_inter - x_min_inter)
    inter_h = max(0, y_max_inter - y_min_inter)
    intersection = inter_w * inter_h

    # Individual areas
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Union area
    union = area1 + area2 - intersection

    # IOU
    iou = intersection / union if union > 0 else 0

    return iou
```

### 3.3 Interpreting IOU Values

**IOU Thresholds in Practice:**

| IOU Range | Interpretation | Usage |
|-----------|----------------|-------|
| 0.0 - 0.3 | Poor match | Reject as detection |
| 0.3 - 0.5 | Weak match | Borderline case |
| 0.5 - 0.7 | Good match | **PASCAL VOC standard** |
| 0.7 - 0.9 | Strong match | High-quality detection |
| 0.9 - 1.0 | Excellent match | Nearly perfect |

**Standard Thresholds:**
- **PASCAL VOC:** IOU > 0.5 for True Positive
- **COCO (Loose):** IOU > 0.5
- **COCO (Strict):** IOU > 0.75
- **COCO (Very Strict):** IOU > 0.95
- **COCO mAP:** Average across [0.5, 0.55, 0.6, ..., 0.95]

**Why 0.5?**
- Historical standard from PASCAL VOC challenge
- Represents "more than half overlap"
- Reasonable balance between strict and lenient
- Widely adopted by research community

### 3.4 Edge Cases

**Case 1: Non-Overlapping Boxes**
```
Box A: [10, 10, 50, 50]
Box B: [100, 100, 150, 150]

Intersection: 0 (boxes don't touch)
Union: 1600 + 2500 = 4100
IOU: 0 / 4100 = 0.0
```

**Case 2: Identical Boxes**
```
Box A: [50, 50, 150, 150]
Box B: [50, 50, 150, 150]

Intersection: 10000
Union: 10000 + 10000 - 10000 = 10000
IOU: 10000 / 10000 = 1.0
```

**Case 3: Nested Boxes (One Inside Another)**
```
Box A (large): [0, 0, 100, 100]     Area = 10000
Box B (small): [25, 25, 75, 75]     Area = 2500

Intersection: 2500 (entire small box)
Union: 10000 + 2500 - 2500 = 10000
IOU: 2500 / 10000 = 0.25
```

Note: Even though one box is completely inside, IOU is only 0.25!

**Case 4: Touching Boxes (Share Edge)**
```
Box A: [0, 0, 50, 50]
Box B: [50, 0, 100, 50]

They share edge at x=50, but no overlapping area
Intersection: 0
IOU: 0.0
```

### 3.5 Vectorized IOU (Batch Computation)

For efficiency, compute IOU between N predicted boxes and M ground truth boxes simultaneously.

**Naive Approach (Loops):**
```python
ious = np.zeros((N, M))
for i in range(N):
    for j in range(M):
        ious[i, j] = calculate_iou(pred_boxes[i], gt_boxes[j])
# Time: O(N × M)
```

**Vectorized Approach (NumPy Broadcasting):**
```python
def calculate_iou_batch(boxes1, boxes2):
    """
    Calculate IOU between N boxes and M boxes

    Args:
        boxes1: (N, 4) array [x_min, y_min, x_max, y_max]
        boxes2: (M, 4) array [x_min, y_min, x_max, y_max]

    Returns:
        iou_matrix: (N, M) array of IOU values
    """
    # Reshape for broadcasting: (N, 1, 4) and (1, M, 4)
    boxes1_exp = boxes1[:, np.newaxis, :]  # (N, 1, 4)
    boxes2_exp = boxes2[np.newaxis, :, :]  # (1, M, 4)

    # Intersection coordinates (N, M)
    x_min_inter = np.maximum(boxes1_exp[:, :, 0], boxes2_exp[:, :, 0])
    y_min_inter = np.maximum(boxes1_exp[:, :, 1], boxes2_exp[:, :, 1])
    x_max_inter = np.minimum(boxes1_exp[:, :, 2], boxes2_exp[:, :, 2])
    y_max_inter = np.minimum(boxes1_exp[:, :, 3], boxes2_exp[:, :, 3])

    # Intersection area (N, M)
    inter_w = np.maximum(0, x_max_inter - x_min_inter)
    inter_h = np.maximum(0, y_max_inter - y_min_inter)
    intersection = inter_w * inter_h

    # Individual areas (N,) and (M,)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Broadcast to (N, M)
    area1_exp = area1[:, np.newaxis]  # (N, 1)
    area2_exp = area2[np.newaxis, :]  # (1, M)

    # Union area (N, M)
    union = area1_exp + area2_exp - intersection

    # IOU (N, M)
    iou = intersection / np.maximum(union, 1e-8)  # Avoid division by zero

    return iou

# Speedup: 10-100x faster for large N and M!
```

---

## Part 4: Evaluation Metrics

### 4.1 Precision and Recall for Object Detection

Unlike classification, object detection evaluation is more nuanced.

**Definitions:**

**True Positive (TP):**
- Predicted box with IOU > threshold AND correct class
- Example: Detected "car" with IOU=0.7 on ground truth "car"

**False Positive (FP):**
- Predicted box with IOU < threshold OR wrong class
- Example 1: Detected "car" with IOU=0.3 (poor localization)
- Example 2: Detected "car" on ground truth "bus" (wrong class)
- Example 3: Detected object where no ground truth exists (hallucination)

**False Negative (FN):**
- Ground truth object not detected
- Missed detection

**Precision:**
```
Precision = TP / (TP + FP)
```
"Of all predictions, what fraction were correct?"

**Recall:**
```
Recall = TP / (TP + FN) = TP / Total Ground Truth Objects
```
"Of all ground truth objects, what fraction did we detect?"

**F1-Score:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
Harmonic mean of precision and recall

### 4.2 Confidence Thresholding

Object detectors output a **confidence score** for each prediction.

**Confidence Score:** s ∈ [0, 1]
- How confident is the model that this is a real object?
- s = 0.9: Very confident
- s = 0.3: Low confidence (might be background)

**Thresholding:**
- Set threshold τ (e.g., 0.5)
- Keep only predictions with s ≥ τ
- Discard predictions with s < τ

**Effect on Metrics:**

| Threshold | Predictions Kept | Precision | Recall |
|-----------|------------------|-----------|--------|
| τ = 0.1 (low) | Many | Low (many FP) | High (few FN) |
| τ = 0.5 (medium) | Moderate | Medium | Medium |
| τ = 0.9 (high) | Few | High (few FP) | Low (many FN) |

**Trade-off:** Higher threshold → Higher precision, Lower recall

### 4.3 Precision-Recall Curve

**Idea:** Plot precision vs recall for different confidence thresholds.

**Procedure:**
1. Sort predictions by confidence (descending)
2. For each threshold τ ∈ {0.05, 0.10, 0.15, ..., 0.95}:
   - Keep predictions with confidence ≥ τ
   - Calculate TP, FP, FN using IOU threshold (e.g., 0.5)
   - Compute precision and recall
   - Plot point (recall, precision)
3. Connect points to form curve

**Characteristics:**
- Starts at high recall, low precision (τ = 0, keep all predictions)
- Ends at low recall, high precision (τ → 1, keep only very confident)
- Ideally: High precision across all recall levels (curve near top-right)

**Example:**
```
Threshold    Predictions    TP    FP    FN    Precision    Recall
0.1          100            80    20    10    0.80         0.89
0.3          80             75    5     15    0.94         0.83
0.5          60             58    2     32    0.97         0.64
0.7          30             29    1     61    0.97         0.32
0.9          10             10    0     80    1.00         0.11
```

### 4.4 Average Precision (AP)

**Average Precision (AP)** summarizes the precision-recall curve as a single number.

**Method 1: 11-Point Interpolation (PASCAL VOC 2007)**
```
AP = (1/11) × Σ P_interp(r)

Where r ∈ {0.0, 0.1, 0.2, ..., 1.0}
P_interp(r) = max_{r' ≥ r} P(r')
```

**Method 2: All-Point Interpolation (PASCAL VOC 2010+, COCO)**
```
AP = Σ (r_n+1 - r_n) × P_interp(r_n+1)

Where r_n are all unique recall values
```

**Intuitive Understanding:**
- AP ≈ Area under precision-recall curve
- AP = 1.0: Perfect detector (always correct)
- AP = 0.5: Moderate detector
- AP = 0.1: Poor detector

**Example Calculation:**
```python
def calculate_ap(precisions, recalls):
    """
    Calculate Average Precision (all-point interpolation)

    Args:
        precisions: List of precision values
        recalls: List of recall values (sorted ascending)

    Returns:
        ap: Average Precision
    """
    # Add sentinel values
    recalls = [0.0] + recalls + [1.0]
    precisions = [0.0] + precisions + [0.0]

    # Interpolate precision (monotonically decreasing)
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Calculate AP as area under curve
    ap = 0.0
    for i in range(len(recalls) - 1):
        ap += (recalls[i + 1] - recalls[i]) * precisions[i + 1]

    return ap
```

### 4.5 Mean Average Precision (mAP)

**Mean Average Precision (mAP)** extends AP to multiple classes.

**Definition:**
```
mAP = (1/C) × Σ AP_c

Where:
- C: Number of classes
- AP_c: Average Precision for class c
```

**Example:**
```
Dataset: 3 classes (person, car, dog)

AP_person = 0.85
AP_car = 0.78
AP_dog = 0.92

mAP = (0.85 + 0.78 + 0.92) / 3 = 0.85
```

**Interpretation:**
- mAP = 0.85 means "on average, 85% precision across all recall levels and all classes"
- Higher mAP = Better detector
- State-of-the-art: 50-60% mAP on COCO (80 classes, strict evaluation)

### 4.6 mAP Variants

**mAP@[IOU]:** IOU threshold used to define TP

**PASCAL VOC mAP:**
- Single IOU threshold: 0.5
- Notation: mAP@0.5 or mAP50
- Lenient: Allows moderate overlap

**COCO mAP (Primary Metric):**
- Average across IOU thresholds: [0.5, 0.55, 0.6, ..., 0.95]
- Notation: mAP@[0.5:0.95] or simply "mAP"
- Strict: Rewards precise localization
- Formula: mAP = (1/10) × Σ mAP@IOU

**COCO Additional Metrics:**
- **mAP@0.5:** Lenient (same as PASCAL VOC)
- **mAP@0.75:** Strict
- **mAP_small:** Only small objects (area < 32²)
- **mAP_medium:** Medium objects (32² < area < 96²)
- **mAP_large:** Large objects (area > 96²)

**Comparison:**
```
Model A: mAP@0.5 = 0.75, mAP@[0.5:0.95] = 0.52
Model B: mAP@0.5 = 0.70, mAP@[0.5:0.95] = 0.55

Model A: Better at detection (finding objects)
Model B: Better at localization (precise bounding boxes)
```

### 4.7 Complete Evaluation Example

**Scenario:**
- 10 images with 50 ground truth "person" objects
- Model predicts 60 bounding boxes (varying confidence)
- IOU threshold: 0.5

**Step 1: Match predictions to ground truth**
```python
for each prediction:
    find ground truth with highest IOU
    if IOU > 0.5 and correct class and not already matched:
        TP += 1
    else:
        FP += 1

FN = Total GT - TP
```

**Step 2: Calculate precision and recall at different confidence thresholds**

**Step 3: Plot precision-recall curve**

**Step 4: Calculate AP**
```
AP_person = 0.82
```

**Step 5: Repeat for all classes, compute mAP**
```
AP_person = 0.82
AP_car = 0.75
AP_dog = 0.88
AP_cat = 0.79

mAP = (0.82 + 0.75 + 0.88 + 0.79) / 4 = 0.81
```

**Conclusion:** Model achieves 81% mAP → Good performance!

---

## Part 5: Classical Approaches

### 5.1 Historical Context

Before deep learning (pre-2012), object detection relied on:
1. **Hand-crafted features** (HOG, SIFT, SURF)
2. **Sliding window** classifiers
3. **Traditional machine learning** (SVM, AdaBoost)

**Limitations:**
- Slow: Seconds per image
- Low accuracy: 30-40% mAP
- Required extensive feature engineering
- Difficult to generalize

### 5.2 Sliding Window Detector

**Idea:** Exhaustively search all possible locations and scales.

**Algorithm:**
```
For each scale s in [0.5, 0.7, 1.0, 1.4, 2.0]:
    Resize image by factor s
    For each window position (x, y):
        Extract window (fixed size, e.g., 128×128)
        Extract features (e.g., HOG)
        Classify with SVM: Object or Background?
        If Object:
            Determine class (person, car, etc.)
            Save (x, y, scale, class, confidence)
```

**Parameters:**
- **Window size:** 64×64, 128×128 (fixed)
- **Stride:** How much to move window (8-16 pixels)
- **Scales:** Multiple sizes to detect different object sizes

**Example:**
```
Image: 640 × 480
Window: 128 × 128
Stride: 16 pixels
Scales: 5

Windows per scale: ((640-128)/16 + 1) × ((480-128)/16 + 1) = 32 × 22 = 704
Total windows: 704 × 5 scales = 3,520 windows

Time: 3,520 windows × 10ms/window = 35 seconds!
```

### 5.3 Computational Cost Analysis

**Bottlenecks:**
1. **Thousands of windows:** For 1024×768 image with 5 scales: 10,000+ windows
2. **Feature extraction:** For each window (slow with HOG/SIFT)
3. **Classification:** SVM evaluation for each window

**Why so slow?**
- Redundant computation: Overlapping windows recompute similar features
- No sharing of computation across windows
- Not end-to-end trainable

**Performance:**
- Speed: 1-10 seconds per image (not real-time!)
- Accuracy: 30-40% mAP on PASCAL VOC
- Memory: High (storing features for all windows)

### 5.4 Selective Search (R-CNN Era)

**Problem:** Sliding window tests too many useless locations (e.g., sky, uniform walls).

**Solution:** Selective Search (2012)
- Generate ~2,000 **region proposals** (candidate boxes likely to contain objects)
- Use image segmentation and hierarchical grouping
- Test only these 2,000 regions (vs 10,000+ sliding windows)

**Algorithm:**
1. Over-segment image into small regions
2. Greedily merge similar regions
3. Generate bounding boxes from merged regions
4. Rank by "objectness" score

**Advantages:**
- Reduces proposals from 10,000+ to ~2,000
- Better coverage of object shapes (not just fixed aspect ratio)
- Used in R-CNN (2014) → 53% mAP!

**Limitations:**
- Still slow (2-3 seconds for selective search alone)
- Not learned (fixed algorithm, can't improve with data)
- Eventually replaced by learned region proposal networks (RPN)

### 5.5 Limitations of Classical Methods

**Why classical methods failed:**

1. **Fixed features:** HOG/SIFT can't adapt to new object types
2. **Manual engineering:** Requires domain expertise, doesn't generalize
3. **Two-stage pipeline:** Features → Classification (not end-to-end)
4. **Slow:** Can't achieve real-time performance
5. **Low accuracy:** Struggle with occlusion, scale variation, clutter

**The Deep Learning Revolution:**
- 2012: AlexNet shows CNNs outperform hand-crafted features
- 2014: R-CNN applies CNNs to object detection → Big accuracy jump
- 2015-2024: YOLO, SSD, Faster R-CNN → Real-time + high accuracy

---

## Part 6: Preview of Modern Methods

### 6.1 Two Paradigms

Modern object detectors fall into two categories:

**1. Two-Stage Detectors** (Week 15: R-CNN Family)
```
Stage 1: Generate region proposals (candidate boxes)
         ↓
Stage 2: Classify and refine each proposal
```

**Examples:**
- R-CNN (2014): 53% mAP, 47 seconds/image
- Fast R-CNN (2015): 66% mAP, 2 seconds/image
- Faster R-CNN (2015): 73% mAP, 0.2 seconds/image

**Characteristics:**
- Higher accuracy (70-80% mAP on COCO)
- Slower (5-20 FPS)
- Better for precision-critical applications (medical imaging)

**2. Single-Shot Detectors** (Week 14: YOLO, SSD)
```
Single CNN forward pass → Directly predict boxes + classes
```

**Examples:**
- YOLO v1 (2016): 63% mAP, 45 FPS
- SSD (2016): 76% mAP, 59 FPS
- YOLO v8 (2023): 90% mAP, 80+ FPS

**Characteristics:**
- Real-time (30-100+ FPS)
- Moderate to high accuracy (60-90% mAP)
- Best for real-time applications (autonomous driving, video)

### 6.2 Key Ideas in Modern Detectors

**1. End-to-End Learning**
- Single neural network: Image → Detections
- Trained with single loss function (localization + classification)
- No hand-crafted features

**2. Anchor Boxes**
- Predefined boxes at different scales and aspect ratios
- Model predicts offsets from anchors (not absolute coordinates)
- Handles variable object sizes

**3. Feature Pyramid**
- Detect objects at multiple scales
- Use features from different CNN layers
- Small objects: High-resolution features
- Large objects: Low-resolution features

**4. Non-Maximum Suppression (NMS)**
- Remove duplicate detections
- Keep only highest-confidence box per object
- Essential post-processing step

**5. Efficient Architecture**
- MobileNet, EfficientNet for speed
- ResNet, ResNeXt for accuracy
- Trade-off: Speed vs accuracy

### 6.3 Performance Comparison

**State-of-the-Art (2024):**

| Model | mAP (COCO) | FPS (GPU) | Use Case |
|-------|-----------|-----------|----------|
| Faster R-CNN | 42% | 5-15 | Accuracy-critical |
| YOLOv8-nano | 37% | 100+ | Mobile/embedded |
| YOLOv8-small | 45% | 80 | Balanced |
| YOLOv8-medium | 50% | 60 | High accuracy |
| YOLOv8-large | 53% | 40 | Best accuracy |
| YOLOv8-xlarge | 55% | 30 | State-of-the-art |

**Evolution:**
```
2014: R-CNN          → 53% mAP,  0.02 FPS
2015: Fast R-CNN     → 66% mAP,  0.5 FPS
2015: Faster R-CNN   → 73% mAP,  5 FPS
2016: YOLO v1        → 63% mAP,  45 FPS
2016: SSD            → 76% mAP,  59 FPS
2018: YOLO v3        → 57% mAP,  35 FPS
2020: YOLOv5         → 67% mAP,  120 FPS
2023: YOLOv8         → 55% mAP,  80 FPS (on COCO)

Progress: 100× faster, 2× more accurate in 9 years!
```

### 6.4 What You'll Learn

**Week 14 (YOLO & SSD):**
- How YOLO divides image into grid
- Anchor boxes and predictions
- Real-time detection in action
- Training YOLO on custom dataset

**Week 15 (R-CNN Family):**
- Region Proposal Networks (RPN)
- ROI pooling and alignment
- Two-stage detection pipeline
- When to use YOLO vs R-CNN

**Foundation This Week:**
- IOU: Matching predictions to ground truth
- mAP: Evaluating detector quality
- Bounding boxes: Representing detections
- Classical methods: Appreciate modern improvements

---

## Part 7: Summary

### 7.1 Key Takeaways

**Object Detection Problem:**
- Localize AND classify multiple objects in image
- Output: {(class, box, confidence), ...} for each object
- Harder than classification: variable outputs, multiple scales, occlusion

**Bounding Boxes:**
- Three formats: XYXY, XYWH, COCO
- Normalized coordinates for resolution independence
- Ground truth annotations required for training

**IOU (Intersection over Union):**
- Fundamental metric: Overlap area / Union area
- Range [0, 1], threshold typically 0.5
- Used to match predictions to ground truth

**Evaluation Metrics:**
- **Precision:** Fraction of correct predictions
- **Recall:** Fraction of ground truth detected
- **AP:** Area under precision-recall curve (per class)
- **mAP:** Average AP across all classes
- **mAP@[0.5:0.95]:** COCO standard (strict)

**Classical vs Modern:**
- Classical: Sliding windows, hand-crafted features, 30% mAP, slow
- Modern: End-to-end CNNs, 80-90% mAP, real-time
- Two paradigms: Two-stage (accurate) vs Single-shot (fast)

### 7.2 Skills Acquired

After Week 13, you should be able to:
- ✅ Explain object detection problem and applications
- ✅ Draw and convert bounding boxes between formats
- ✅ Calculate IOU between two boxes
- ✅ Compute precision, recall, and mAP
- ✅ Interpret mAP values and compare models
- ✅ Understand why modern methods outperform classical approaches
- ✅ Appreciate the need for YOLO and R-CNN (Weeks 14-15)

### 7.3 Looking Ahead

**Week 14: YOLO & SSD (Real-Time Detection)**
- Use YOLOv8 pre-trained model
- Detect objects in images and videos
- Train YOLO on custom dataset
- Understand grid-based detection

**Week 15: R-CNN Family (High-Accuracy Detection)**
- Region Proposal Networks
- Faster R-CNN architecture
- Compare YOLO vs R-CNN
- Decision framework: When to use which?

**Final Exam Preparation:**
- Understand fundamentals (this week)
- Hands-on experience (Weeks 14-15)
- Comparison and decision-making
- Real-world application scenarios

---

## Part 8: Practice Questions

### 8.1 Two-Mark Questions

1. Define object detection. How is it different from image classification?
2. What is a bounding box? Name three coordinate formats.
3. Define IOU (Intersection over Union).
4. What does mAP stand for? What does it measure?
5. What is a True Positive in object detection?
6. Why do we normalize bounding box coordinates?
7. What is the range of IOU values?
8. What is the PASCAL VOC IOU threshold for True Positive?
9. Name two real-world applications of object detection.
10. What was the main limitation of sliding window detectors?

**Answers:**

1. Object detection localizes AND classifies multiple objects. Classification only identifies what's in the image, not where.

2. Bounding box is a rectangle enclosing an object. Formats: XYXY [x_min, y_min, x_max, y_max], XYWH [x_center, y_center, width, height], COCO [x, y, width, height].

3. IOU = Area of Intersection / Area of Union. Measures overlap between predicted and ground truth boxes.

4. mAP = mean Average Precision. Measures detector accuracy across all classes and recall levels.

5. True Positive: Prediction with IOU > threshold AND correct class label.

6. To make coordinates resolution-independent. Allows models to work on different image sizes.

7. IOU ∈ [0, 1]. 0 = no overlap, 1 = perfect match.

8. IOU > 0.5 (50% overlap).

9. Autonomous vehicles (pedestrian detection), Medical imaging (tumor detection).

10. Extremely slow: thousands of windows, redundant computation, not real-time.

### 8.2 Five-Mark Questions

1. Explain the difference between image classification, object localization, and object detection with examples.

2. Calculate IOU for the following boxes (show all steps):
   - Box A: [50, 100, 200, 300]
   - Box B: [150, 200, 350, 400]

3. Describe the sliding window approach for object detection. Why is it impractical for real-time applications?

4. Explain how precision and recall are calculated in object detection. What is the trade-off between them?

5. What is mAP@0.5 and how is it different from mAP@[0.5:0.95]? Which is more strict?

**Sample Answer (Question 2):**

**Calculate IOU:**

Given:
- Box A: [50, 100, 200, 300] (XYXY format)
- Box B: [150, 200, 350, 400]

**Step 1: Intersection coordinates**
```
x_min_inter = max(50, 150) = 150
y_min_inter = max(100, 200) = 200
x_max_inter = min(200, 350) = 200
y_max_inter = min(300, 400) = 300
```

**Step 2: Intersection area**
```
inter_width = 200 - 150 = 50
inter_height = 300 - 200 = 100
intersection = 50 × 100 = 5,000
```

**Step 3: Individual areas**
```
area_A = (200 - 50) × (300 - 100) = 150 × 200 = 30,000
area_B = (350 - 150) × (400 - 200) = 200 × 200 = 40,000
```

**Step 4: Union area**
```
union = 30,000 + 40,000 - 5,000 = 65,000
```

**Step 5: IOU**
```
IOU = 5,000 / 65,000 = 0.077 (7.7%)
```

**Interpretation:** Very low overlap, would not be considered a True Positive (IOU < 0.5).

### 8.3 Ten-Mark Questions

1. **Design a complete evaluation pipeline for an object detector:**
   - Given 100 images with ground truth annotations
   - Model outputs predictions with confidence scores
   - Calculate mAP@0.5 across 3 classes (person, car, dog)
   - Show step-by-step procedure with example calculations

2. **Compare classical sliding window approach with modern CNN-based detection:**
   - Describe both methods in detail
   - Analyze computational complexity
   - Compare accuracy and speed
   - Explain why modern methods are superior
   - Provide numerical examples

3. **Implement IOU calculation from scratch:**
   - Write detailed pseudocode for calculate_iou(box1, box2)
   - Write detailed pseudocode for vectorized calculate_iou_batch(boxes1, boxes2)
   - Analyze time complexity of both approaches
   - Explain when to use each approach
   - Provide example test cases with expected outputs

4. **Explain the precision-recall trade-off in object detection:**
   - Define precision and recall
   - Show how varying confidence threshold affects both metrics
   - Plot example precision-recall curve
   - Calculate Average Precision from the curve
   - Discuss how to choose threshold for different applications

---

## References and Further Reading

**Foundational Papers:**
1. **Object Detection Surveys:**
   - Zou et al. "Object Detection in 20 Years: A Survey" (2019)
   - Liu et al. "Deep Learning for Generic Object Detection: A Survey" (2020)

2. **Evaluation Metrics:**
   - Everingham et al. "The PASCAL Visual Object Classes Challenge" (2010)
   - Lin et al. "Microsoft COCO: Common Objects in Context" (2014)

**Classical Methods:**
3. Dalal & Triggs. "Histograms of Oriented Gradients" (2005)
4. Uijlings et al. "Selective Search for Object Recognition" (2013)

**Modern Deep Learning Methods (Preview for Weeks 14-15):**
5. Girshick et al. "R-CNN: Rich feature hierarchies" (2014)
6. Ren et al. "Faster R-CNN" (2015)
7. Redmon et al. "You Only Look Once: Unified Real-Time Detection" (2016)
8. Liu et al. "SSD: Single Shot MultiBox Detector" (2016)
9. Redmon & Farhadi. "YOLO9000: Better, Faster, Stronger" (2017)
10. Jocher et al. "YOLOv8" (2023)

**Datasets:**
- **PASCAL VOC:** http://host.robots.ox.ac.uk/pascal/VOC/
- **MS COCO:** https://cocodataset.org/
- **Open Images:** https://storage.googleapis.com/openimages/web/index.html
- **Objects365:** https://www.objects365.org/

**Online Resources:**
- TensorFlow Object Detection API: https://github.com/tensorflow/models/tree/master/research/object_detection
- Ultralytics YOLOv8: https://docs.ultralytics.com/
- COCO Evaluation Metrics: https://cocodataset.org/#detection-eval

---

## Appendix A: Common Dataset Formats

### PASCAL VOC XML Format
```xml
<annotation>
    <filename>image001.jpg</filename>
    <size>
        <width>640</width>
        <height>480</height>
    </size>
    <object>
        <name>person</name>
        <bndbox>
            <xmin>100</xmin>
            <ymin>150</ymin>
            <xmax>300</xmax>
            <ymax>450</ymax>
        </bndbox>
    </object>
    <object>
        <name>car</name>
        <bndbox>
            <xmin>350</xmin>
            <ymin>200</ymin>
            <xmax>550</xmax>
            <ymax>400</ymax>
        </bndbox>
    </object>
</annotation>
```

### COCO JSON Format
```json
{
    "images": [
        {"id": 1, "file_name": "image001.jpg", "width": 640, "height": 480}
    ],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [100, 150, 200, 300],
            "area": 60000,
            "iscrowd": 0
        }
    ],
    "categories": [
        {"id": 1, "name": "person"},
        {"id": 2, "name": "car"}
    ]
}
```

### YOLO TXT Format
```
# File: image001.txt
# Format: class_id x_center y_center width height (all normalized)
0 0.3125 0.625 0.3125 0.625
1 0.7031 0.625 0.3125 0.4167
```

---

## Appendix B: IOU Calculation Examples

**Example 1: Partial Overlap**
```
Box A: [0, 0, 100, 100]    Area = 10,000
Box B: [50, 50, 150, 150]  Area = 10,000

Intersection: [50, 50, 100, 100]  Area = 2,500
Union: 10,000 + 10,000 - 2,500 = 17,500

IOU = 2,500 / 17,500 = 0.143 (14.3%)
```

**Example 2: High Overlap**
```
Box A: [0, 0, 100, 100]    Area = 10,000
Box B: [10, 10, 90, 90]    Area = 6,400

Intersection: [10, 10, 90, 90]  Area = 6,400
Union: 10,000 + 6,400 - 6,400 = 10,000

IOU = 6,400 / 10,000 = 0.64 (64%)
```

**Example 3: No Overlap**
```
Box A: [0, 0, 50, 50]      Area = 2,500
Box B: [100, 100, 150, 150]  Area = 2,500

Intersection: None  Area = 0
Union: 2,500 + 2,500 - 0 = 5,000

IOU = 0 / 5,000 = 0.0 (0%)
```

---

**End of Lecture Notes**

**Total Pages:** ~18 pages
**Reading Time:** 60-90 minutes
**Next:** Week 14 - YOLO & SSD (Real-Time Object Detection)

**Prepared by:** Prof. Ramesh Babu
**Course:** 21CSE558T - Deep Neural Network Architectures
**Module 5:** Object Localization and Detection
**Version:** 1.0 (November 2025)
