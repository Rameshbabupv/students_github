# Week 14: YOLO & SSD - Real-Time Object Detection
## Comprehensive Lecture Notes

**Course:** 21CSE558T - Deep Neural Network Architectures
**Module:** 5 - Object Localization and Detection
**Week:** 14 of 15
**Date:** November 2025
**Instructor:** Prof. Ramesh Babu

---

## Table of Contents

1. [Introduction to Real-Time Object Detection](#part-1-introduction-to-real-time-object-detection)
2. [YOLO Architecture Deep Dive](#part-2-yolo-architecture-deep-dive)
3. [YOLO Training and Inference](#part-3-yolo-training-and-inference)
4. [SSD Architecture](#part-4-ssd-architecture)
5. [Non-Maximum Suppression (NMS)](#part-5-non-maximum-suppression-nms)
6. [Performance Analysis](#part-6-performance-analysis)
7. [Practical Applications](#part-7-practical-applications)
8. [Summary and Practice Questions](#part-8-summary-and-practice-questions)

---

## Part 1: Introduction to Real-Time Object Detection

### 1.1 The Need for Speed

In Week 13, we learned about object detection fundamentals: IOU, mAP, and classical methods (sliding windows). We saw that classical approaches were slow (1-10 seconds per image) and inaccurate (30-40% mAP).

**Real-time object detection** means processing at **30+ frames per second (FPS)**, enabling:
- **Autonomous vehicles:** Must detect pedestrians/vehicles in milliseconds
- **Robotics:** Real-time navigation and manipulation
- **Video surveillance:** Live monitoring of multiple camera feeds
- **Augmented reality:** Instant object recognition for AR experiences
- **Sports analytics:** Track players and ball in real-time

**Performance Requirements by Application:**

| Application | Required FPS | Latency Tolerance | Accuracy Requirement |
|-------------|--------------|-------------------|----------------------|
| Autonomous Driving | 30-60 FPS | <33 ms | Very High (>90% mAP) |
| Video Surveillance | 10-30 FPS | <100 ms | High (>70% mAP) |
| Robotics | 30-60 FPS | <33 ms | High (>80% mAP) |
| Mobile Apps | 15-30 FPS | <66 ms | Medium (>60% mAP) |
| Industrial Inspection | 5-15 FPS | <200 ms | Very High (>95% mAP) |

### 1.2 Evolution: From R-CNN to Single-Shot Detectors

**Timeline of Object Detection:**

```
2012: AlexNet (ImageNet winner)
  ↓
2014: R-CNN (Regions with CNN) → 53% mAP, 47 seconds/image
  - Two-stage: Selective Search → CNN classification
  - Slow but accurate
  ↓
2015: Fast R-CNN → 66% mAP, 2 seconds/image
  - Shared computation, ROI pooling
  ↓
2015: Faster R-CNN → 73% mAP, 0.2 seconds/image (5 FPS)
  - Region Proposal Network (RPN)
  - First "nearly real-time" detector
  ↓
2016: YOLO v1 (You Only Look Once) → 63% mAP, 45 FPS 🚀
  - Single-shot detection
  - Real-time breakthrough!
  ↓
2016: SSD (Single Shot MultiBox Detector) → 76% mAP, 59 FPS
  - Multi-scale feature maps
  - Faster and more accurate than YOLOv1
  ↓
2018: YOLO v3 → 57% mAP (COCO), 35 FPS
  - Multi-scale predictions
  - State-of-the-art real-time
  ↓
2020: YOLO v5 → 67% mAP (COCO), 120 FPS
  - PyTorch implementation
  - Easier to train
  ↓
2023: YOLO v8 → 53-55% mAP (COCO), 80 FPS
  - Latest state-of-the-art
  - Best accuracy-speed tradeoff
```

###1.3 Two Paradigms: Two-Stage vs Single-Shot

**Two-Stage Detectors (R-CNN Family):**
```
Stage 1: Generate region proposals (candidate boxes)
         ↓
Stage 2: Classify and refine each proposal
```

**Characteristics:**
- ✅ Higher accuracy (75-80% mAP on COCO)
- ❌ Slower (5-20 FPS)
- ✅ Better for small objects
- ❌ More complex architecture

**Examples:** R-CNN, Fast R-CNN, Faster R-CNN, Mask R-CNN

**Single-Shot Detectors (YOLO & SSD):**
```
Single CNN forward pass → Directly predict boxes + classes
```

**Characteristics:**
- ✅ Real-time speed (30-100+ FPS)
- ✅ Simpler architecture
- ❌ Slightly lower accuracy (60-75% mAP on COCO)
- ✅ Better for large objects

**Examples:** YOLO (all versions), SSD, RetinaNet

### 1.4 This Week's Focus

We'll dive deep into **two single-shot detectors**:
1. **YOLO (You Only Look Once)** - Most popular, easy to use
2. **SSD (Single Shot MultiBox Detector)** - Alternative approach

By the end of this week, you'll:
- Understand how YOLO works internally
- Use pre-trained YOLO for detection
- Train YOLO on custom datasets
- Compare YOLO and SSD
- Make informed model selection decisions

---

## Part 2: YOLO Architecture Deep Dive

### 2.1 The Core Idea: "You Only Look Once"

**Traditional approach** (Sliding Window, R-CNN):
- Look at thousands of regions separately
- Classify each region independently
- Slow and redundant

**YOLO approach:**
- Look at entire image **once**
- Divide image into grid
- Each grid cell predicts bounding boxes
- Single forward pass through network
- Fast and efficient!

**Key Insight:** "Detection as regression problem"
- Instead of classification + localization separately
- YOLO directly regresses to box coordinates and class probabilities
- End-to-end differentiable

### 2.2 Grid-Based Detection Mechanism

**Step 1: Divide image into S×S grid**
```
Example: S = 13 (YOLO v3)

┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
└──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘

13×13 = 169 grid cells
```

**Step 2: Each cell predicts B bounding boxes**
- B = 3 (typically) in YOLO v3/v8
- Each box: 5 values + C class probabilities
  - (x, y): Box center relative to cell
  - (w, h): Box width/height relative to image
  - confidence: Objectness score (is there an object?)
  - class_probs: Probability for each of C classes

**Step 3: Output tensor shape**
```
S × S × (B × (5 + C))

For YOLOv3 on COCO (80 classes):
13 × 13 × (3 × (5 + 80)) = 13 × 13 × 255
```

**Example for one cell:**
```
Cell[i,j] predicts 3 boxes:

Box 1: [x, y, w, h, confidence, p(person), p(car), ..., p(dog)]
Box 2: [x, y, w, h, confidence, p(person), p(car), ..., p(dog)]
Box 3: [x, y, w, h, confidence, p(person), p(car), ..., p(dog)]

Total: 3 × 85 = 255 values per cell
```

### 2.3 Anchor Boxes

**Problem:** Different objects have different aspect ratios
- Person: tall and narrow (w:h ≈ 1:3)
- Car: wide and short (w:h ≈ 3:1)
- Ball: square (w:h ≈ 1:1)

**Solution:** Predefined **anchor boxes** at different scales and aspect ratios

**YOLO v3/v8 uses 3 anchors per grid cell:**
- Small anchor: (10, 13) pixels
- Medium anchor: (30, 61) pixels
- Large anchor: (62, 45) pixels

**How it works:**
1. Model predicts **offsets** from anchor boxes
2. Not absolute box coordinates
3. Makes training more stable

**Mathematical Formula:**
```python
# Predicted offsets: (t_x, t_y, t_w, t_h)
# Anchor box: (p_w, p_h)
# Grid cell top-left: (c_x, c_y)

# Final box coordinates:
b_x = sigmoid(t_x) + c_x  # Center x (in grid units)
b_y = sigmoid(t_y) + c_y  # Center y (in grid units)
b_w = p_w * exp(t_w)      # Width (in pixels)
b_h = p_h * exp(t_h)      # Height (in pixels)
```

**Why sigmoid for (t_x, t_y)?**
- Constrains center to stay within grid cell
- sigmoid(t_x) ∈ [0, 1]
- Prevents boxes from wandering across cells

### 2.4 YOLO Network Architecture

**YOLOv8 Architecture (Simplified):**
```
Input Image (640×640×3)
    ↓
Backbone (CSPDarknet53)
    ├─ Conv layers
    ├─ Bottleneck blocks
    └─ Feature Pyramid Network (FPN)
    ↓
Neck (PANet)
    ├─ Bottom-up path
    ├─ Top-down path
    └─ Multi-scale feature fusion
    ↓
Head (Detection Heads at 3 scales)
    ├─ Large objects:   20×20 grid (for images 640×640)
    ├─ Medium objects:  40×40 grid
    └─ Small objects:   80×80 grid
    ↓
Predictions
    ├─ 20×20×255 (large)
    ├─ 40×40×255 (medium)
    └─ 80×80×255 (small)
    ↓
Post-processing (NMS)
    ↓
Final Detections
```

**Key Components:**

1. **Backbone (CSPDarknet53):**
   - Extracts features from input image
   - Cross-Stage Partial connections (CSP)
   - Efficient and accurate

2. **Neck (PANet - Path Aggregation Network):**
   - Fuses features from different scales
   - Bottom-up: High resolution → Low resolution
   - Top-down: Low resolution → High resolution
   - Helps detect objects at multiple scales

3. **Head (Detection Heads):**
   - Three detection layers at different scales
   - Each layer specializes in different object sizes:
     - 80×80: Small objects (< 32² pixels)
     - 40×40: Medium objects (32² - 96² pixels)
     - 20×20: Large objects (> 96² pixels)

### 2.5 Multi-Scale Predictions

**Why multiple scales?**
- Objects appear at different sizes in images
- A small bird in background vs large person in foreground
- Need different receptive fields

**YOLO v3/v8 uses 3 scales:**

```
Scale 1 (Large objects): 20×20 grid
  - Receptive field: Large (covers ~20% of image)
  - Best for: Cars, people (close-up), large animals

Scale 2 (Medium objects): 40×40 grid
  - Receptive field: Medium (covers ~10% of image)
  - Best for: People (medium distance), bikes, signs

Scale 3 (Small objects): 80×80 grid
  - Receptive field: Small (covers ~5% of image)
  - Best for: Small objects far away, details
```

**Total predictions:**
```
20×20×3 + 40×40×3 + 80×80×3
= 1,200 + 4,800 + 19,200
= 25,200 anchor boxes evaluated!
```

**But we only keep ~100 detections after NMS**

### 2.6 Loss Function

YOLO minimizes a multi-part loss function:

**Total Loss = Localization Loss + Objectness Loss + Classification Loss**

**1. Localization Loss (Box Regression):**
```
L_box = Σ 1_{obj}^{ij} [(x - x̂)² + (y - ŷ)²]  # Center
      + Σ 1_{obj}^{ij} [(√w - √ŵ)² + (√h - √ĥ)²]  # Size

Where:
- 1_{obj}^{ij}: Indicator (1 if object in cell i, box j)
- (x, y, w, h): Predicted box
- (x̂, ŷ, ŵ, ĥ): Ground truth box
- Square roots: Penalize errors in small boxes more
```

**2. Objectness Loss (Confidence):**
```
L_obj = Σ 1_{obj}^{ij} (C - Ĉ)²  # If object present
      + λ_noobj Σ 1_{noobj}^{ij} (C - Ĉ)²  # If no object

Where:
- C: Predicted confidence
- Ĉ: Ground truth (1 if object, 0 if background)
- λ_noobj: Weight for background (typically 0.5)
  - Less important than object predictions
```

**3. Classification Loss (Class Probabilities):**
```
L_class = Σ 1_{obj}^{i} Σ_{c ∈ classes} (p_c - p̂_c)²

Where:
- p_c: Predicted probability for class c
- p̂_c: Ground truth (1 for correct class, 0 otherwise)
```

**Combined Loss:**
```
L_total = λ_coord × L_box + L_obj + λ_class × L_class

Typical weights:
- λ_coord = 5.0 (localization is important)
- λ_class = 1.0
```

**Modern YOLO (v8) uses:**
- **CIoU Loss** (Complete IoU) for bounding boxes
- **Binary Cross-Entropy** for objectness and classification
- More sophisticated than original YOLO

### 2.7 YOLO Evolution: v1 → v3 → v5 → v8

**YOLOv1 (2016):**
- First real-time detector
- 7×7 grid, 2 boxes per cell
- 63% mAP, 45 FPS
- Limitations: Struggled with small objects

**YOLOv2 (YOLO9000, 2017):**
- Batch normalization
- Anchor boxes introduced
- Multi-scale training
- 78.6% mAP, 40 FPS

**YOLOv3 (2018):**
- Multi-scale predictions (3 scales)
- Darknet-53 backbone
- Better small object detection
- 57% mAP (COCO), 35 FPS

**YOLOv5 (2020):**
- PyTorch implementation (v1-v3 were Darknet/C)
- Easier to train and deploy
- Better data augmentation (Mosaic)
- 67% mAP (COCO), 120 FPS

**YOLOv8 (2023):**
- State-of-the-art accuracy
- Anchor-free design (predicts box centers directly)
- Improved backbone (CSPDarknet53)
- Multiple variants: yolov8n/s/m/l/x
- 53-55% mAP (COCO mAP@0.5:0.95), 80 FPS

**Comparison Table:**

| Version | Year | mAP (COCO) | FPS | Backbone | Key Innovation |
|---------|------|------------|-----|----------|----------------|
| YOLOv1 | 2016 | 63% @0.5 | 45 | Custom 24-layer | Grid detection |
| YOLOv2 | 2017 | 78.6% @0.5 | 40 | Darknet-19 | Anchor boxes |
| YOLOv3 | 2018 | 57% @0.5:0.95 | 35 | Darknet-53 | Multi-scale |
| YOLOv5 | 2020 | 67% @0.5:0.95 | 120 | CSPDarknet | PyTorch, Mosaic |
| YOLOv8 | 2023 | 53-55% @0.5:0.95 | 80 | CSPDarknet53 | Anchor-free |

---

## Part 3: YOLO Training and Inference

### 3.1 Data Preparation

**YOLO Dataset Format:**

**Directory Structure:**
```
dataset/
├── images/
│   ├── train/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   │   ├── img001.txt
│   │   ├── img002.txt
│   │   └── ...
│   ├── val/
│   └── test/
└── data.yaml
```

**Label File Format (.txt):**
```
# Each line: class_id x_center y_center width height
# All values normalized to [0, 1]

0 0.5 0.5 0.3 0.4     # Person at center, size 0.3×0.4
2 0.2 0.3 0.1 0.15    # Car in upper-left
0 0.7 0.8 0.2 0.25    # Another person
```

**Example Calculation:**
```
Image size: 640×480 pixels
Object: Person
  Bounding box (pixels): [200, 150, 400, 350] (XYXY format)

Convert to YOLO format:
  x_center = ((200 + 400) / 2) / 640 = 300 / 640 = 0.469
  y_center = ((150 + 350) / 2) / 480 = 250 / 480 = 0.521
  width = (400 - 200) / 640 = 200 / 640 = 0.3125
  height = (350 - 150) / 480 = 200 / 480 = 0.4167

YOLO label: 0 0.469 0.521 0.3125 0.4167
```

**data.yaml Configuration:**
```yaml
# Dataset paths
path: /path/to/dataset
train: images/train
val: images/val
test: images/test  # optional

# Class names
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  # ... 75 more classes for COCO

# Number of classes
nc: 80
```

### 3.2 Transfer Learning

**Starting from Scratch vs Pre-trained:**

**From Scratch:**
- ❌ Requires millions of images
- ❌ Weeks of training time
- ❌ Often lower accuracy
- ✅ No dependency on ImageNet

**Transfer Learning (RECOMMENDED):**
- ✅ Start with pre-trained weights (COCO dataset)
- ✅ Fine-tune on your data
- ✅ Faster training (minutes to hours)
- ✅ Better accuracy with less data
- ✅ Works with as few as 100-500 images

**YOLOv8 Pre-trained Models:**
```python
from ultralytics import YOLO

# Load pre-trained model (trained on COCO)
model = YOLO('yolov8n.pt')  # Nano (6 MB)
model = YOLO('yolov8s.pt')  # Small (22 MB)
model = YOLO('yolov8m.pt')  # Medium (52 MB)
model = YOLO('yolov8l.pt')  # Large (88 MB)
model = YOLO('yolov8x.pt')  # XLarge (136 MB)

# Train on your custom dataset
results = model.train(
    data='custom_data.yaml',
    epochs=50,
    imgsz=640
)
```

### 3.3 Training Hyperparameters

**Key Hyperparameters:**

**1. Image Size (`imgsz`):**
- Default: 640×640
- Options: 320, 416, 640, 1280
- Larger = more accurate, slower
- Must be multiple of 32 (due to downsampling)

**2. Batch Size (`batch`):**
- Default: 16
- GPU memory dependent
- Larger = faster training, better gradients
- If OOM error → reduce batch size

**3. Epochs:**
- Default: 100-300
- More epochs = better fit (risk overfitting)
- Use early stopping (patience=10-50)

**4. Learning Rate (`lr0`):**
- Default: 0.01 (auto-adjusted)
- Too high → unstable training
- Too low → slow convergence

**5. Momentum (`momentum`):**
- Default: 0.937
- Helps optimization

**6. Weight Decay (`weight_decay`):**
- Default: 0.0005
- Regularization term

**Example Training Command:**
```python
results = model.train(
    data='hardhat.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    lr0=0.01,
    patience=20,  # Early stopping
    device=0,  # GPU 0 (use 'cpu' for CPU)
    workers=8,  # Data loading threads
    save=True,  # Save checkpoints
    project='runs/detect',
    name='hardhat_detector'
)
```

### 3.4 Data Augmentation

YOLO v8 includes powerful built-in augmentation:

**Geometric Augmentations:**
1. **Mosaic (default):** Combines 4 images into one
   - Forces model to learn partial objects
   - Increases effective batch size

2. **Random Flip:** Horizontal (50% probability)

3. **Random Scale & Crop:** ±50% zoom

4. **Random Rotation:** ±10°

5. **Random Translation:** ±10% shift

**Photometric Augmentations:**
1. **HSV Color Jitter:**
   - Hue: ±0.015
   - Saturation: ±0.7
   - Value: ±0.4

2. **Random Brightness/Contrast**

**Advanced Augmentations:**
1. **MixUp:** Blend two images and labels
2. **Copy-Paste:** Copy objects from one image to another
3. **Cutout:** Random rectangular occlusion

**Configuration:**
```python
results = model.train(
    data='data.yaml',
    # Augmentation parameters
    hsv_h=0.015,  # HSV-Hue augmentation
    hsv_s=0.7,    # HSV-Saturation
    hsv_v=0.4,    # HSV-Value
    degrees=10.0, # Rotation (±deg)
    translate=0.1, # Translation (±fraction)
    scale=0.5,    # Scale (±fraction)
    shear=0.0,    # Shear (±deg)
    perspective=0.0, # Perspective
    flipud=0.0,   # Flip up-down probability
    fliplr=0.5,   # Flip left-right probability
    mosaic=1.0,   # Mosaic probability
    mixup=0.1     # MixUp probability
)
```

### 3.5 Inference Pipeline

**Step-by-Step Inference:**

```python
from ultralytics import YOLO

# 1. Load trained model
model = YOLO('best.pt')  # or yolov8n.pt for pre-trained

# 2. Run inference
results = model('image.jpg', conf=0.5, iou=0.45)

# 3. Access predictions
for result in results:
    boxes = result.boxes  # Bounding boxes
    for box in boxes:
        # Get box coordinates (XYXY format)
        x1, y1, x2, y2 = box.xyxy[0]

        # Get confidence and class
        confidence = float(box.conf)
        class_id = int(box.cls)
        class_name = model.names[class_id]

        print(f"{class_name}: {confidence:.2f} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

# 4. Visualize
annotated = results[0].plot()  # Draw boxes on image
```

**Batch Inference:**
```python
# Process multiple images
results = model(['img1.jpg', 'img2.jpg', 'img3.jpg'])

# Process video
results = model('video.mp4', save=True)

# Process webcam
results = model(source=0, show=True)  # 0 = default webcam
```

**Confidence Thresholding:**
```python
# Only keep detections with confidence > 0.7
results = model('image.jpg', conf=0.7)
```

### 3.6 Model Export and Deployment

**Export Formats:**
```python
# Export to different formats
model.export(format='onnx')      # ONNX (cross-platform)
model.export(format='torchscript') # TorchScript
model.export(format='tflite')    # TensorFlow Lite (mobile)
model.export(format='coreml')    # CoreML (iOS)
model.export(format='engine')    # TensorRT (NVIDIA GPUs)
```

**Deployment Platforms:**
- **Cloud:** AWS, Google Cloud, Azure
- **Edge:** NVIDIA Jetson, Raspberry Pi, Intel NCS
- **Mobile:** iOS (CoreML), Android (TFLite)
- **Web:** ONNX.js (browser-based)

---

## Part 4: SSD Architecture

### 4.1 SSD: Single Shot MultiBox Detector

**Published:** 2016 (same year as YOLO v1)
**Authors:** Wei Liu et al. (UNC Chapel Hill, Google)
**Key Idea:** Multi-scale feature maps for detection

**SSD vs YOLO (High-Level):**
- **YOLO:** Custom backbone, 3 detection scales
- **SSD:** VGG16 backbone, 6 detection scales
- **YOLO:** 3 anchors per location
- **SSD:** 4-6 default boxes per location
- **YOLO:** Faster (45-80 FPS)
- **SSD:** Slightly more accurate on small objects

### 4.2 SSD Architecture

**SSD300 Architecture (300×300 input):**
```
Input (300×300×3)
    ↓
VGG16 Base Network (truncated, no FC layers)
    ├─ Conv1_1, Conv1_2 → Pool1
    ├─ Conv2_1, Conv2_2 → Pool2
    ├─ Conv3_1, Conv3_2, Conv3_3 → Pool3
    ├─ Conv4_1, Conv4_2, Conv4_3 → Pool4 (used for detection!)
    └─ Conv5_1, Conv5_2, Conv5_3 → Pool5
    ↓
Extra Feature Layers
    ├─ Conv6 (FC6 converted to conv)
    ├─ Conv7 (FC7 converted to conv) → (19×19, used for detection!)
    ├─ Conv8_1, Conv8_2 → (10×10, used for detection!)
    ├─ Conv9_1, Conv9_2 → (5×5, used for detection!)
    ├─ Conv10_1, Conv10_2 → (3×3, used for detection!)
    └─ Conv11_1, Conv11_2 → (1×1, used for detection!)
    ↓
Multi-Scale Detections (6 feature maps)
    ├─ 38×38 (Conv4_3)   → 4 boxes/location → 5,776 boxes
    ├─ 19×19 (Conv7)     → 6 boxes/location → 2,166 boxes
    ├─ 10×10 (Conv8_2)   → 6 boxes/location → 600 boxes
    ├─ 5×5 (Conv9_2)     → 6 boxes/location → 150 boxes
    ├─ 3×3 (Conv10_2)    → 4 boxes/location → 36 boxes
    └─ 1×1 (Conv11_2)    → 4 boxes/location → 4 boxes
    ↓
Total: 8,732 default boxes!
    ↓
Non-Maximum Suppression
    ↓
Final Detections (~100)
```

**Key Insight:** Use features from multiple layers
- Early layers (38×38): High resolution → detect small objects
- Later layers (1×1): Low resolution → detect large objects

### 4.3 Default Boxes (Anchors)

**SSD uses pre-defined "default boxes" at each location:**

**Scales:**
```python
# 6 feature maps = 6 scales
s_min = 0.2  # 20% of image size
s_max = 0.9  # 90% of image size

scales = [s_min + (s_max - s_min) * k / (6 - 1) for k in range(6)]
# scales ≈ [0.2, 0.34, 0.48, 0.62, 0.76, 0.9]

For 300×300 input:
  Layer 1 (38×38): box size ≈ 30×30 pixels
  Layer 2 (19×19): box size ≈ 60×60 pixels
  ...
  Layer 6 (1×1): box size ≈ 300×300 pixels
```

**Aspect Ratios:**
```python
aspect_ratios = [1, 2, 3, 1/2, 1/3]

For each location, generate boxes with:
  - Width = s × √(aspect_ratio)
  - Height = s / √(aspect_ratio)

Example for s=0.3, aspect_ratio=2:
  - Width = 0.3 × √2 ≈ 0.42
  - Height = 0.3 / √2 ≈ 0.21
```

**Total Boxes Calculation:**
```
38×38×4 + 19×19×6 + 10×10×6 + 5×5×6 + 3×3×4 + 1×1×4
= 5,776 + 2,166 + 600 + 150 + 36 + 4
= 8,732 default boxes
```

### 4.4 SSD Loss Function

**Similar to YOLO, but with hard negative mining:**

**Total Loss = Localization Loss + Confidence Loss**

**1. Localization Loss (Smooth L1):**
```
L_loc = Σ_{i ∈ Pos} Σ_{m ∈ {cx,cy,w,h}} smooth_L1(l_i^m - ĝ_i^m)

Where:
- Pos: Positive matches (IoU > 0.5 with ground truth)
- l: Predicted offsets
- ĝ: Ground truth offsets (encoded)
- smooth_L1: Less sensitive to outliers than L2

smooth_L1(x) = {
  0.5x²        if |x| < 1
  |x| - 0.5    otherwise
}
```

**2. Confidence Loss (Cross-Entropy):**
```
L_conf = -Σ_{i ∈ Pos} log(ĉ_i^p) - Σ_{i ∈ Neg} log(ĉ_i^0)

Where:
- ĉ_i^p: Confidence for predicted class
- ĉ_i^0: Confidence for background class
```

**Hard Negative Mining:**
- Problem: 99% of default boxes are background
- Solution: Keep only hard negatives
- Algorithm:
  1. Sort negative boxes by confidence loss
  2. Keep top N negatives (ratio 3:1 with positives)
  3. Ignore easy negatives

**Combined Loss:**
```
L = (L_conf + α × L_loc) / N

Where:
- α = 1 (weight for localization)
- N = number of matched default boxes
```

### 4.5 SSD300 vs SSD512

**SSD300:**
- Input: 300×300
- Speed: 59 FPS (GTX Titan X)
- mAP: 74.3% (PASCAL VOC)
- Best for: Real-time applications

**SSD512:**
- Input: 512×512
- Speed: 22 FPS
- mAP: 76.8% (PASCAL VOC)
- Best for: Higher accuracy needed

**Trade-off:**
- Larger input → Better for small objects
- Larger input → Slower inference

---

## Part 5: Non-Maximum Suppression (NMS)

### 5.1 The Duplicate Detection Problem

**Problem:** Object detectors generate thousands of predictions
- YOLO: 25,200 predictions (before NMS)
- SSD: 8,732 predictions (before NMS)
- Many boxes detect the same object!

**Example:**
```
Same person detected by 20 different boxes:
├─ Box 1: confidence 0.95, IoU 0.85 with ground truth
├─ Box 2: confidence 0.92, IoU 0.78
├─ Box 3: confidence 0.88, IoU 0.72
├─ ...
└─ Box 20: confidence 0.51, IoU 0.55

We want to keep ONLY Box 1 (highest confidence)!
```

### 5.2 NMS Algorithm

**Non-Maximum Suppression (NMS):**

**Input:**
- List of predicted boxes: {B₁, B₂, ..., Bₙ}
- Confidence scores: {s₁, s₂, ..., sₙ}
- IoU threshold: τ (typically 0.45-0.5)

**Algorithm:**
```python
def nms(boxes, scores, iou_threshold=0.45):
    """
    Non-Maximum Suppression

    Args:
        boxes: List of bounding boxes [x1, y1, x2, y2]
        scores: Confidence scores for each box
        iou_threshold: IoU threshold for suppression

    Returns:
        keep: Indices of boxes to keep
    """
    # Sort boxes by confidence (descending)
    indices = scores.argsort()[::-1]

    keep = []

    while len(indices) > 0:
        # Pick box with highest confidence
        current = indices[0]
        keep.append(current)

        # Compute IoU with remaining boxes
        ious = compute_iou(boxes[current], boxes[indices[1:]])

        # Remove boxes with IoU > threshold
        indices = indices[1:][ious <= iou_threshold]

    return keep
```

**Step-by-Step Example:**
```
Initial boxes (sorted by confidence):
  B1: score=0.95, box=[100, 100, 200, 200]
  B2: score=0.90, box=[105, 105, 205, 205]  # High overlap with B1
  B3: score=0.85, box=[300, 300, 400, 400]  # Different object
  B4: score=0.80, box=[310, 305, 410, 405]  # High overlap with B3
  B5: score=0.75, box=[110, 95, 210, 195]   # High overlap with B1

Step 1: Pick B1 (highest score), keep=[B1]
Step 2: Compute IoU(B1, B2)=0.82 > 0.45 → suppress B2
        Compute IoU(B1, B3)=0.0 < 0.45 → keep B3
        Compute IoU(B1, B4)=0.0 < 0.45 → keep B4
        Compute IoU(B1, B5)=0.75 > 0.45 → suppress B5

Step 3: Pick B3 (next highest), keep=[B1, B3]
Step 4: Compute IoU(B3, B4)=0.78 > 0.45 → suppress B4

Final: keep=[B1, B3]
```

### 5.3 Class-Specific vs Class-Agnostic NMS

**Class-Specific NMS (Default):**
- Apply NMS separately for each class
- Box can detect multiple classes simultaneously
- Used by YOLO and SSD

**Class-Agnostic NMS:**
- Apply NMS regardless of class
- One box = one object only
- Used in some Faster R-CNN implementations

**Example:**
```
Overlapping boxes:
  Box A: "person" (0.9), "chair" (0.2)
  Box B: "chair" (0.85)

Class-Specific NMS:
  - Keep Box A for "person" (higher confidence)
  - Keep Box B for "chair" (higher confidence)
  - Result: 2 detections

Class-Agnostic NMS:
  - Keep Box A only (overall higher confidence)
  - Result: 1 detection ("person")
```

### 5.4 Soft-NMS

**Problem with Standard NMS:**
- Hard suppression: IoU > threshold → completely remove
- What if two objects genuinely overlap?

**Soft-NMS:**
- Don't remove, just **reduce confidence**
- Boxes with high IoU get penalized, not eliminated

**Algorithm:**
```python
def soft_nms(boxes, scores, iou_threshold=0.5, sigma=0.5):
    """
    Soft Non-Maximum Suppression
    """
    indices = scores.argsort()[::-1]
    keep = []

    while len(indices) > 0:
        current = indices[0]
        keep.append(current)

        # Compute IoU with remaining boxes
        ious = compute_iou(boxes[current], boxes[indices[1:]])

        # Decay scores instead of removing
        # Gaussian decay: s_i = s_i × exp(-(iou²/sigma))
        scores[indices[1:]] *= np.exp(-(ious ** 2) / sigma)

        # Remove boxes with very low scores
        indices = indices[1:][scores[indices[1:]] > 0.01]

    return keep
```

**Soft-NMS improves mAP by ~1-2%** in crowded scenes

---

## Part 6: Performance Analysis

### 6.1 Speed Benchmarks

**Hardware:** NVIDIA Tesla T4 GPU (Google Colab)

| Model | Input Size | FPS (GPU) | FPS (CPU) | Latency (ms) |
|-------|------------|-----------|-----------|--------------|
| YOLOv8n | 640×640 | 120 | 15 | 8.3 |
| YOLOv8s | 640×640 | 100 | 10 | 10.0 |
| YOLOv8m | 640×640 | 80 | 5 | 12.5 |
| YOLOv8l | 640×640 | 60 | 3 | 16.7 |
| YOLOv8x | 640×640 | 40 | 2 | 25.0 |
| SSD MobileNet | 300×300 | 85 | 18 | 11.8 |
| SSD ResNet50 | 512×512 | 30 | 4 | 33.3 |

**Key Observations:**
- **YOLOv8n:** Fastest, suitable for mobile/edge devices
- **YOLOv8m:** Best balance of speed and accuracy
- **SSD MobileNet:** Competitive with YOLO on speed
- **GPU acceleration:** 5-10× speedup over CPU

### 6.2 Accuracy Benchmarks

**Dataset:** MS COCO val2017 (5,000 images, 80 classes)

| Model | mAP@0.5 | mAP@0.5:0.95 | AP_small | AP_medium | AP_large |
|-------|---------|--------------|----------|-----------|----------|
| YOLOv8n | 52.5% | 37.3% | 18.5% | 41.2% | 51.1% |
| YOLOv8s | 61.8% | 44.9% | 24.7% | 49.3% | 61.5% |
| YOLOv8m | 67.2% | 50.2% | 31.8% | 55.1% | 66.2% |
| YOLOv8l | 70.5% | 52.9% | 35.2% | 58.3% | 69.8% |
| YOLOv8x | 72.9% | 53.9% | 37.1% | 59.7% | 70.9% |
| SSD300 | 68.0% | 41.2% | 9.9% | 44.2% | 59.4% |
| SSD512 | 71.5% | 46.5% | 18.8% | 51.3% | 64.2% |

**Interpretation:**
- **mAP@0.5:0.95:** COCO standard (strict)
  - YOLOv8x: 53.9% → State-of-the-art for real-time
  - SSD512: 46.5% → Good but behind YOLO

- **Small Objects (area < 32²):**
  - YOLOv8x: 37.1%
  - SSD512: 18.8%
  - YOLO significantly better!

- **Large Objects (area > 96²):**
  - Both perform well (60-70%)

### 6.3 Speed-Accuracy Tradeoff

**Visualization:**
```
Accuracy (mAP@0.5:0.95)
60% ├──────────────────────────────────────────┐
    │                                    YOLOv8x●
55% │                               YOLOv8l●
    │                          YOLOv8m●
50% │                     YOLOv8s●  SSD512●
    │                YOLOv8n●
45% │           SSD300●
    │
40% └───────────────────────────────────────────
    0    20   40   60   80  100  120  140  FPS
                    Speed →
```

**Key Trade-offs:**
1. **YOLOv8n:** Fastest (120 FPS) but lowest accuracy (37%)
2. **YOLOv8m:** Best balance (80 FPS, 50% mAP)
3. **YOLOv8x:** Highest accuracy (54%) but slower (40 FPS)
4. **SSD:** Competitive but generally behind YOLO v8

**Choosing the Right Model:**
```
Application              Model        Reason
─────────────────────────────────────────────────────
Autonomous Driving       YOLOv8m/l    Balance speed + accuracy
Mobile App               YOLOv8n      Small, fast
High-Accuracy Offline    YOLOv8x      Best mAP
Video Surveillance       YOLOv8s      Real-time, moderate accuracy
Robotics (edge device)   YOLOv8n      Fast inference, low memory
Industrial Inspection    YOLOv8l/x    High accuracy critical
```

### 6.4 Model Size and Memory

| Model | Parameters | Model Size | GPU Memory (Training) | GPU Memory (Inference) |
|-------|------------|------------|----------------------|------------------------|
| YOLOv8n | 3.2M | 6 MB | ~2 GB | ~500 MB |
| YOLOv8s | 11.2M | 22 MB | ~3 GB | ~800 MB |
| YOLOv8m | 25.9M | 52 MB | ~5 GB | ~1.5 GB |
| YOLOv8l | 43.7M | 88 MB | ~7 GB | ~2.0 GB |
| YOLOv8x | 68.2M | 136 MB | ~10 GB | ~3.0 GB |
| SSD MobileNet | 6.9M | 20 MB | ~2 GB | ~600 MB |
| SSD ResNet50 | 35.0M | 90 MB | ~6 GB | ~1.8 GB |

**Deployment Considerations:**
- **Mobile:** Use YOLOv8n or SSD MobileNet (< 25 MB)
- **Edge (Jetson):** YOLOv8s/m (good balance)
- **Cloud/Server:** YOLOv8l/x (maximize accuracy)

### 6.5 YOLO vs SSD vs Faster R-CNN

**Comprehensive Comparison:**

| Feature | YOLO v8 | SSD | Faster R-CNN |
|---------|---------|-----|--------------|
| **Speed** | 40-120 FPS | 20-85 FPS | 5-15 FPS |
| **Accuracy (mAP)** | 50-54% | 41-46% | 42-45% |
| **Architecture** | Single-shot | Single-shot | Two-stage |
| **Backbone** | CSPDarknet | VGG16/ResNet | ResNet/VGG |
| **Training** | Easy (PyTorch) | Medium (TF/PyTorch) | Complex |
| **Deployment** | Very Easy | Easy | Medium |
| **Small Objects** | Good | Medium | Best |
| **Real-Time** | Yes | Yes | No |
| **Community** | Very Active | Active | Active |
| **Updates** | Frequent | Moderate | Slow |
| **Use Case** | General-purpose | TensorFlow projects | High accuracy needed |

**When to Choose:**
- **YOLO:** Most projects (best all-around)
- **SSD:** Legacy TensorFlow projects, specific requirements
- **Faster R-CNN:** Offline, accuracy-critical (medical imaging)

---

## Part 7: Practical Applications

### 7.1 Autonomous Vehicles

**Requirements:**
- Real-time: 30+ FPS
- High accuracy: 90%+ mAP
- Multi-class: Pedestrians, vehicles, cyclists, traffic signs

**Model Choice:** YOLOv8m/l
- Speed: 60-80 FPS (real-time)
- Accuracy: 50-53% mAP (high)
- Multi-scale: Good for near and far objects

**Example: Tesla Autopilot**
- Uses custom CNN (not YOLO, but similar single-shot approach)
- 8 cameras, 360° coverage
- Detects: Cars, pedestrians, lanes, traffic lights, signs

**Implementation Considerations:**
- **Redundancy:** Multiple models for safety
- **Hardware:** NVIDIA Drive AGX (high-performance GPU)
- **Latency:** < 100 ms end-to-end
- **Robustness:** Weather, lighting conditions

### 7.2 Video Surveillance

**Requirements:**
- Real-time on multiple streams
- Person detection and tracking
- Anomaly detection (intrusion, abandoned objects)

**Model Choice:** YOLOv8n/s
- Speed: 100-120 FPS (can handle 4-6 streams per GPU)
- Accuracy: 37-45% mAP (sufficient for person detection)
- Low latency: < 10 ms

**Example: Smart City Surveillance**
- 100+ cameras citywide
- Real-time alerts for incidents
- Privacy-preserving (on-device processing)

### 7.3 Retail Analytics

**Requirements:**
- Product detection on shelves
- Customer counting and tracking
- Queue detection

**Model Choice:** YOLOv8s/m
- Accuracy: Important for inventory
- Speed: 60-100 FPS

**Example: Amazon Go**
- Cashier-less stores
- Computer vision tracks items picked
- Automatic checkout

### 7.4 Industrial Inspection

**Requirements:**
- Defect detection (screws, cracks, misalignments)
- Very high accuracy: 95%+ needed
- Real-time on assembly line

**Model Choice:** YOLOv8l/x
- Accuracy: 52-54% mAP (general), fine-tuned to 90%+ on specific defects
- Speed: 40-60 FPS (sufficient for assembly line)

**Example: Automotive Manufacturing**
- Inspect car parts for defects
- 100% automated quality control
- Reduces human error

### 7.5 Medical Imaging

**Requirements:**
- Tumor/lesion detection
- Very high accuracy and recall (can't miss cancers!)
- Offline (speed less critical)

**Model Choice:** Faster R-CNN or YOLOv8x + post-processing
- Accuracy: Prioritize over speed
- Recall: Critical (better to have false positives than false negatives)

**Example: Lung Nodule Detection**
- CT scan slices analyzed
- YOLO detects candidate nodules
- Radiologist reviews detections
- Improves early cancer detection

---

## Part 8: Summary and Practice Questions

### 8.1 Key Takeaways

**YOLO (You Only Look Once):**
- ✅ Single-shot detector (one forward pass)
- ✅ Grid-based detection (S×S cells)
- ✅ Anchor boxes for multiple scales/aspect ratios
- ✅ Multi-scale predictions (3 detection layers)
- ✅ Real-time speed (30-120 FPS)
- ✅ Easy to train and deploy (PyTorch, Ultralytics)
- ✅ State-of-the-art accuracy (50-54% mAP on COCO)

**SSD (Single Shot MultiBox Detector):**
- ✅ Multi-scale feature maps (6 scales)
- ✅ Default boxes (8,732 anchors!)
- ✅ VGG16/ResNet backbone
- ✅ Real-time speed (20-85 FPS)
- ✅ Good accuracy (41-46% mAP on COCO)
- ⚠️ Struggles with small objects compared to YOLO

**Non-Maximum Suppression (NMS):**
- Removes duplicate detections
- Keeps highest-confidence box per object
- Class-specific or class-agnostic
- Soft-NMS variant for crowded scenes

**Applications:**
- Autonomous vehicles, surveillance, retail, medical imaging, robotics, sports analytics

### 8.2 Two-Mark Questions

1. What does "You Only Look Once" (YOLO) mean?
2. How many anchor boxes does YOLO v8 use per grid cell?
3. What is the purpose of Non-Maximum Suppression (NMS)?
4. Name three real-world applications of real-time object detection.
5. What is the difference between single-shot and two-stage detectors?
6. How many feature maps does SSD300 use for detection?
7. What is an anchor box?
8. Why do we need multi-scale predictions in YOLO?
9. What is Hard Negative Mining in SSD?
10. What is the typical IoU threshold for NMS?

**Answers:**

1. Single forward pass through CNN to detect all objects (no region proposals needed).

2. 3 anchor boxes per grid cell (at each of 3 detection scales).

3. Remove duplicate/overlapping detections, keep only the highest-confidence box per object.

4. Autonomous driving, video surveillance, retail analytics, robotics, medical imaging (any 3).

5. Single-shot: One pass (YOLO, SSD). Two-stage: First generate proposals, then classify (R-CNN).

6. 6 feature maps at different scales (38×38, 19×19, 10×10, 5×5, 3×3, 1×1).

7. Pre-defined box at specific scale/aspect ratio; model predicts offsets from anchors.

8. To detect objects at different sizes (small, medium, large) in the same image.

9. Selecting hardest negative examples (highest loss) to balance positive/negative samples during training.

10. Typically 0.45-0.5 (IoU > threshold → suppress).

### 8.3 Five-Mark Questions

1. Explain how YOLO divides an image into a grid and makes predictions. Include anchor boxes.

2. Describe the three components of YOLO's loss function with mathematical notation.

3. Compare YOLO and SSD architectures. What are the key differences?

4. Explain the Non-Maximum Suppression (NMS) algorithm step-by-step with an example.

5. What is transfer learning in the context of YOLO training? Why is it beneficial?

**Sample Answer (Question 1):**

**YOLO Grid-Based Detection:**

YOLO divides the input image into an S×S grid (e.g., S=13 for a 416×416 image). Each grid cell is responsible for detecting objects whose center falls within that cell.

**Predictions Per Cell:**
- Each cell predicts B bounding boxes (B=3 in YOLO v3/v8)
- Each box has 5 + C values:
  - (x, y): Box center coordinates relative to cell
  - (w, h): Box width and height relative to image
  - confidence: Objectness score (probability that cell contains an object)
  - C class probabilities: P(class|object)

**Anchor Boxes:**
- Pre-defined boxes at different scales/aspect ratios
- YOLO predicts offsets (Δx, Δy, Δw, Δh) from anchor boxes
- Makes learning more stable (predict small adjustments, not absolute coordinates)
- Example anchors: (10,13), (30,61), (62,45) pixels

**Output Tensor:**
- Shape: S × S × (B × (5 + C))
- For COCO (C=80): 13 × 13 × (3 × 85) = 13 × 13 × 255
- Total predictions: 13×13×3 = 507 boxes from one detection layer
- YOLO v8 uses 3 scales: 20×20, 40×40, 80×80 → 25,200 total predictions
- After NMS: ~100 final detections

### 8.4 Ten-Mark Questions

1. **Design a complete YOLO-based object detection system for autonomous vehicles:**
   - Dataset preparation and annotation
   - Model selection and training strategy
   - Inference pipeline and post-processing
   - Deployment considerations (hardware, latency)
   - Safety and redundancy measures

2. **Implement and explain the YOLO loss function:**
   - Write mathematical formulas for all components
   - Explain localization, objectness, and classification losses
   - Provide pseudocode for loss calculation
   - Discuss why certain design choices were made (e.g., square roots for w/h)

3. **Compare YOLO, SSD, and Faster R-CNN comprehensively:**
   - Architecture diagrams for each
   - Speed and accuracy benchmarks (with actual numbers)
   - Pros and cons of each approach
   - Recommendations for different applications
   - Future trends in object detection

4. **Explain the complete inference pipeline for YOLO:**
   - Image preprocessing and normalization
   - Forward pass through network
   - Decoding predictions (grid → absolute coordinates)
   - Confidence thresholding
   - Non-Maximum Suppression (detailed algorithm)
   - Post-processing and visualization

---

## References and Further Reading

**Foundational Papers:**

1. **YOLO v1:** Redmon et al. "You Only Look Once: Unified, Real-Time Object Detection" (2016)
   - https://arxiv.org/abs/1506.02640

2. **YOLO v2 (YOLO9000):** Redmon & Farhadi. "YOLO9000: Better, Faster, Stronger" (2017)
   - https://arxiv.org/abs/1612.08242

3. **YOLO v3:** Redmon & Farhadi. "YOLOv3: An Incremental Improvement" (2018)
   - https://arxiv.org/abs/1804.02767

4. **SSD:** Liu et al. "SSD: Single Shot MultiBox Detector" (2016)
   - https://arxiv.org/abs/1512.02325

**Modern Implementations:**

5. **YOLOv8:** Ultralytics official documentation
   - https://docs.ultralytics.com/

6. **YOLOv5:** https://github.com/ultralytics/yolov5

**Tutorials and Guides:**

7. **Object Detection Tutorial:** TensorFlow official guide
   - https://www.tensorflow.org/tutorials/images/object_detection

8. **YOLO Training Tutorial:** Roboflow blog
   - https://blog.roboflow.com/how-to-train-yolov8/

**Datasets:**

9. **MS COCO:** https://cocodataset.org/
10. **Pascal VOC:** http://host.robots.ox.ac.uk/pascal/VOC/
11. **Open Images V7:** https://storage.googleapis.com/openimages/web/index.html

**Tools:**

12. **Ultralytics YOLOv8:** https://github.com/ultralytics/ultralytics
13. **TensorFlow Object Detection API:** https://github.com/tensorflow/models/tree/master/research/object_detection
14. **Annotation Tools:**
    - labelImg: https://github.com/heartexlabs/labelImg
    - Roboflow: https://roboflow.com/
    - CVAT: https://github.com/openvinotoolkit/cvat

---

**End of Lecture Notes**

**Total Pages:** ~23 pages
**Reading Time:** 90-120 minutes
**Next:** Week 15 - R-CNN Family (Two-Stage Object Detection)

**Prepared by:** Prof. Ramesh Babu
**Course:** 21CSE558T - Deep Neural Network Architectures
**Module 5:** Object Localization and Detection
**Version:** 1.0 (November 2025)
