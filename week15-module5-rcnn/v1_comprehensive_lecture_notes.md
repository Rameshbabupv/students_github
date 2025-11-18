# Week 15: R-CNN Family - Two-Stage Object Detection
# Course: 21CSE558T - Deep Neural Network Architectures
# Module 5: Object Detection - Part 3 (Final Week)

**Course Information:**
- Course Code: 21CSE558T
- Module: 5 (Object Detection)
- Week: 15 (Final Week)
- Credits: 3 (2L + 1T + 0P)
- Institution: SRM University, M.Tech Program

**Instructor Notes:**
- This is the FINAL week of the course (Nov 18-21, 2025)
- Focus: Two-stage detectors vs single-shot detectors
- Tutorial T15: Hands-on Faster R-CNN implementation
- Final exam preparation: Module 5 complete review
- Emphasis on UNDERSTANDING architecture over training

**Prerequisites:**
- Week 13: Object detection fundamentals (IOU, mAP, bounding boxes)
- Week 14: YOLO & SSD architecture and training
- Understanding of CNNs and transfer learning (Weeks 10-12)
- Basic PyTorch/TensorFlow knowledge

**Learning Outcomes (Week 15):**
By the end of this week, students should be able to:
1. Explain the evolution of R-CNN family (R-CNN → Fast → Faster)
2. Understand the two-stage detection paradigm
3. Explain Region Proposal Networks (RPN) and their role
4. Compare two-stage vs single-shot detectors
5. Use pre-trained Faster R-CNN models for object detection
6. Decide when to use R-CNN vs YOLO based on application requirements

---

## Table of Contents

1. [Introduction: The R-CNN Revolution](#1-introduction)
2. [R-CNN (2014): The Original Two-Stage Detector](#2-rcnn-original)
3. [Fast R-CNN (2015): Speeding Up Detection](#3-fast-rcnn)
4. [Faster R-CNN (2015): The RPN Breakthrough](#4-faster-rcnn)
5. [Region Proposal Networks (RPN) Deep Dive](#5-rpn-deep-dive)
6. [ROI Pooling vs ROI Align](#6-roi-pooling-align)
7. [Two-Stage vs Single-Shot Detectors](#7-two-stage-vs-single-shot)
8. [Practical Implementation with Faster R-CNN](#8-practical-implementation)
9. [When to Use Which Detector](#9-when-to-use)
10. [Summary and Practice Questions](#10-summary)

---

## 1. Introduction: The R-CNN Revolution {#1-introduction}

### 1.1 What Makes R-CNN Different?

**Two-Stage Detection Paradigm:**
1. **Stage 1**: Propose regions that might contain objects (Region Proposals)
2. **Stage 2**: Classify each proposed region and refine bounding boxes

**Contrast with Single-Shot Detectors (YOLO/SSD):**
- YOLO/SSD: One forward pass through the network
- R-CNN family: Two forward passes (proposals + classification)

**Key Insight:**
```
Single-Shot:  Image → CNN → Predictions (one step)
Two-Stage:    Image → Region Proposals → CNN → Predictions (two steps)
```

### 1.2 The R-CNN Family Timeline

**R-CNN (2014)** - Ross Girshick et al.
- First successful deep learning approach to object detection
- Achieved 53.7% mAP on PASCAL VOC 2010 (huge improvement)
- Problem: Extremely slow (47 seconds per image)

**Fast R-CNN (2015)** - Ross Girshick
- 9× faster training, 213× faster testing than R-CNN
- Reduced to 2.3 seconds per image
- Problem: Still dependent on slow selective search

**Faster R-CNN (2015)** - Ren, He, Girshick, Sun
- Introduced Region Proposal Network (RPN)
- 10× faster than Fast R-CNN (0.2 seconds per image)
- Modern standard for two-stage detection

**Timeline:**
```
2014: R-CNN (47 sec/image) → 53.7% mAP
2015: Fast R-CNN (2.3 sec/image) → 66.9% mAP
2015: Faster R-CNN (0.2 sec/image) → 73.2% mAP
2017: Mask R-CNN (instance segmentation)
2020+: Modern variants (Cascade R-CNN, etc.)
```

### 1.3 Why Study R-CNN in 2025?

**Reasons:**
1. **Accuracy**: Still among the most accurate detectors
2. **Foundational**: Understanding R-CNN helps understand modern detectors
3. **Industry Use**: Used in applications where accuracy > speed
4. **Academic Importance**: Highly cited papers (R-CNN: 24,000+ citations)
5. **Pre-trained Models**: Easy to use torchvision implementations

**Real-World Applications:**
- Medical imaging (accuracy critical)
- Autonomous vehicles (pedestrian detection)
- Satellite image analysis
- Quality control in manufacturing

---

## 2. R-CNN (2014): The Original Two-Stage Detector {#2-rcnn-original}

### 2.1 R-CNN Architecture Overview

**Four-Step Process:**

**Step 1: Region Proposals (Selective Search)**
- Generate ~2,000 region proposals per image
- Use selective search algorithm (classical computer vision)
- No learning involved in this step

**Step 2: Feature Extraction (CNN)**
- Warp each region to 227×227 (AlexNet input size)
- Forward pass through CNN (AlexNet or VGG)
- Extract 4096-dimensional feature vector

**Step 3: Classification (SVM)**
- Train one SVM per class (e.g., 20 SVMs for PASCAL VOC)
- Input: 4096-dimensional CNN features
- Output: Class scores for each region

**Step 4: Bounding Box Regression**
- Refine bounding box coordinates
- Linear regression model per class
- Improves localization accuracy

**Architecture Diagram:**
```
Input Image (any size)
    ↓
Selective Search → ~2,000 Region Proposals
    ↓
Warp each region to 227×227
    ↓
CNN (AlexNet/VGG) → 4096-dim features
    ↓
SVM Classifier → Class scores
    ↓
Bounding Box Regressor → Refined boxes
    ↓
Non-Maximum Suppression (NMS)
    ↓
Final Detections
```

### 2.2 R-CNN Training Process

**Three-Stage Training:**

**Stage 1: Pre-train CNN**
- Pre-train on ImageNet (1000 classes)
- Standard image classification task
- Transfer learning foundation

**Stage 2: Fine-tune CNN**
- Fine-tune on detection dataset (e.g., PASCAL VOC)
- Positive samples: IOU ≥ 0.5 with ground truth
- Negative samples: IOU < 0.5

**Stage 3: Train SVMs and Regressors**
- Train one-vs-all SVMs for each class
- Train bounding box regressors for each class
- Use CNN features as input

### 2.3 R-CNN Limitations

**Critical Problems:**

**1. Speed:**
- 47 seconds per image (on GPU!)
- 2,000 forward passes through CNN per image
- Not suitable for real-time applications

**2. Disk Space:**
- Must save CNN features for all proposals
- ~200 GB of features for PASCAL VOC training

**3. Training Complexity:**
- Three separate training stages
- Must train CNN, SVMs, and regressors separately
- No end-to-end training

**4. Memory:**
- Must load all features into memory for SVM training
- Requires large RAM

**Why It Matters Historically:**
Despite these limitations, R-CNN proved that deep learning could work for object detection, achieving dramatic improvements over classical methods.

---

## 3. Fast R-CNN (2015): Speeding Up Detection {#3-fast-rcnn}

### 3.1 Key Innovation: Shared Computation

**Main Idea:**
Instead of running CNN on each proposal separately, run it ONCE on the entire image, then extract features for each proposal from the shared feature map.

**Architectural Change:**
```
R-CNN:           For each proposal → Warp → CNN → Features
Fast R-CNN:      Image → CNN → Feature Map → ROI Pooling → Features
```

**Speed Improvement:**
- Training: 9× faster than R-CNN
- Testing: 213× faster than R-CNN
- Reduced from 47 sec/image to 2.3 sec/image

### 3.2 Fast R-CNN Architecture

**End-to-End Pipeline:**

**Step 1: Feature Extraction**
- Input entire image to CNN (VGG16)
- Get feature map (e.g., 512×14×14 for VGG16)

**Step 2: ROI Projection**
- Project region proposals onto feature map
- Use ROI pooling to get fixed-size features

**Step 3: ROI Pooling**
- Convert variable-size ROI to fixed size (e.g., 7×7)
- Max pooling within each grid cell

**Step 4: Fully Connected Layers**
- FC layers process ROI features
- Two sibling output layers:
  - Softmax classifier (classes)
  - Bounding box regressor (4 coordinates)

**Architecture Diagram:**
```
Input Image
    ↓
CNN (VGG16) → Feature Map (512×H'×W')
    ↓
Region Proposals (Selective Search) → Project to feature map
    ↓
ROI Pooling → Fixed-size features (7×7×512)
    ↓
Fully Connected Layers (4096 → 4096)
    ↓
         ↙               ↘
Softmax (Classes)    BBox Regressor (4×C)
```

### 3.3 ROI Pooling Explained

**Problem:**
Region proposals have different sizes, but FC layers need fixed-size input.

**Solution: ROI Pooling**

**Algorithm:**
1. Divide ROI into H×W grid (e.g., 7×7)
2. Max pool within each grid cell
3. Result: Fixed H×W×C output

**Example:**
```
Input ROI: 9×14 feature map
Target: 3×3 output

Divide into 3×3 grid:
- Cell (0,0): size 3×5 → max pool → one value
- Cell (0,1): size 3×5 → max pool → one value
- ...
Result: 3×3 feature map
```

**Mathematical Formulation:**
For ROI with top-left (x₀, y₀) and bottom-right (x₁, y₁):
- Cell (i, j) covers region: [x₀ + i·w/H, x₀ + (i+1)·w/H] × [y₀ + j·h/W, y₀ + (j+1)·h/W]
- Output(i, j) = max pooling over that region

### 3.4 Fast R-CNN Training

**Multi-Task Loss:**
```
L = L_cls + λ·L_bbox

L_cls: Softmax cross-entropy for classification
L_bbox: Smooth L1 loss for bounding box regression
λ: Balance parameter (usually 1)
```

**Smooth L1 Loss:**
```
smooth_L1(x) = 0.5·x²        if |x| < 1
               |x| - 0.5     otherwise

More robust to outliers than L2 loss
```

**Training Details:**
- Batch size: 2 images, 128 ROIs (64 per image)
- Positive ROIs: IOU ≥ 0.5 with ground truth
- Negative ROIs: 0.1 ≤ IOU < 0.5
- Ratio: 25% positive, 75% negative

### 3.5 Fast R-CNN Remaining Limitation

**Bottleneck: Region Proposals**
- Still using selective search (classical algorithm)
- Takes ~2 seconds per image
- Not learned end-to-end
- Limits further speed improvements

**Next Step:**
Make region proposals learnable → Faster R-CNN with RPN

---

## 4. Faster R-CNN (2015): The RPN Breakthrough {#4-faster-rcnn}

### 4.1 The RPN Revolution

**Key Innovation:**
Replace selective search with a learned Region Proposal Network (RPN)

**Impact:**
- 10× faster than Fast R-CNN
- 0.2 seconds per image (5 FPS)
- Fully end-to-end trainable
- Better proposals than selective search

**Architecture:**
```
Faster R-CNN = RPN + Fast R-CNN
```

### 4.2 Faster R-CNN Complete Architecture

**Full Pipeline:**

**Step 1: Feature Extraction**
- Input image to CNN (VGG16, ResNet50)
- Get convolutional feature map

**Step 2: Region Proposal Network (RPN)**
- Propose regions directly from feature map
- Output: ~300 region proposals with objectness scores

**Step 3: ROI Pooling**
- Pool features for each proposal
- Fixed-size output (7×7)

**Step 4: Detection Head**
- Classification: Which class?
- Regression: Refine bounding box

**Step 5: Post-Processing**
- Non-Maximum Suppression (NMS)
- Final detections

**Diagram:**
```
Input Image
    ↓
Shared CNN (VGG16/ResNet50)
    ↓
Feature Map (H×W×C)
    ↓         ↓
    RPN       ↓
    ↓         ↓
Proposals    ↓
    ↓         ↓
ROI Pooling ←┘
    ↓
Detection Head
    ↓
NMS → Final Detections
```

### 4.3 Two-Stage Detection Paradigm

**Stage 1: Region Proposal Network**
- Input: Feature map
- Output: Region proposals + objectness scores
- Task: "Is there an object here?" (binary classification)

**Stage 2: Detection Network**
- Input: Proposed regions
- Output: Class labels + refined boxes
- Task: "What is this object?" (multi-class classification)

**Why Two Stages?**
1. **Divide and Conquer**: Separate localization from classification
2. **Efficiency**: Focus computation on likely object regions
3. **Accuracy**: Specialized networks for each task

---

## 5. Region Proposal Networks (RPN) Deep Dive {#5-rpn-deep-dive}

### 5.1 RPN Architecture

**Core Idea:**
Slide a small network over the feature map to predict region proposals

**Architecture:**

**Step 1: Sliding Window**
- Slide 3×3 window over feature map
- At each position, use 512-dim feature vector

**Step 2: Anchor Boxes**
- At each position, generate k anchor boxes (k=9 typical)
- 3 scales × 3 aspect ratios = 9 anchors
- Scales: 128², 256², 512² pixels
- Aspect ratios: 1:1, 1:2, 2:1

**Step 3: Two Sibling Outputs**
- Objectness: 2k scores (object vs background)
- Box Regression: 4k coordinates (dx, dy, dw, dh)

**Diagram:**
```
Feature Map (H×W×512)
    ↓
Sliding 3×3 Conv → 512-dim
    ↓
    ├─→ 1×1 Conv → 2k objectness scores
    └─→ 1×1 Conv → 4k box coordinates

k = number of anchors (typically 9)
```

### 5.2 Anchor Boxes Explained

**Anchor Boxes at Each Position:**

**Example Setup (k=9):**
```
Scales:        128, 256, 512 (pixels)
Aspect Ratios: 1:1, 1:2, 2:1

Anchors at position (x, y):
1. 128×128 (1:1)
2. 128×256 (1:2)
3. 256×128 (2:1)
4. 256×256 (1:1)
5. 256×512 (1:2)
6. 512×256 (2:1)
7. 512×512 (1:1)
8. 512×1024 (1:2)
9. 1024×512 (2:1)
```

**Total Anchors:**
For 800×600 image with VGG16:
- Feature map: ~50×38
- Anchors per position: 9
- Total anchors: 50×38×9 ≈ 17,000 anchors

**Purpose:**
- Cover objects of different sizes and shapes
- Pre-defined templates (no need to learn shapes)
- RPN only predicts offsets from anchors

### 5.3 RPN Training

**Objectness Loss (Binary Classification):**
```
L_cls = Cross-entropy for object vs background

Positive anchors: IOU ≥ 0.7 with any ground truth
Negative anchors: IOU < 0.3 with all ground truth
Ignored: 0.3 ≤ IOU < 0.7
```

**Box Regression Loss:**
```
L_reg = Smooth L1 loss on (dx, dy, dw, dh)

Only computed for positive anchors
```

**Parameterization:**
Predict offsets from anchor to ground truth:
```
dx = (x_gt - x_anchor) / w_anchor
dy = (y_gt - y_anchor) / h_anchor
dw = log(w_gt / w_anchor)
dh = log(h_gt / h_anchor)

Log scale for width/height → better numerical stability
```

**Total RPN Loss:**
```
L_RPN = L_cls + λ·L_reg

λ = 10 (balance parameter)
```

### 5.4 RPN Inference

**Proposal Generation:**

**Step 1: Generate Anchors**
- ~17,000 anchors across feature map

**Step 2: Apply Predicted Offsets**
- Transform anchors using predicted (dx, dy, dw, dh)

**Step 3: Clip to Image**
- Ensure proposals stay within image boundaries

**Step 4: Filter by Objectness**
- Keep top 6,000 proposals by objectness score (training)
- Keep top 300 proposals (testing)

**Step 5: Non-Maximum Suppression**
- NMS with IOU threshold 0.7
- Remove redundant proposals

**Step 6: Output**
- Final ~300 proposals for detection head

---

## 6. ROI Pooling vs ROI Align {#6-roi-pooling-align}

### 6.1 ROI Pooling (Fast R-CNN)

**Problem ROI Pooling Solves:**
Region proposals have arbitrary sizes, but FC layers need fixed-size input.

**How It Works:**
1. Divide ROI into H×W grid (e.g., 7×7)
2. Max pool within each grid cell
3. Output: Fixed H×W feature map

**Issue: Quantization**

**Example:**
```
Input image: 800×800
Feature map: 25×25 (stride 32)
ROI: [13, 9, 247, 231] (in image coordinates)

Projected to feature map: [13/32, 9/32, 247/32, 231/32]
                        = [0.40, 0.28, 7.71, 7.22]

Quantized: [0, 0, 7, 7]  ← Lost precision!

Divide into 7×7 grid:
Each cell size: 7/7 = 1.0
Again quantized to integer coordinates
```

**Impact:**
- Small misalignments in feature extraction
- Degrades accuracy, especially for small objects
- More problematic for instance segmentation (pixel-level)

### 6.2 ROI Align (Mask R-CNN, 2017)

**Solution: Bilinear Interpolation**

**Key Changes:**
1. **No quantization**: Keep floating-point coordinates
2. **Bilinear interpolation**: Sample at exact locations
3. **Average pooling**: Instead of max pooling (optional)

**Algorithm:**
```
For each grid cell:
1. Sample at 4 regular locations (can be more)
2. Use bilinear interpolation for each sample
3. Average or max pool the samples
```

**Example:**
```
ROI: [0.40, 0.28, 7.71, 7.22] (floating-point)
Cell (i, j): Sample at exact floating-point positions
Use bilinear interpolation from 4 neighboring pixels
```

**Bilinear Interpolation:**
```
Given point (x, y) between integer coordinates:
- Find 4 neighbors: (⌊x⌋, ⌊y⌋), (⌊x⌋+1, ⌊y⌋), etc.
- Interpolate: value = Σ weight_i × value_i
```

**Impact:**
- Improved accuracy (especially for segmentation)
- Slightly more computation (negligible)
- Now standard in modern detectors

### 6.3 Comparison

| Aspect | ROI Pooling | ROI Align |
|--------|-------------|-----------|
| Coordinates | Quantized (integer) | Floating-point |
| Sampling | Grid corners only | Multiple samples per cell |
| Interpolation | None | Bilinear |
| Accuracy | Good for detection | Better for segmentation |
| Speed | Slightly faster | Slightly slower |
| Usage | Fast R-CNN | Mask R-CNN onwards |

---

## 7. Two-Stage vs Single-Shot Detectors {#7-two-stage-vs-single-shot}

### 7.1 Fundamental Difference

**Two-Stage (Faster R-CNN):**
```
Stage 1: RPN → Where are objects?
Stage 2: Detection Head → What are they?

Two forward passes through network
```

**Single-Shot (YOLO, SSD):**
```
One Stage: CNN → Predictions

One forward pass through network
```

### 7.2 Speed Comparison

**Inference Time (on same GPU):**

| Model | FPS | Inference Time |
|-------|-----|----------------|
| Faster R-CNN (ResNet-50) | 5-7 FPS | ~150-200 ms |
| Faster R-CNN (ResNet-101) | 3-5 FPS | ~200-300 ms |
| YOLOv3 | 30-40 FPS | ~25-30 ms |
| YOLOv5 | 60-100 FPS | ~10-15 ms |
| YOLOv8 | 80-120 FPS | ~8-12 ms |
| SSD-300 | 40-60 FPS | ~15-25 ms |

**Why Faster R-CNN is Slower:**
1. RPN network (first stage)
2. ROI pooling for each proposal
3. Detection head for each ROI
4. More parameters overall

### 7.3 Accuracy Comparison

**COCO Test-Dev mAP:**

| Model | mAP | mAP@50 | mAP@75 |
|-------|-----|--------|--------|
| Faster R-CNN (ResNet-50-FPN) | 37.4% | 58.1% | 40.3% |
| Faster R-CNN (ResNet-101-FPN) | 39.8% | 61.3% | 43.4% |
| YOLOv3 | 33.0% | 57.9% | 34.4% |
| YOLOv5l | 49.0% | 67.3% | 53.4% |
| YOLOv8x | 53.9% | 71.4% | 58.8% |
| SSD-512 | 31.2% | 50.4% | 33.3% |

**Observations:**
- Modern YOLOv8 surpasses Faster R-CNN
- Faster R-CNN better at high IOU thresholds (mAP@75)
- YOLO evolved significantly (v3 → v8)
- Trade-off depends on model size and hardware

### 7.4 Detailed Comparison Table

| Aspect | Faster R-CNN | YOLO/SSD |
|--------|--------------|----------|
| **Paradigm** | Two-stage | Single-shot |
| **Speed** | 5-7 FPS | 30-120 FPS |
| **Accuracy** | High (especially localization) | Good (improving) |
| **Small Objects** | Better | Weaker |
| **Training Time** | Longer | Shorter |
| **Training Complexity** | Higher | Lower |
| **Real-Time** | Challenging | Yes |
| **Hardware** | GPU preferred | CPU possible (YOLO) |
| **Use Case** | Accuracy critical | Speed critical |

### 7.5 When to Use Each

**Use Faster R-CNN When:**
1. **Accuracy is critical**: Medical imaging, autonomous vehicles
2. **Small object detection**: Satellite imagery, crowd counting
3. **Offline processing**: Batch processing acceptable
4. **High-quality annotations**: Need precise localization
5. **Research**: Academic benchmarking

**Use YOLO/SSD When:**
1. **Real-time required**: Video surveillance, robotics
2. **Resource-constrained**: Edge devices, mobile
3. **Large objects**: Typical scenes, common objects
4. **Quick deployment**: Pre-trained models work well
5. **Production systems**: Speed and reliability

**Hybrid Approach:**
- Use YOLO for initial detection (fast)
- Use Faster R-CNN for refinement (accurate)
- Example: Video surveillance (YOLO) + forensic analysis (Faster R-CNN)

---

## 8. Practical Implementation with Faster R-CNN {#8-practical-implementation}

### 8.1 Using Torchvision's Pre-Trained Faster R-CNN

**Installation:**
```python
# Required libraries
pip install torch torchvision opencv-python matplotlib pillow
```

**Basic Usage:**
```python
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
import matplotlib.pyplot as plt

# Load pre-trained model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Load image
image = Image.open('test.jpg')
image_tensor = torchvision.transforms.ToTensor()(image)

# Inference
with torch.no_grad():
    predictions = model([image_tensor])

# predictions[0] contains:
# - boxes: [N, 4] (x1, y1, x2, y2)
# - labels: [N] (class IDs)
# - scores: [N] (confidence)
```

### 8.2 Model Variants in Torchvision

**Available Models:**

**1. Faster R-CNN with ResNet-50 + FPN**
```python
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# Size: ~160 MB
# Accuracy: 37.0% mAP on COCO
# Speed: ~150 ms/image (GPU)
```

**2. Faster R-CNN with MobileNet V3**
```python
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
# Size: ~19 MB
# Accuracy: 32.8% mAP on COCO
# Speed: ~50-70 ms/image (GPU), faster on CPU
```

**3. Mask R-CNN (Instance Segmentation)**
```python
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
# Includes segmentation masks
# Size: ~170 MB
```

### 8.3 Inference Pipeline

**Complete Example:**
```python
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# COCO class names (80 classes)
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    # ... (total 91, indices 1-90, some unused)
]

def detect_objects(image_path, threshold=0.5):
    # Load model
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = torchvision.transforms.ToTensor()(image)

    # Inference
    with torch.no_grad():
        predictions = model([image_tensor])

    # Filter by threshold
    boxes = predictions[0]['boxes'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()

    keep = scores >= threshold
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    return boxes, labels, scores

def visualize_detections(image_path, boxes, labels, scores):
    # Load image
    image = Image.open(image_path)
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)

    # Draw boxes
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1

        # Draw rectangle
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2,
                                 edgecolor='red', facecolor='none')
        ax.add_patch(rect)

        # Add label
        class_name = COCO_CLASSES[label]
        ax.text(x1, y1 - 5, f'{class_name}: {score:.2f}',
                bbox=dict(facecolor='yellow', alpha=0.5),
                fontsize=10, color='black')

    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Usage
image_path = 'test_image.jpg'
boxes, labels, scores = detect_objects(image_path, threshold=0.7)
visualize_detections(image_path, boxes, labels, scores)
```

### 8.4 Fine-Tuning for Custom Dataset

**Training Faster R-CNN on Custom Data:**

**Step 1: Prepare Dataset**
```python
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, image_dir, annotations):
        self.image_dir = image_dir
        self.annotations = annotations  # List of dicts

    def __getitem__(self, idx):
        # Load image
        img_path = self.annotations[idx]['image_path']
        img = Image.open(img_path).convert('RGB')
        img_tensor = torchvision.transforms.ToTensor()(img)

        # Load annotations
        boxes = self.annotations[idx]['boxes']  # [N, 4] (x1,y1,x2,y2)
        labels = self.annotations[idx]['labels']  # [N]

        target = {
            'boxes': torch.FloatTensor(boxes),
            'labels': torch.LongTensor(labels)
        }

        return img_tensor, target

    def __len__(self):
        return len(self.annotations)
```

**Step 2: Modify Model for Custom Classes**
```python
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Load pre-trained model
model = fasterrcnn_resnet50_fpn(pretrained=True)

# Replace classification head
num_classes = 5  # 4 custom classes + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
```

**Step 3: Training Loop**
```python
import torch.optim as optim

# Prepare data
dataset = CustomDataset(image_dir, annotations)
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x))
)

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(data_loader):.4f}')

# Save model
torch.save(model.state_dict(), 'custom_faster_rcnn.pth')
```

### 8.5 GPU vs CPU Performance

**Inference Time Comparison:**

| Model | GPU (RTX 3060) | CPU (i7) | Speedup |
|-------|----------------|----------|---------|
| Faster R-CNN ResNet-50 | 150 ms | 2,500 ms | 16.7× |
| Faster R-CNN MobileNet | 50 ms | 800 ms | 16× |
| YOLOv8n | 10 ms | 150 ms | 15× |

**Recommendations:**
- **GPU**: Essential for training, recommended for inference
- **CPU**: Possible for inference (especially MobileNet variant)
- **Edge Devices**: Use MobileNet variant or YOLO
- **Colab**: Free T4 GPU sufficient for all notebooks

---

## 9. When to Use Which Detector {#9-when-to-use}

### 9.1 Decision Framework

**Question 1: Real-Time Required?**
```
YES → YOLO/SSD (30+ FPS possible)
NO  → Consider accuracy requirements
```

**Question 2: Accuracy Critical?**
```
YES → Faster R-CNN (especially for small objects)
NO  → YOLO/SSD sufficient
```

**Question 3: Hardware Constraints?**
```
Edge Device/Mobile → YOLOv8n or MobileNet-based Faster R-CNN
CPU Only → YOLO preferred (faster on CPU)
GPU Available → Any detector
```

**Question 4: Object Size?**
```
Small Objects (< 32×32 px) → Faster R-CNN
Large Objects → YOLO/SSD
Mixed → Depends on priority
```

**Question 5: Training Data?**
```
Small Dataset → Pre-trained YOLO (better generalization)
Large Dataset → Fine-tune either
Custom Classes → Fine-tune either
```

### 9.2 Application-Specific Recommendations

**Autonomous Vehicles:**
- **Primary**: YOLO for real-time detection
- **Secondary**: Faster R-CNN for pedestrian detection (accuracy critical)
- **Hybrid**: YOLO for general objects, Faster R-CNN for safety-critical

**Medical Imaging:**
- **Primary**: Faster R-CNN (accuracy critical, offline processing)
- **Why**: Small lesions, high precision required
- **Trade-off**: Speed not critical

**Surveillance:**
- **Primary**: YOLO (real-time, multiple cameras)
- **Secondary**: Faster R-CNN for forensic analysis
- **Why**: Real-time alerting vs offline investigation

**Satellite Imagery:**
- **Primary**: Faster R-CNN (small objects, high resolution)
- **Why**: Objects often small (vehicles, buildings)
- **Processing**: Batch processing acceptable

**Retail/Inventory:**
- **Primary**: YOLO (speed, simplicity)
- **Why**: Objects typically large enough, real-time preferred

**Robotics:**
- **Primary**: YOLO (real-time navigation)
- **Consideration**: Hardware constraints favor YOLO
- **Exception**: Precise manipulation tasks → Faster R-CNN

### 9.3 Hybrid Systems

**Best of Both Worlds:**

**Example 1: Video Surveillance**
```
YOLO (real-time) → Alert generation
    ↓
Faster R-CNN (offline) → Forensic analysis, evidence
```

**Example 2: Manufacturing QC**
```
YOLO (fast screening) → Initial defect detection
    ↓
Faster R-CNN (detailed) → Defect classification
```

**Example 3: Autonomous Driving**
```
YOLO (general detection) → Vehicles, signs, lights
Faster R-CNN (critical) → Pedestrians, cyclists
```

---

## 10. Summary and Practice Questions {#10-summary}

### 10.1 Key Takeaways

**R-CNN Evolution:**
1. **R-CNN (2014)**: First deep learning detector, slow (47 sec/image)
2. **Fast R-CNN (2015)**: Shared computation, 9× faster training
3. **Faster R-CNN (2015)**: RPN for learned proposals, 10× faster

**Two-Stage Paradigm:**
- **Stage 1**: Region Proposal Network (where are objects?)
- **Stage 2**: Detection head (what are they?)
- **Advantage**: Higher accuracy, especially for small objects
- **Disadvantage**: Slower than single-shot detectors

**RPN (Region Proposal Network):**
- Learned region proposals (replaces selective search)
- Anchor boxes: 3 scales × 3 aspect ratios = 9 anchors
- Outputs: Objectness scores + box offsets
- Training: Multi-task loss (classification + regression)

**ROI Pooling/Align:**
- **ROI Pooling**: Fixed-size features from variable-size ROIs
- **ROI Align**: Bilinear interpolation, no quantization (better)

**Faster R-CNN vs YOLO:**
- **Speed**: YOLO faster (30-120 FPS vs 5-7 FPS)
- **Accuracy**: Comparable (depends on model size)
- **Use Case**: YOLO for real-time, Faster R-CNN for accuracy

**Practical Use:**
- Pre-trained models in torchvision
- Easy to use for inference
- Fine-tunable for custom datasets

### 10.2 Practice Questions

**Conceptual Understanding:**

1. **Explain the difference between R-CNN, Fast R-CNN, and Faster R-CNN.**
   - Focus: Evolution, speed improvements, architectural changes

2. **What is the two-stage detection paradigm? How does it differ from single-shot detection?**
   - Focus: RPN + detection head vs direct prediction

3. **Describe how the Region Proposal Network (RPN) works.**
   - Focus: Anchor boxes, objectness, box regression

4. **What are anchor boxes? Why are they used in Faster R-CNN?**
   - Focus: Pre-defined templates, multi-scale/aspect ratio coverage

5. **Compare ROI Pooling and ROI Align. Which is better and why?**
   - Focus: Quantization issues, bilinear interpolation

6. **When would you use Faster R-CNN over YOLO, and vice versa?**
   - Focus: Speed vs accuracy trade-off, application requirements

**Technical Questions:**

7. **Calculate the number of anchors generated by RPN for an 800×600 input image with VGG16 backbone (stride 16), using 9 anchors per position.**
   - Solution: Feature map = 800/16 × 600/16 = 50×38
   - Anchors = 50×38×9 = 17,100

8. **Given an anchor box at position (100, 100) with size 256×256, and predicted offsets (dx=0.1, dy=-0.1, dw=0.2, dh=0.1), calculate the final predicted box.**
   - Solution:
     - x = 100 + 0.1×256 = 125.6
     - y = 100 - 0.1×256 = 74.4
     - w = 256 × exp(0.2) ≈ 312
     - h = 256 × exp(0.1) ≈ 283
     - Box: [125.6, 74.4, 437.6, 357.4] (x1, y1, x2, y2)

9. **Why does Faster R-CNN use log scale for width/height regression (dw, dh)?**
   - Solution: Numerical stability, scale invariance, unbounded range

10. **Compare the training complexity of Faster R-CNN vs YOLOv8.**
    - Focus: Multi-stage training (RPN + detector) vs single-stage

**Implementation Questions:**

11. **Write PyTorch code to load a pre-trained Faster R-CNN model and perform inference on an image.**
    - Use torchvision.models.detection.fasterrcnn_resnet50_fpn

12. **How would you modify Faster R-CNN for a custom dataset with 5 object classes?**
    - Solution: Replace FastRCNNPredictor with num_classes=6 (5 + background)

13. **Explain the data format required for training Faster R-CNN.**
    - Solution: Images + targets (boxes, labels)

14. **What are the key hyperparameters for training Faster R-CNN?**
    - Solution: Learning rate, batch size, IOU thresholds, NMS threshold

15. **How can you improve Faster R-CNN inference speed?**
    - Solution: Use MobileNet backbone, reduce proposals, TensorRT optimization

### 10.3 Module 5 Complete Review

**Week 13: Object Detection Foundations**
- IOU, mAP, bounding box formats
- Evaluation metrics (precision, recall, AP)
- Classical methods (sliding windows, selective search)

**Week 14: YOLO & SSD**
- Single-shot detection paradigm
- YOLO architecture and evolution
- Real-time detection (30+ FPS)
- Training custom YOLO models

**Week 15: R-CNN Family**
- Two-stage detection paradigm
- RPN (learned region proposals)
- Faster R-CNN architecture
- When to use which detector

**Comprehensive Understanding:**
Students should now be able to:
1. Explain object detection from fundamentals to state-of-the-art
2. Implement YOLO and Faster R-CNN for custom tasks
3. Evaluate detectors using mAP and other metrics
4. Choose appropriate detector for specific applications
5. Fine-tune pre-trained models for custom datasets

### 10.4 Final Exam Preparation Tips

**Key Topics to Master:**
1. Object detection problem definition
2. Evaluation metrics (IOU, precision, recall, mAP)
3. YOLO architecture and evolution
4. R-CNN family evolution and improvements
5. Two-stage vs single-shot comparison
6. Practical implementation skills

**Expected Question Types:**
- **Conceptual**: Explain RPN, two-stage paradigm, anchor boxes
- **Comparative**: YOLO vs Faster R-CNN trade-offs
- **Computational**: Calculate IOU, mAP, anchors
- **Practical**: Code snippets for inference, training
- **Application**: Choose detector for given scenario

**Study Strategy:**
1. Review all 15 notebooks (Weeks 13-15)
2. Practice coding exercises (T13, T14, T15)
3. Compare YOLO vs Faster R-CNN on same images
4. Understand trade-offs (speed vs accuracy)
5. Memorize key numbers (FPS, mAP benchmarks)

---

## Additional Resources

### Research Papers (Chronological Order)

1. **R-CNN (2014)**
   - Title: "Rich feature hierarchies for accurate object detection and semantic segmentation"
   - Authors: Girshick et al.
   - Venue: CVPR 2014
   - Citations: 24,000+

2. **Fast R-CNN (2015)**
   - Title: "Fast R-CNN"
   - Authors: Ross Girshick
   - Venue: ICCV 2015
   - Citations: 17,000+

3. **Faster R-CNN (2015)**
   - Title: "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"
   - Authors: Ren, He, Girshick, Sun
   - Venue: NeurIPS 2015
   - Citations: 33,000+

4. **Mask R-CNN (2017)**
   - Title: "Mask R-CNN"
   - Authors: He, Gkioxari, Dollar, Girshick
   - Venue: ICCV 2017
   - Citations: 18,000+

### Official Implementations

1. **Torchvision (PyTorch)**
   - URL: https://pytorch.org/vision/stable/models/detection.html
   - Models: Faster R-CNN, Mask R-CNN, RetinaNet
   - Recommended for Tutorial T15

2. **Detectron2 (Facebook Research)**
   - URL: https://github.com/facebookresearch/detectron2
   - Advanced implementations, highly modular
   - Optional for advanced students

3. **TensorFlow Object Detection API**
   - URL: https://github.com/tensorflow/models/tree/master/research/object_detection
   - TensorFlow-based implementations

### Datasets

1. **COCO (Common Objects in Context)**
   - 80 object classes
   - 330,000 images
   - Standard benchmark for object detection

2. **PASCAL VOC**
   - 20 object classes
   - Older but still used for teaching

3. **Open Images**
   - 600 object classes
   - Larger and more diverse than COCO

### Video Tutorials

1. **CS231n (Stanford)**
   - Lecture 11: Detection and Segmentation
   - URL: http://cs231n.stanford.edu/

2. **Fast.ai**
   - Practical Deep Learning for Coders
   - Includes object detection lesson

---

## End of Week 15 Lecture Notes

**Course Completion:**
Congratulations! You have completed Module 5 (Object Detection) and the entire Deep Neural Network Architectures course.

**Final Exam:**
- Covers all 5 modules
- Emphasis on Modules 4-5 (CNNs, Transfer Learning, Object Detection)
- Practice questions provided in each week's materials

**Next Steps:**
1. Complete Tutorial T15 (Faster R-CNN hands-on)
2. Review all 3 weeks of Module 5
3. Compare YOLO vs Faster R-CNN on custom images
4. Prepare for final exam using practice questions

**Feedback:**
For questions or clarifications, contact instructor during office hours or class sessions.

---

**Document Information:**
- Created: November 2025
- Version: 1.0
- Last Updated: November 18, 2025
- Status: Complete
- Pages: 20
- Lines: ~1,700

**Tutorial T15 Reference:**
See `notebooks/03_faster_rcnn_pretrained.ipynb` for hands-on Faster R-CNN implementation.

**Assessment:**
Final examination will include questions from Module 5 (Weeks 13-15) covering object detection foundations, YOLO/SSD, and R-CNN family.

---

**End of Comprehensive Lecture Notes - Week 15**
