# Week 13: Object Detection Foundations

**Course:** 21CSE558T - Deep Neural Network Architectures
**Module:** 5 - Object Localization and Detection
**Week:** 13 of 15
**Date:** November 2025
**Instructor:** Prof. Ramesh Babu

---

## Overview

Welcome to **Week 13** - the foundation week for understanding modern object detection systems! This week marks a critical transition in our course: we're moving from image classification (telling us WHAT is in an image) to object detection (telling us WHERE things are).

**What Makes This Week Special?**

Unlike previous weeks where we jumped straight into deep learning frameworks, Week 13 takes a deliberate step back to build rock-solid foundations. We focus on the fundamental concepts and mathematical tools that power ALL modern object detectors - whether it's YOLO, R-CNN, or future architectures you'll encounter in industry. These concepts are framework-agnostic and timeless.

**What You'll Build This Week:**

Using only NumPy, Matplotlib, and OpenCV, you'll implement from scratch the core building blocks of object detection: bounding box representations, Intersection over Union (IoU) calculations, Non-Maximum Suppression (NMS), and evaluation metrics like mean Average Precision (mAP). By the end of this week, you'll understand the "magic" behind those colored boxes that appear around detected objects.

**Why This Foundation Matters:**

Week 13 is not optional - it's the essential prerequisite for Weeks 14 and 15. Without mastering IoU, you won't understand how YOLO selects the best bounding boxes. Without understanding mAP, you can't evaluate whether your R-CNN model is actually working. This week transforms you from a deep learning user into someone who truly understands object detection at a fundamental level.

**The Learning Journey:**

You'll progress through 5 interactive notebooks (totaling ~65 minutes) that build on each other systematically. Each notebook introduces one core concept, provides visualizations, and includes hands-on coding exercises. By the time you reach Notebook 05, you'll have implemented a complete (though naive) sliding window detector entirely from scratch - a powerful confidence booster before we tackle modern architectures.

---

## Learning Objectives

After completing Week 13, you will be able to:

1. **Understand Object Detection Fundamentals**
   - Differentiate between classification, localization, and detection tasks
   - Explain why object detection is significantly harder than classification
   - Identify real-world applications where object detection is critical
   - Describe the input/output format of detection systems

2. **Master Bounding Box Mathematics**
   - Represent bounding boxes in multiple coordinate systems (XYWH, XYXY, center-based)
   - Convert between different bounding box formats programmatically
   - Calculate Intersection over Union (IoU) from first principles
   - Visualize and interpret IoU scores for box overlap quality

3. **Implement Core Detection Algorithms**
   - Code Non-Maximum Suppression (NMS) algorithm from scratch
   - Apply NMS to eliminate duplicate detections
   - Understand the role of IoU thresholds in NMS
   - Implement sliding window detection (classical approach)

4. **Evaluate Detection Performance**
   - Calculate precision, recall, and F1-score for detection tasks
   - Compute Average Precision (AP) and mean Average Precision (mAP)
   - Interpret precision-recall curves and AP values
   - Compare detection models using standardized metrics

5. **Prepare for Modern Architectures**
   - Understand why classical approaches (sliding windows) are impractical
   - Recognize the computational bottlenecks that deep learning solves
   - Build intuition for YOLO and R-CNN architectures (coming in Weeks 14-15)
   - Apply learned concepts to any object detection framework

---

## Prerequisites

**Required Knowledge:**

Before starting Week 13, you should be comfortable with:

- **Python Programming:** Functions, loops, conditionals, NumPy arrays
- **Linear Algebra:** Basic vector/matrix operations, understanding of coordinates
- **CNNs (Weeks 10-12):** Image classification, convolutional layers, feature extraction
- **Transfer Learning (Week 12):** Loading pre-trained models, fine-tuning concepts
- **Image Processing:** Understanding of pixel coordinates, image dimensions (H×W×C)

**Optional But Helpful:**

- Familiarity with Matplotlib for visualization
- Basic understanding of precision/recall from binary classification
- Experience with Google Colab notebooks

**Skills Assessment:**

If you can answer these questions confidently, you're ready:
- "What does a CNN output for image classification?" → A probability distribution over classes
- "How are images represented in NumPy?" → (Height, Width, Channels) array
- "What is transfer learning?" → Reusing pre-trained model features for new tasks

**If you need a refresher:**
- Review Week 10-11 notebooks on CNNs
- Revisit Week 12 Transfer Learning concepts
- Check NumPy documentation: https://numpy.org/doc/

---

## Notebook Sequence

This week includes **5 progressive notebooks** (Notebooks 01-05) that build your object detection foundation. Each notebook is self-contained but builds on previous concepts. Total estimated time: **60-70 minutes**.

**Note:** Tutorial T13 (Notebook 06) will be added later and will involve hands-on YOLO implementation.

---

### **01_object_detection_introduction.ipynb**

**Duration:** 10 minutes
**Difficulty:** Beginner
**Focus:** Conceptual understanding

**What You'll Learn:**
- The difference between classification, localization, and detection
- Why object detection is one of the most important CV tasks
- Real-world applications: autonomous vehicles, medical imaging, surveillance
- Challenges: occlusion, scale variation, crowded scenes
- Input/output format for detection systems

**Key Demonstrations:**
- Visual comparison of classification vs localization vs detection
- Synthetic multi-object images with bounding box annotations
- Challenge scenarios: overlapping objects, small objects, different scales

**Hands-On Activities:**
- Create simple bounding box visualizations
- Understand different coordinate systems (XYWH, XYXY)
- Preview the learning path for Weeks 13-15

**Expected Output:**
```
You'll create visualizations showing:
- Single object with bounding box (localization)
- Multiple objects with different labels (detection)
- Challenging detection scenarios (occlusion, crowds)
```

**Why This Matters:**
This notebook sets the stage. You'll understand WHY we need all the tools in the following notebooks and WHERE this fits in the broader deep learning landscape.

**Run This:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)

---

### **02_iou_calculation_hands_on.ipynb**

**Duration:** 12 minutes
**Difficulty:** Beginner-Intermediate
**Focus:** Core metric implementation

**What You'll Learn:**
- What is Intersection over Union (IoU)?
- Why IoU is the most important metric in object detection
- How to calculate IoU from bounding box coordinates
- Interpreting IoU scores: 0.0 (no overlap) to 1.0 (perfect match)
- IoU thresholds for detection quality (e.g., 0.5, 0.75)

**Key Demonstrations:**
- Step-by-step IoU calculation with visualizations
- Examples: perfect overlap (IoU=1.0), partial overlap (IoU=0.5), no overlap (IoU=0.0)
- Interactive widget: drag boxes and see IoU update in real-time
- Comparison of good predictions (IoU>0.7) vs poor predictions (IoU<0.3)

**Hands-On Activities:**
- Implement `calculate_iou(box1, box2)` function from scratch
- Test your implementation on various box pairs
- Visualize intersection and union areas with color coding
- Experiment with threshold values

**Code You'll Write:**
```python
def calculate_iou(box1, box2):
    """
    Calculate IoU between two bounding boxes
    box format: [x, y, width, height]
    """
    # 1. Find intersection coordinates
    # 2. Calculate intersection area
    # 3. Calculate union area
    # 4. Return IoU = intersection / union
```

**Expected Output:**
```
Example IoU calculations:
- Perfect overlap: IoU = 1.00 ✓
- Good detection: IoU = 0.73 ✓
- Poor detection: IoU = 0.28 ✗
- No overlap: IoU = 0.00 ✗
```

**Why This Matters:**
IoU is used EVERYWHERE in object detection: training loss functions, Non-Maximum Suppression, evaluation metrics (mAP), anchor box matching in YOLO. Master this, and you'll understand 50% of detection systems.

**Run This:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)

---

### **03_bounding_box_visualization.ipynb**

**Duration:** 10 minutes
**Difficulty:** Beginner
**Focus:** Practical visualization skills

**What You'll Learn:**
- How to draw bounding boxes on images using Matplotlib and OpenCV
- Different box coordinate formats and when to use each
- Annotating boxes with labels, confidence scores, and colors
- Best practices for visualization (colors, line thickness, transparency)
- Handling edge cases: boxes outside image boundaries

**Key Demonstrations:**
- Load real images and overlay bounding boxes
- Multi-class detection with color-coded boxes
- Add text labels with background for readability
- Compare ground truth (green) vs predictions (red/yellow)

**Hands-On Activities:**
- Implement `draw_bounding_box()` utility function
- Visualize detection results on sample images
- Create a detection output similar to YOLO/R-CNN displays
- Experiment with visualization parameters (colors, thickness, fonts)

**Code You'll Write:**
```python
def draw_bounding_box(image, bbox, label, confidence, color):
    """
    Draw a bounding box on an image with label and score
    """
    # Draw rectangle
    # Add label background
    # Add text (label + confidence)
    return annotated_image
```

**Expected Output:**
```
You'll create professional-looking detection visualizations:
- Green boxes for ground truth
- Red boxes for predictions
- Labels: "cat (0.95)", "dog (0.87)"
- Color-coded by confidence or class
```

**Why This Matters:**
Visualization is crucial for debugging detection systems. You need to SEE what your model is predicting to understand where it's failing. These skills transfer directly to real projects.

**Run This:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)

---

### **04_evaluation_metrics_map.ipynb**

**Duration:** 15 minutes
**Difficulty:** Intermediate
**Focus:** Performance measurement

**What You'll Learn:**
- How to evaluate object detection models (beyond simple accuracy)
- Precision and Recall in the context of detection
- What is Average Precision (AP) and why it matters
- Computing mean Average Precision (mAP) across classes
- Interpreting mAP scores and comparing models

**Key Demonstrations:**
- Calculate True Positives, False Positives, False Negatives using IoU
- Plot Precision-Recall curves for detection tasks
- Compute AP as area under the PR curve
- Calculate mAP for multi-class detection (average AP across classes)
- Compare model A (mAP=0.65) vs model B (mAP=0.78)

**Hands-On Activities:**
- Implement `calculate_ap()` for single class
- Extend to `calculate_map()` for multiple classes
- Analyze detection results and compute metrics
- Understand the impact of IoU threshold on mAP

**Key Concepts:**
```
True Positive (TP): Detection with IoU ≥ threshold (e.g., 0.5)
False Positive (FP): Detection with IoU < threshold OR duplicate
False Negative (FN): Ground truth object not detected

Precision = TP / (TP + FP)  # How many detections are correct?
Recall = TP / (TP + FN)     # How many objects were found?

AP = Area under Precision-Recall curve
mAP = Average of AP across all classes (THE standard metric)
```

**Expected Output:**
```
Class: cat
  AP = 0.82
Class: dog
  AP = 0.76
Class: person
  AP = 0.91

mAP@0.5 = 0.83 (mean across all classes)
```

**Why This Matters:**
mAP is THE metric used to compare object detectors in research papers and competitions. When you hear "YOLO achieves 50% mAP on COCO dataset," you'll know exactly what that means and how it was calculated.

**Run This:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)

---

### **05_naive_sliding_windows.ipynb**

**Duration:** 15 minutes
**Difficulty:** Intermediate
**Focus:** Classical detection approach

**What You'll Learn:**
- How classical object detection worked before deep learning
- The sliding window approach: exhaustive search over image
- Multi-scale detection: handling objects of different sizes
- Why sliding windows are computationally prohibitive
- How YOLO and R-CNN solve the computational problem

**Key Demonstrations:**
- Slide a fixed-size window across an image
- Extract patches and run classifier on each
- Detect objects at multiple scales (small, medium, large)
- Visualize ALL 10,000+ windows evaluated per image
- Compare runtime: sliding windows (30 sec) vs YOLO (0.03 sec)

**Hands-On Activities:**
- Implement `sliding_window_detector()` from scratch
- Run detector on sample images (no deep learning!)
- Apply Non-Maximum Suppression to remove duplicates
- Count total windows evaluated (understand computational cost)

**The Algorithm:**
```python
def sliding_window_detector(image, window_size, stride):
    detections = []
    for scale in [0.5, 1.0, 2.0]:  # Multi-scale
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                # Extract patch at (x, y)
                # Run classifier
                # If object detected, save bounding box
                if confidence > threshold:
                    detections.append([x, y, w, h, conf])

    # Apply NMS to remove duplicates
    final_detections = non_max_suppression(detections)
    return final_detections
```

**Expected Output:**
```
Sliding window stats:
- Image size: 640×480
- Window sizes: [64×64, 128×128, 256×256]
- Stride: 32 pixels
- Total windows evaluated: 12,544
- Time: 28.3 seconds
- Detections found: 47 (before NMS)
- Final detections: 3 (after NMS)

Conclusion: Sliding windows work but are VERY slow!
```

**Why This Matters:**
Understanding the computational nightmare of sliding windows will make you appreciate the genius of YOLO (which processes the entire image in one pass) and R-CNN (which uses smart region proposals instead of exhaustive search).

**Run This:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)

---

## Learning Path Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│                    WEEK 13 LEARNING JOURNEY                        │
│              Building Blocks for Object Detection                  │
└────────────────────────────────────────────────────────────────────┘

                          START HERE
                               ↓
        ┌──────────────────────────────────────────┐
        │  [01] Object Detection Introduction      │
        │  ⏱ 10 min  │  🎯 Conceptual Foundation  │
        ├──────────────────────────────────────────┤
        │  Learn:                                  │
        │  ✓ Classification vs Detection           │
        │  ✓ Real-world applications               │
        │  ✓ Key challenges                        │
        │  ✓ Input/output formats                  │
        └──────────────────────────────────────────┘
                               ↓
        ┌──────────────────────────────────────────┐
        │  [02] IoU Calculation (Hands-On)         │
        │  ⏱ 12 min  │  🎯 Core Metric  ⭐         │
        ├──────────────────────────────────────────┤
        │  Learn:                                  │
        │  ✓ What is IoU?                          │
        │  ✓ Calculate from coordinates            │
        │  ✓ Interpret scores (0.0 to 1.0)         │
        │  ✓ IoU thresholds (0.5, 0.75)            │
        │                                          │
        │  Code: calculate_iou() from scratch      │
        └──────────────────────────────────────────┘
                               ↓
        ┌──────────────────────────────────────────┐
        │  [03] Bounding Box Visualization         │
        │  ⏱ 10 min  │  🎯 Practical Skills        │
        ├──────────────────────────────────────────┤
        │  Learn:                                  │
        │  ✓ Draw boxes with Matplotlib/OpenCV     │
        │  ✓ Add labels and confidence scores      │
        │  ✓ Color coding and best practices       │
        │  ✓ Compare predictions vs ground truth   │
        │                                          │
        │  Code: draw_bounding_box() utility       │
        └──────────────────────────────────────────┘
                               ↓
        ┌──────────────────────────────────────────┐
        │  [04] Evaluation Metrics (mAP)           │
        │  ⏱ 15 min  │  🎯 Performance Measurement │
        ├──────────────────────────────────────────┤
        │  Learn:                                  │
        │  ✓ Precision & Recall for detection      │
        │  ✓ Average Precision (AP) calculation    │
        │  ✓ mean Average Precision (mAP)          │
        │  ✓ Model comparison using mAP            │
        │                                          │
        │  Code: calculate_ap() and calculate_map()│
        └──────────────────────────────────────────┘
                               ↓
        ┌──────────────────────────────────────────┐
        │  [05] Naive Sliding Windows              │
        │  ⏱ 15 min  │  🎯 Classical Approach      │
        ├──────────────────────────────────────────┤
        │  Learn:                                  │
        │  ✓ Sliding window detection              │
        │  ✓ Multi-scale search                    │
        │  ✓ Why it's computationally expensive    │
        │  ✓ Motivation for modern methods         │
        │                                          │
        │  Code: Complete detector from scratch!   │
        └──────────────────────────────────────────┘
                               ↓
                    ✅ WEEK 13 COMPLETE!
                               ↓
        ┌──────────────────────────────────────────┐
        │         You're now ready for:            │
        │  → Week 14: YOLO (Single-shot detector)  │
        │  → Week 15: R-CNN (Two-stage detector)   │
        │  → Tutorial T13: YOLOv8 implementation   │
        └──────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│  💡 KEY INSIGHT: No Deep Learning This Week!                       │
│                                                                    │
│  We deliberately use only NumPy/Matplotlib/OpenCV to focus on     │
│  timeless fundamentals. These concepts apply to ANY detector      │
│  architecture - past, present, or future.                         │
└────────────────────────────────────────────────────────────────────┘
```

---

## Setup Instructions

### Option 1: Google Colab (Recommended) ✅

**Why Colab?**
- No installation needed
- All libraries pre-installed (NumPy, Matplotlib, OpenCV)
- Free tier is sufficient (NO GPU needed for Week 13!)
- Same environment for all students
- Easy sharing and collaboration

**Steps:**
1. **Open any notebook** by clicking the "Open in Colab" badge
2. **Sign in** with your Google account
3. **Run cells** using Shift+Enter or the Play button
4. **No GPU needed** - Keep default runtime settings

**First-Time Setup (30 seconds):**
```python
# Cell 1: Install OpenCV (only library not pre-installed in Colab)
!pip install opencv-python

# Cell 2: Verify installation
import numpy as np
import matplotlib.pyplot as plt
import cv2
print("✓ All libraries ready!")
```

**That's it!** You're ready to start learning.

---

### Option 2: Local Jupyter Notebook (Alternative)

**Requirements:**
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- Internet connection (for downloading sample images)

**Installation Steps:**

**Step 1: Create Virtual Environment (Recommended)**
```bash
# Navigate to week13 directory
cd course_planning/weekly_plans/week13-module5-object-localization

# Create virtual environment
python -m venv week13_env

# Activate environment
# On macOS/Linux:
source week13_env/bin/activate
# On Windows:
week13_env\Scripts\activate
```

**Step 2: Install Dependencies**
```bash
# Install all required packages
pip install -r requirements.txt

# OR install individually:
pip install numpy matplotlib opencv-python jupyter ipywidgets seaborn pillow scikit-learn pandas
```

**Step 3: Launch Jupyter**
```bash
# Start Jupyter Notebook
jupyter notebook

# OR start JupyterLab
jupyter lab

# Your browser will open automatically
```

**Step 4: Open Notebooks**
- Navigate to `notebooks/` folder
- Open `01_object_detection_introduction.ipynb`
- Run cells to verify setup

---

### Verification Checklist

Run this code in the first cell of any notebook to verify your environment:

```python
# Week 13 Environment Check
import sys
print(f"Python version: {sys.version}")

import numpy as np
print(f"✓ NumPy {np.__version__}")

import matplotlib
print(f"✓ Matplotlib {matplotlib.__version__}")

import cv2
print(f"✓ OpenCV {cv2.__version__}")

try:
    import PIL
    print(f"✓ Pillow {PIL.__version__}")
except ImportError:
    print("⚠ Pillow not installed (optional)")

print("\n✅ All core libraries ready for Week 13!")
print("🎯 No GPU required - NumPy computations only")
```

**Expected Output:**
```
Python version: 3.10.x
✓ NumPy 1.23.5
✓ Matplotlib 3.7.1
✓ OpenCV 4.8.0
✓ Pillow 9.4.0

✅ All core libraries ready for Week 13!
🎯 No GPU required - NumPy computations only
```

---

## Quick Start Guide

### For Students

Choose the learning path that fits your schedule and style:

---

#### **Path A: Complete Sequential Learning (60-70 min)**

**Best for:** Students who want deep understanding and plan to specialize in computer vision

**Timeline:**
```
Session 1 (30 min):
├── Notebook 01: Introduction (10 min)
├── Notebook 02: IoU Calculation (12 min)
└── Notebook 03: Visualization (10 min)

Session 2 (30 min):
├── Notebook 04: Evaluation Metrics (15 min)
└── Notebook 05: Sliding Windows (15 min)

Total: ~60 minutes + breaks
```

**Process:**
1. Open Notebook 01 → Read all markdown cells → Run all code cells → Complete exercises
2. Move to Notebook 02 → Repeat
3. Continue through Notebook 05
4. Review v1_comprehensive_lecture_notes.md for deeper theory
5. Attempt practice questions at the end of each notebook

**Expected Outcome:**
- Deep understanding of object detection fundamentals
- Ability to implement detection metrics from scratch
- Confidence to tackle YOLO and R-CNN in Weeks 14-15
- Strong foundation for computer vision career

---

#### **Path B: Essential Fast Track (30 min)**

**Best for:** Students who need core concepts quickly or have scheduling constraints

**Timeline:**
```
Focus Session (30 min):
├── Notebook 01: Intro - READ all, RUN key cells (7 min)
├── Notebook 02: IoU - FOCUS HERE, implement yourself (12 min) ⭐
├── Notebook 03: Viz - SKIM, run pre-written code (5 min)
├── Notebook 04: mAP - READ explanations, trust formulas (8 min)
└── Notebook 05: SKIP for now (revisit later if time)

Total: ~30 minutes
```

**Process:**
1. Notebook 01: Understand classification vs detection difference
2. Notebook 02: Master IoU calculation (MOST IMPORTANT!)
3. Notebook 03: See visualization examples, don't code yourself
4. Notebook 04: Understand what mAP measures, skip implementation details
5. Plan to revisit Notebook 05 before Week 14

**Expected Outcome:**
- Understand what object detection is and why it matters
- Can calculate IoU (the single most important metric)
- Prepared for Week 14 YOLO basics
- Foundation to build on later

---

#### **Path C: Homework Self-Study**

**Best for:** Students who attended live class demo and need to complete remaining notebooks

**Scenario:** Instructor demonstrated Notebooks 01-02 in class

**Your Homework:**
```
After Class (30 min):
├── Re-run Notebooks 01-02 yourself (10 min)
│   └── Don't just watch - type the code yourself!
├── Complete Notebooks 03-04 (20 min)
│   └── Focus on understanding, not speed
└── Submit: Screenshots of outputs + written answers

Before Next Class (20 min):
├── Complete Notebook 05 (15 min)
└── Review lecture notes (5 min)

Before FT2 Exam (15 min):
└── Review all notebooks and create 1-page cheat sheet
```

**Submission Requirements (if assigned):**
- Screenshots of final outputs from each notebook
- Answers to "TODO Exercise" questions
- 1-paragraph reflection: "What was most surprising/difficult?"

---

### For Instructors

Recommended classroom delivery strategies:

---

#### **Plan A: Full Interactive Session (90 min)**

**Best for:** 2-hour class slot with engaged students

```
Class Structure:
├── 00-10 min: Introduction
│   ├── Show real YOLO/R-CNN demos (YouTube videos)
│   ├── Explain Week 13-15 progression
│   └── Set learning objectives
│
├── 10-25 min: Notebook 01 (Live Demo)
│   ├── Share screen, open in Colab
│   ├── Run cells while explaining
│   ├── Emphasize classification vs detection difference
│   └── Show challenge scenarios (occlusion, scale)
│
├── 25-50 min: Notebook 02 (Interactive Coding) ⭐
│   ├── Explain IoU concept with hand-drawn diagrams
│   ├── Live code calculate_iou() function together
│   ├── Students code along in their own Colab
│   ├── Test on example boxes, debug together
│   └── "IoU is the heart of object detection!"
│
├── 50-60 min: Notebook 03 (Quick Demo)
│   ├── Show visualization examples
│   ├── Run pre-written code
│   └── Students will complete as homework
│
├── 60-75 min: Notebook 04 (Conceptual Walkthrough)
│   ├── Explain mAP with slides/whiteboard
│   ├── Show Precision-Recall curve examples
│   ├── Run mAP calculation code (don't code together)
│   └── Emphasize: "This is how YOLO/R-CNN are evaluated"
│
├── 75-85 min: Notebook 05 (Motivation Demo)
│   ├── Show sliding window animation
│   ├── Reveal computational cost (10,000+ windows!)
│   ├── Build excitement: "YOLO solves this in 0.03 sec!"
│   └── Preview Week 14 content
│
└── 85-90 min: Wrap-Up
    ├── Assign Notebooks 03-05 as homework
    ├── Announce Tutorial T13 next week
    └── Q&A

Homework Assignment:
- Complete Notebooks 03, 04, 05 individually
- Submit screenshots + exercise answers
- Due: Before next class
```

---

#### **Plan B: Focused Demo + Self-Study (50 min) ⭐ RECOMMENDED**

**Best for:** Standard class period, typical schedule

```
In-Class (50 min):
├── 00-10 min: Motivation
│   ├── Show object detection applications video
│   ├── Explain learning path (Weeks 13→14→15)
│   └── Open Notebook 01, quick walkthrough
│
├── 10-35 min: Deep Dive on IoU (Notebook 02)
│   ├── Whiteboard explanation of IoU formula
│   ├── Live code calculate_iou() together
│   ├── Students follow along in Colab
│   ├── Run on multiple examples
│   └── "Master IoU = Understand 50% of detection!"
│
├── 35-45 min: Quick Tour (Notebooks 03-05)
│   ├── Show Notebook 03 visualization outputs
│   ├── Explain Notebook 04 mAP concept
│   ├── Demo Notebook 05 sliding window problem
│   └── "You'll implement these for homework"
│
└── 45-50 min: Homework Briefing
    ├── Explain submission requirements
    ├── Point to v1_lecture_notes.md for reference
    └── Preview Tutorial T13 (YOLOv8)

Self-Study (Outside Class):
- Students complete all 5 notebooks individually
- Estimated time: 60 minutes
- Office hours available for questions
- Submit before Week 14 class
```

---

#### **Plan C: Lecture + Short Demo (30 min)**

**Best for:** Theory-focused course, limited lab time

```
Lecture Component (60 min):
├── Use v1_comprehensive_lecture_notes.md
├── Cover all concepts with slides/whiteboard
├── Emphasize IoU calculation (work through examples)
└── Explain mAP conceptually (no coding)

Live Demo Component (30 min):
├── Open Notebook 02 (IoU)
├── Run calculate_iou() on example boxes
├── Show visualizations of good/bad overlaps
├── Open Notebook 04, run mAP calculation
└── Show sliding window animation (Notebook 05)

Student Assignment:
├── All 5 notebooks assigned as homework
├── Provide video recordings of your demo
├── Graded assignment (10 points)
└── Due: 1 week
```

---

## Expected Results

### After Notebook 01: Object Detection Introduction

**Understanding:**
- ✓ Clear distinction: classification outputs a label, detection outputs multiple [label, box, confidence] tuples
- ✓ Recognition that detection is significantly harder due to variable number of objects
- ✓ Awareness of real-world applications and their requirements (accuracy vs speed tradeoffs)

**Outputs You'll See:**
```
Visualization 1: Classification
  → Image of cat → Output: "Cat"

Visualization 2: Localization
  → Image of cat → Output: "Cat" + bounding box [100, 50, 200, 150]

Visualization 3: Detection
  → Image with 4 animals → Output: List of 4 detections with boxes

Challenge Visualizations:
  → 6 images showing occlusion, small objects, crowds, scale variation
```

---

### After Notebook 02: IoU Calculation

**Skills Acquired:**
- ✓ Can calculate IoU between any two bounding boxes by hand
- ✓ Implemented `calculate_iou()` function that works for any box format
- ✓ Understands that IoU=0.5 is the minimum for "good detection" in COCO dataset
- ✓ Can debug detection issues by checking IoU values

**Code You Wrote:**
```python
def calculate_iou(box1, box2):
    # Your implementation
    x1_min, y1_min, x1_max, y1_max = convert_to_xyxy(box1)
    x2_min, y2_min, x2_max, y2_max = convert_to_xyxy(box2)

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    intersection = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection

    return intersection / union
```

**Test Results:**
```
Test Case 1: Perfect Overlap
  Box1: [100, 100, 200, 200]
  Box2: [100, 100, 200, 200]
  IoU: 1.00 ✓

Test Case 2: Partial Overlap
  Box1: [100, 100, 200, 200]
  Box2: [150, 150, 200, 200]
  IoU: 0.42 ✓

Test Case 3: No Overlap
  Box1: [100, 100, 200, 200]
  Box2: [400, 400, 200, 200]
  IoU: 0.00 ✓
```

---

### After Notebook 03: Bounding Box Visualization

**Skills Acquired:**
- ✓ Can overlay bounding boxes on any image using Matplotlib
- ✓ Implemented professional-looking detection visualizations
- ✓ Understands color conventions (green=ground truth, red=prediction)
- ✓ Can debug visually by comparing predicted vs ground truth boxes

**Visualizations Created:**
```
1. Single Detection:
   - Image of cat
   - Red bounding box
   - Label: "cat (0.95)"

2. Multi-Class Detection:
   - Image with 3 objects
   - Color-coded boxes: red (cat), blue (dog), green (person)
   - Labels with confidence scores

3. Comparison View:
   - Ground truth boxes (green, solid)
   - Predicted boxes (yellow, dashed)
   - Visual IoU assessment
```

**Code Pattern Learned:**
```python
# Standard pattern for detection visualization
fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(image)

for det in detections:
    x, y, w, h = det['bbox']
    conf = det['confidence']
    label = det['class_name']

    # Draw box
    rect = patches.Rectangle((x, y), w, h, linewidth=2,
                             edgecolor='red', facecolor='none')
    ax.add_patch(rect)

    # Add label
    ax.text(x, y-5, f"{label} ({conf:.2f})",
            color='white', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='red', alpha=0.8))
```

---

### After Notebook 04: Evaluation Metrics (mAP)

**Understanding Gained:**
- ✓ Can explain why accuracy is NOT a good metric for detection
- ✓ Understands precision-recall tradeoff in detection context
- ✓ Knows that mAP@0.5 means "mean AP at IoU threshold 0.5"
- ✓ Can interpret research papers that report mAP scores

**Metrics Calculated:**
```
Detection Results on Test Set (100 images, 3 classes):

Class: cat
  True Positives: 87
  False Positives: 13
  False Negatives: 8
  Precision: 0.87
  Recall: 0.92
  AP: 0.845

Class: dog
  True Positives: 76
  False Positives: 24
  False Negatives: 12
  Precision: 0.76
  Recall: 0.86
  AP: 0.782

Class: person
  True Positives: 92
  False Positives: 8
  False Negatives: 5
  Precision: 0.92
  Recall: 0.95
  AP: 0.912

mAP@0.5 = (0.845 + 0.782 + 0.912) / 3 = 0.846 ✓

Interpretation: Our detector achieves 84.6% mAP, meaning it correctly
localizes and classifies objects 84.6% of the time (at IoU≥0.5).
```

**Precision-Recall Curve:**
```
You'll see plots showing:
- X-axis: Recall (0 to 1)
- Y-axis: Precision (0 to 1)
- Curve shape: Typically decreasing from top-left to bottom-right
- Area under curve = AP
```

---

### After Notebook 05: Naive Sliding Windows

**Insights Gained:**
- ✓ Understands how object detection worked before deep learning
- ✓ Appreciates the computational challenge: 10,000+ windows per image!
- ✓ Motivated to learn YOLO (processes whole image in one pass)
- ✓ Ready to understand why R-CNN uses selective region proposals

**Implementation Completed:**
```python
# Your complete sliding window detector
def sliding_window_detector(image, classifier, scales=[0.5, 1.0, 1.5]):
    detections = []

    for scale in scales:
        window_size = int(128 * scale)
        stride = int(window_size * 0.5)  # 50% overlap

        for y in range(0, image.shape[0] - window_size, stride):
            for x in range(0, image.shape[1] - window_size, stride):
                # Extract window
                window = image[y:y+window_size, x:x+window_size]

                # Run classifier
                label, confidence = classifier(window)

                if confidence > 0.5:  # Threshold
                    detections.append({
                        'bbox': [x, y, window_size, window_size],
                        'label': label,
                        'confidence': confidence
                    })

    # Apply NMS to remove duplicates
    final_detections = non_max_suppression(detections, iou_threshold=0.3)
    return final_detections
```

**Performance Analysis:**
```
Sliding Window Detection Statistics:

Image: 640×480 pixels
Scales: [64×64, 128×128, 256×256]
Stride: 32 pixels

Windows Evaluated:
  - 64×64 scale: 3,600 windows
  - 128×128 scale: 1,680 windows
  - 256×256 scale: 330 windows
  - TOTAL: 5,610 windows

Raw Detections: 47 windows with confidence > 0.5
After NMS (IoU=0.3): 3 final detections

Processing Time:
  - Feature extraction: 22.4 seconds
  - Classification: 4.8 seconds
  - NMS: 0.1 seconds
  - TOTAL: 27.3 seconds per image

Comparison to YOLO:
  - YOLO processes same image in 0.03 seconds (900× faster!)
  - YOLO evaluates entire image in single pass (no redundant computation)
```

**Visual Output:**
```
You'll create a visualization showing:
- All 5,610 sliding windows (semi-transparent grid)
- 47 initial detections (yellow boxes)
- 3 final detections after NMS (green boxes)
- Annotation: "Evaluated 5,610 windows in 27.3 sec"
```

---

## Technical Requirements

### Hardware Requirements

**Minimum:**
- **Processor:** Any modern CPU (Intel i3 / AMD Ryzen 3 or better)
- **RAM:** 4 GB (8 GB recommended)
- **Storage:** 500 MB free space for notebooks and images
- **GPU:** NOT required (Week 13 uses NumPy only, no neural networks)

**Recommended:**
- **Processor:** Intel i5 / AMD Ryzen 5 or better (for faster NumPy operations)
- **RAM:** 8 GB (smoother Jupyter experience)
- **Internet:** Stable connection for Google Colab and downloading sample images

**Note:** Week 13 deliberately avoids deep learning to focus on fundamentals. GPU will be needed for Weeks 14-15 (YOLO/R-CNN), but Google Colab provides free GPU access.

---

### Software Requirements

**For Google Colab (Recommended):**
- ✓ Modern web browser (Chrome, Firefox, Safari, Edge)
- ✓ Google account (free)
- ✓ Internet connection
- ✓ **That's all!** No installation needed.

**For Local Jupyter:**
- **Python:** 3.8, 3.9, 3.10, or 3.11 (tested)
- **Package Manager:** pip or conda
- **Jupyter:** Notebook or JupyterLab
- **Required Libraries:** See `requirements.txt`

---

### Library Versions (Tested Configurations)

```
Core Libraries (Required):
├── numpy >= 1.21.0, < 2.0.0
├── matplotlib >= 3.5.0, < 4.0.0
└── opencv-python >= 4.6.0, < 5.0.0

Jupyter Environment:
├── jupyter >= 1.0.0
├── ipywidgets >= 7.7.0  (for interactive visualizations)
└── notebook >= 6.4.0

Visualization & Utilities:
├── seaborn >= 0.11.0  (for pretty plots)
├── pillow >= 9.0.0  (image handling)
├── scikit-learn >= 1.0.0  (for metrics utilities)
└── pandas >= 1.3.0  (for result tables)

NOT Required for Week 13:
✗ TensorFlow / PyTorch (not used until Weeks 14-15)
✗ ultralytics (YOLO library, used in Tutorial T13 later)
✗ GPU drivers (no neural networks this week)
```

---

### Accounts & Access

**Google Colab (Recommended Path):**
- ✓ **Free Google Account** (Gmail or institutional)
- ✓ **Colab Access:** https://colab.research.google.com/
- ✓ **Free Tier Limits:** 12 hours continuous runtime (more than enough)
- ✓ **Storage:** Notebooks saved to Google Drive
- ✓ **GPU:** Available but not needed for Week 13

**Optional But Helpful:**
- GitHub account (for accessing course repository)
- Kaggle account (for downloading additional datasets if interested)

---

## Troubleshooting

### Common Issues and Solutions

---

#### **Issue 1: OpenCV Import Error in Colab**

**Error Message:**
```
ModuleNotFoundError: No module named 'cv2'
```

**Solution:**
```python
# Run this cell in Colab
!pip install opencv-python

# Then restart runtime: Runtime → Restart runtime
# Then re-run your imports
import cv2
print(f"OpenCV version: {cv2.__version__}")  # Should work now
```

**Why This Happens:** OpenCV is the only library not pre-installed in Google Colab.

---

#### **Issue 2: Matplotlib Plots Not Showing**

**Symptoms:**
- Code runs without errors
- No plots appear below cells
- Blank output area

**Solution:**
```python
# Add this magic command at the top of your notebook
%matplotlib inline

# For interactive plots (optional):
%matplotlib widget

# Then run your plotting code
plt.figure()
plt.plot([1, 2, 3])
plt.show()
```

**Alternative:** Make sure you call `plt.show()` after creating plots.

---

#### **Issue 3: "Runtime Disconnected" in Colab**

**Error Message:**
```
Cannot connect to runtime
```

**Causes:**
- Browser tab inactive for long time
- Colab free tier usage limits
- Network interruption

**Solutions:**
1. **Reconnect:** Runtime → Reconnect to runtime
2. **Keep Tab Active:** Don't minimize browser for long periods
3. **Save Progress:** Download notebook frequently (File → Download .ipynb)
4. **Factory Reset:** If reconnecting fails: Runtime → Factory reset runtime

---

#### **Issue 4: Slow NumPy Operations**

**Symptoms:**
- IoU calculations take > 5 seconds for 100 boxes
- Sliding window notebook runs very slowly

**Solutions:**
1. **Check NumPy Version:** Make sure you have NumPy 1.21+
   ```python
   import numpy as np
   print(np.__version__)  # Should be 1.21 or higher
   ```

2. **Use Vectorized Operations:** Avoid Python loops for large arrays
   ```python
   # Slow (Python loop)
   for i in range(len(boxes)):
       ious[i] = calculate_iou(box, boxes[i])

   # Fast (vectorized NumPy)
   ious = np.array([calculate_iou(box, b) for b in boxes])
   ```

3. **Reduce Image Size:** In Notebook 05, use smaller test images
   ```python
   # Resize large images
   if image.shape[0] > 640:
       image = cv2.resize(image, (640, 480))
   ```

---

#### **Issue 5: Memory Error / Kernel Crash**

**Error Message:**
```
The kernel appears to have died. It will restart automatically.
```

**Causes:**
- Creating too many large arrays
- Loading high-resolution images without resizing
- Notebook 05 sliding windows creating thousands of patches

**Solutions:**
1. **Clear Variables:** Delete unused large arrays
   ```python
   del large_array  # Free memory
   import gc; gc.collect()  # Garbage collection
   ```

2. **Restart Kernel:** Kernel → Restart (or Restart & Clear Output)

3. **In Notebook 05:** Reduce number of scales or increase stride
   ```python
   # Original (might cause memory issues)
   scales = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
   stride = 8

   # Memory-friendly
   scales = [0.75, 1.0, 1.25]
   stride = 32
   ```

---

#### **Issue 6: IoU Calculation Returns Negative Values**

**Symptoms:**
- `calculate_iou()` returns negative numbers
- IoU values > 1.0

**Common Mistakes:**
1. **Wrong Coordinate Format:**
   ```python
   # If box is in [x, y, w, h] format, don't forget to convert to [x_min, y_min, x_max, y_max]
   x_max = x + width  # NOT just width
   y_max = y + height  # NOT just height
   ```

2. **Intersection Area Calculation:**
   ```python
   # Check for no overlap BEFORE calculating area
   if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
       return 0.0  # No intersection
   ```

3. **Union Formula:**
   ```python
   # CORRECT
   union = area1 + area2 - intersection

   # WRONG (double counts intersection)
   union = area1 + area2
   ```

**Debug Strategy:**
- Print intermediate values (inter_x_min, inter_x_max, intersection, union)
- Test on simple cases first (perfect overlap should give IoU=1.0)

---

#### **Issue 7: Jupyter Notebook Won't Start Locally**

**Error Message:**
```
jupyter: command not found
```

**Solution:**
```bash
# Make sure Jupyter is installed in your active environment
pip install jupyter

# Verify installation
jupyter --version

# If using conda
conda install jupyter

# Then try again
jupyter notebook
```

**If Port 8888 is Busy:**
```bash
# Start Jupyter on different port
jupyter notebook --port 8889

# Or find and kill process using port 8888
# On macOS/Linux:
lsof -ti:8888 | xargs kill -9

# On Windows:
netstat -ano | findstr :8888
taskkill /PID <PID> /F
```

---

#### **Issue 8: Visualization Colors Look Wrong**

**Symptoms:**
- Images have strange colors (blue sky looks orange)
- Bounding boxes misaligned

**Cause:** OpenCV uses BGR format, Matplotlib uses RGB

**Solution:**
```python
# When loading with OpenCV
image_bgr = cv2.imread('image.jpg')

# Convert to RGB before displaying with Matplotlib
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Now plot
plt.imshow(image_rgb)  # Colors correct now!
```

---

#### **Issue 9: mAP Calculation Gives Unexpected Values**

**Symptoms:**
- mAP = 0.0 even though predictions look good
- mAP > 1.0 (impossible!)

**Common Mistakes:**
1. **Forgot to Sort by Confidence:**
   ```python
   # MUST sort detections by confidence (descending) before calculating AP
   detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
   ```

2. **Wrong IoU Threshold:**
   ```python
   # Make sure you're using consistent threshold
   iou_threshold = 0.5  # COCO standard
   is_tp = iou >= iou_threshold  # Not >
   ```

3. **Not Handling Multiple Detections per Ground Truth:**
   ```python
   # Each ground truth can only be matched once
   # Mark ground truth as "used" after first TP match
   ```

**Debug Strategy:**
- Calculate AP for single class first (easier to debug)
- Print precision/recall at each detection
- Verify ground truth count matches expected

---

### Getting Help

**If issues persist:**

1. **Check Notebook Comments:** Many cells have troubleshooting hints
2. **Review Lecture Notes:** `v1_comprehensive_lecture_notes.md` has detailed explanations
3. **Ask Instructor:** Office hours or class Q&A session
4. **Peer Discussion:** Study groups can help debug together
5. **Online Resources:**
   - NumPy documentation: https://numpy.org/doc/
   - Matplotlib gallery: https://matplotlib.org/stable/gallery/
   - OpenCV tutorials: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html

---

## Assessment Alignment

### How Week 13 Prepares You for FT2 (Formative Test 2)

**Exam Date:** November 14, 2025
**Coverage:** Modules 3-4 (CNNs, Transfer Learning) + Module 5 Introduction (Week 13)
**Format:** 40 MCQs (40 marks) + 2 SAQs (10 marks) = 50 marks total

---

### Week 13 Topics in FT2

**Expected MCQ Topics (5-8 questions):**

1. **Object Detection Basics (2-3 MCQs)**
   - Q: "What is the output format of an object detection system?"
   - Q: "Which task outputs multiple [label, bounding box] pairs?"
   - Q: "What is the difference between localization and detection?"
   - **Notebook 01 prepares you for these**

2. **IoU Calculation (2-3 MCQs)**
   - Q: "If IoU = 0.8, what does this indicate?"
   - Q: "What IoU threshold is commonly used in COCO dataset?"
   - Q: "Given two boxes, calculate the IoU (show formula)"
   - **Notebook 02 prepares you for these**

3. **Evaluation Metrics (1-2 MCQs)**
   - Q: "What does mAP@0.5 measure?"
   - Q: "Why is accuracy not suitable for object detection evaluation?"
   - **Notebook 04 prepares you for these**

**Expected SAQ Topics (1 question, 5 marks):**

**SAQ: Explain Object Detection with Example (5 marks)**
```
Question: "Explain the object detection task with a real-world example.
How is it different from classification? What are the key challenges?"

Expected Answer Structure:
├── Definition (1 mark): Detection finds all objects and their locations
├── Example (1 mark): Autonomous car detecting pedestrians, cars, signs
├── Difference from classification (1 mark): Classification outputs 1 label,
│   detection outputs N labels + bounding boxes + confidence scores
├── Challenges (2 marks): Occlusion, scale variation, speed requirements,
│   variable number of objects
```

**How Notebooks Help:**
- Notebook 01: Provides perfect examples and challenge scenarios
- Notebook 02: Explains bounding box representation (often asked)
- Notebook 04: Clarifies evaluation (comparing models)

---

### Key Concepts to Memorize for FT2

**From Week 13:**

1. **IoU Formula:**
   ```
   IoU = Intersection Area / Union Area

   Intersection = max(0, min(x1_max, x2_max) - max(x1_min, x2_min)) ×
                  max(0, min(y1_max, y2_max) - max(y1_min, y2_min))

   Union = Area1 + Area2 - Intersection
   ```

2. **IoU Thresholds:**
   - IoU ≥ 0.5: Considered "correct detection" in COCO
   - IoU ≥ 0.75: Strict threshold for high-quality detection
   - IoU < 0.5: False Positive

3. **Detection Output Format:**
   ```
   [
       {'class': 'cat', 'bbox': [x, y, w, h], 'confidence': 0.95},
       {'class': 'dog', 'bbox': [x, y, w, h], 'confidence': 0.87},
       ...
   ]
   ```

4. **mAP Definition:**
   - mAP = mean Average Precision
   - AP = Area under Precision-Recall curve (per class)
   - mAP = Average of AP across all classes
   - mAP@0.5 means "mAP at IoU threshold 0.5"

5. **Precision vs Recall:**
   ```
   Precision = TP / (TP + FP)  → "How many detections are correct?"
   Recall = TP / (TP + FN)     → "How many objects were found?"
   ```

---

### Practice Questions (FT2 Style)

**MCQ Practice:**

**Q1:** Which task requires outputting multiple bounding boxes per image?
- A) Image classification
- B) Object localization
- C) Object detection ✓
- D) Semantic segmentation

**Q2:** If a predicted bounding box has IoU = 0.65 with ground truth, and the threshold is 0.5, this is classified as:
- A) False Positive
- B) True Positive ✓
- C) False Negative
- D) True Negative

**Q3:** What is the primary reason sliding window detection is impractical?
- A) Cannot detect multiple objects
- B) Computationally expensive (10,000+ windows per image) ✓
- C) Requires labeled data
- D) Cannot handle different object scales

**Q4:** If a model achieves mAP@0.5 = 0.78, this means:
- A) 78% of pixels are correctly classified
- B) 78% average precision across all classes at IoU≥0.5 ✓
- C) 78% of images contain detected objects
- D) Detection speed is 78 FPS

**SAQ Practice:**

**Q5:** (5 marks) Explain Intersection over Union (IoU) and its role in object detection. Include the formula and interpretation of IoU values.

**Model Answer:**
```
IoU (Intersection over Union) measures the overlap between predicted and
ground truth bounding boxes. (1 mark)

Formula: IoU = (Area of Intersection) / (Area of Union)
Where Union = Area1 + Area2 - Intersection (1 mark)

Interpretation:
- IoU = 1.0: Perfect overlap (predicted box exactly matches ground truth)
- IoU = 0.5-0.7: Moderate overlap (acceptable detection)
- IoU < 0.5: Poor overlap (usually considered False Positive)
- IoU = 0.0: No overlap (1.5 marks)

Role in Detection:
IoU is used for:
1. Evaluating detection quality (is prediction correct?)
2. Non-Maximum Suppression (removing duplicate detections)
3. Calculating mAP metric (determines True Positives) (1.5 marks)
```

**Study Strategy:**
- Review all 5 notebooks before FT2
- Memorize IoU formula and interpretation
- Practice calculating IoU by hand (2-3 examples)
- Understand mAP conceptually (don't need to memorize full algorithm)
- Focus on Notebooks 01, 02, 04 (most likely to be tested)

---

## Learning Outcomes

### What You'll Be Able to Do After Week 13

---

#### **1. Conceptual Understanding (Knowledge)**

**Aligned with Course Outcomes CO-1, CO-5**

You will be able to:
- ✅ **Define object detection** and differentiate it from classification and localization
- ✅ **Explain the challenges** unique to detection: variable object count, occlusion, scale variation
- ✅ **Identify applications** where detection is critical (autonomous vehicles, medical imaging, surveillance)
- ✅ **Describe the evolution** from classical methods (sliding windows) to modern deep learning approaches
- ✅ **Understand the computational problem** that YOLO and R-CNN solve

**Assessment:**
- Can explain in 2-3 sentences what object detection is to a non-technical person
- Can list 5 real-world applications and their specific requirements
- Can explain why sliding windows are impractical (computational cost)

---

#### **2. Mathematical Skills (Comprehension & Application)**

**Aligned with Course Outcomes CO-2, CO-3**

You will be able to:
- ✅ **Calculate IoU** between any two bounding boxes (by hand or code)
- ✅ **Convert between coordinate formats**: XYWH ↔ XYXY ↔ center-based
- ✅ **Interpret IoU values**: Understand what 0.3, 0.5, 0.75, 1.0 mean qualitatively
- ✅ **Apply IoU thresholds** correctly in different contexts (NMS, mAP calculation)
- ✅ **Compute intersection and union areas** from box coordinates

**Assessment:**
- Given two boxes, can calculate IoU in <2 minutes
- Can implement `calculate_iou()` from memory in 5-10 minutes
- Can explain why IoU is better than simple distance metrics

---

#### **3. Implementation Skills (Application & Analysis)**

**Aligned with Course Outcomes CO-4, CO-5**

You will be able to:
- ✅ **Implement bounding box utilities** from scratch (drawing, coordinate conversion)
- ✅ **Code Non-Maximum Suppression** to eliminate duplicate detections
- ✅ **Visualize detections** professionally using Matplotlib/OpenCV
- ✅ **Build a complete sliding window detector** (even though it's slow)
- ✅ **Debug detection systems** by visualizing IoU, inspecting predictions

**Code Patterns Mastered:**
```python
# Pattern 1: IoU Calculation
def calculate_iou(box1, box2): ...

# Pattern 2: Bounding Box Visualization
def draw_bounding_box(image, bbox, label, confidence): ...

# Pattern 3: Non-Maximum Suppression
def non_max_suppression(detections, iou_threshold): ...

# Pattern 4: Coordinate Conversion
def xywh_to_xyxy(bbox): ...
def xyxy_to_xywh(bbox): ...
```

**Assessment:**
- Can implement any of the above functions from scratch in 10-15 minutes
- Can debug a detection visualization by inspecting coordinates
- Can modify code to handle edge cases (boxes outside image boundaries)

---

#### **4. Evaluation Skills (Analysis & Evaluation)**

**Aligned with Course Outcomes CO-5**

You will be able to:
- ✅ **Calculate detection metrics**: Precision, Recall, F1, AP, mAP
- ✅ **Interpret mAP scores**: Understand what "mAP@0.5 = 0.72" means
- ✅ **Compare detection models** using standardized metrics
- ✅ **Analyze failure cases**: Identify why false positives/negatives occur
- ✅ **Choose appropriate IoU thresholds** for different applications

**Metric Interpretation Skills:**
```
Scenario 1: Model A has mAP@0.5 = 0.65, Model B has mAP@0.75 = 0.45
Q: Which model is better?
A: Depends on application. Model A is better overall, but Model B might
   be better if you need very precise localization (IoU>0.75).

Scenario 2: Detector has Precision=0.95, Recall=0.40
Q: What's the problem?
A: Too conservative. Detector is very accurate when it fires (high precision)
   but misses many objects (low recall). Should lower confidence threshold.
```

**Assessment:**
- Can calculate mAP for a small dataset (10 images) by hand
- Can interpret precision-recall curves
- Can recommend model improvements based on metric analysis

---

#### **5. Preparation for Advanced Topics (Synthesis)**

**Aligned with Course Outcomes CO-5**

You will be able to:
- ✅ **Understand YOLO architecture** (Week 14) because you know IoU and mAP
- ✅ **Understand R-CNN pipeline** (Week 15) because you know region proposals vs sliding windows
- ✅ **Read research papers** on object detection with full comprehension
- ✅ **Implement custom detectors** using TensorFlow/PyTorch in future projects
- ✅ **Contribute to open-source detection projects** (e.g., YOLOv8, Detectron2)

**Real-World Readiness:**
After Week 13 + Weeks 14-15, you'll be ready to:
- Build an autonomous robot that detects obstacles
- Create a medical imaging tool that detects tumors
- Develop a retail analytics system that counts products
- Contribute to self-driving car perception systems
- Build augmented reality apps with object tracking

---

### Self-Assessment Checklist

**After completing Week 13, can you:**

**Basic (Must-Have):**
- [ ] Explain the difference between classification and detection in 30 seconds
- [ ] Calculate IoU between two boxes by hand in 2 minutes
- [ ] Implement `calculate_iou()` from memory in 10 minutes
- [ ] Draw bounding boxes on an image with labels
- [ ] Explain why sliding windows are computationally expensive

**Intermediate (Should-Have):**
- [ ] Implement Non-Maximum Suppression from scratch
- [ ] Calculate precision and recall for detection results
- [ ] Convert between XYWH and XYXY coordinate formats
- [ ] Visualize detection results like professional tools (YOLO-style)
- [ ] Debug detection failures by inspecting IoU values

**Advanced (Nice-to-Have):**
- [ ] Calculate mAP for a multi-class detection task
- [ ] Explain the difference between mAP@0.5 and mAP@0.75
- [ ] Implement a complete sliding window detector
- [ ] Analyze a detection model's performance and suggest improvements
- [ ] Explain how YOLO/R-CNN improve upon sliding windows

**If you can check all "Basic" boxes, you're ready for Week 14!**

---

## Assignment Ideas

Optional homework assignments to deepen understanding:

---

### **Assignment 1: IoU Calculator (5 points)**

**Objective:** Master IoU calculation through implementation and testing

**Task:**
1. Implement `calculate_iou(box1, box2)` that handles BOTH XYWH and XYXY formats
2. Add input validation (negative dimensions, boxes outside image, etc.)
3. Test on at least 10 different box pairs
4. Create visualization showing intersection and union areas for each pair

**Deliverables:**
- Python file: `iou_calculator.py`
- Test cases: `iou_tests.py` with 10+ test cases
- Visualization: `iou_visualizations.png` showing 6 examples

**Grading Rubric:**
- Correct implementation (2 points)
- Edge case handling (1 point)
- Comprehensive tests (1 point)
- Clear visualizations (1 point)

---

### **Assignment 2: Detection Visualizer Tool (10 points)**

**Objective:** Build a reusable visualization utility for any detection task

**Task:**
Create a tool that:
1. Loads an image
2. Reads detection results from JSON file
3. Overlays bounding boxes with labels and confidence scores
4. Color-codes boxes by class or confidence
5. Saves annotated image

**Input Format (JSON):**
```json
{
  "image": "path/to/image.jpg",
  "detections": [
    {"class": "cat", "bbox": [100, 50, 200, 150], "confidence": 0.95},
    {"class": "dog", "bbox": [350, 200, 180, 220], "confidence": 0.87}
  ]
}
```

**Deliverables:**
- Script: `detection_visualizer.py`
- Sample JSON: `sample_detections.json`
- Output: `visualized_detections.jpg`
- README: Explain usage and parameters

**Bonus (+2 points):** Add command-line interface with argparse

---

### **Assignment 3: mAP Calculator from Scratch (15 points)**

**Objective:** Implement the core evaluation metric for object detection

**Task:**
1. Given ground truth and predictions (provided JSON files), calculate:
   - Precision and Recall for each confidence threshold
   - Average Precision (AP) for each class
   - mean Average Precision (mAP) across all classes
2. Plot Precision-Recall curves for each class
3. Write a report explaining your results

**Provided Data:**
- `ground_truth.json`: 50 images with labeled objects
- `predictions.json`: Model predictions on the same 50 images

**Deliverables:**
- Python file: `map_calculator.py`
- Output: `pr_curves.png` (Precision-Recall curves)
- Report: `map_analysis.pdf` (2 pages maximum)

**Grading Rubric:**
- Correct TP/FP/FN calculation (5 points)
- Correct AP calculation (5 points)
- Clear visualizations (3 points)
- Insightful analysis in report (2 points)

---

### **Assignment 4: Sliding Window Optimization (20 points)**

**Objective:** Improve naive sliding window detector performance

**Task:**
Starting with Notebook 05's sliding window detector, optimize it:
1. **Implement selective search** instead of exhaustive sliding
2. **Add image pyramid** for multi-scale detection
3. **Optimize NMS** (use vectorized operations)
4. **Measure performance improvement** (speed and accuracy)

**Deliverables:**
- Code: `optimized_detector.py`
- Benchmark: `performance_comparison.csv` (before/after metrics)
- Visualization: `detection_results.jpg` (showing detections)
- Report: `optimization_report.pdf` (3 pages)

**Metrics to Compare:**
- Windows evaluated (before: 10,000+, target: <1,000)
- Processing time (before: 27 sec, target: <5 sec)
- Detection accuracy (maintain or improve mAP)

**Grading Rubric:**
- Selective search implementation (6 points)
- Multi-scale pyramid (4 points)
- NMS optimization (4 points)
- Performance analysis (4 points)
- Code quality and documentation (2 points)

---

### **Assignment 5: Real-World Detection Application (25 points)**

**Objective:** Apply Week 13 concepts to a real problem of your choice

**Task:**
Choose a domain and build a prototype detection system:

**Domain Options:**
- **Medical:** Detect cells/lesions in microscopy images
- **Agriculture:** Detect fruits/vegetables for yield estimation
- **Wildlife:** Detect animals in camera trap images
- **Retail:** Detect products on shelves
- **Traffic:** Detect vehicles and pedestrians

**Requirements:**
1. Collect or find dataset (min 50 images, 3+ classes)
2. Annotate bounding boxes (use LabelImg or similar tool)
3. Implement evaluation using your mAP calculator
4. Create visualization of detection results
5. Write comprehensive report with analysis

**Deliverables:**
- Code: `my_detector.py`
- Dataset: `annotations.json` + images
- Results: `detection_samples/` folder (10+ annotated images)
- Report: `final_report.pdf` (5 pages max)
  - Problem description
  - Dataset details
  - Implementation approach
  - Results and analysis
  - Challenges and future work

**Grading Rubric:**
- Problem significance (5 points)
- Dataset quality (5 points)
- Implementation correctness (5 points)
- Results analysis (5 points)
- Report clarity and depth (5 points)

**Bonus (+5 points):** Deploy as web app using Gradio/Streamlit

---

## Additional Resources

### Official Documentation

**Python Libraries:**
- **NumPy User Guide:** https://numpy.org/doc/stable/user/
  - Array operations, broadcasting, vectorization
- **Matplotlib Tutorials:** https://matplotlib.org/stable/tutorials/index.html
  - Plotting, subplots, patches (for bounding boxes)
- **OpenCV Python Tutorials:** https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
  - Image I/O, drawing functions, color conversion

**Object Detection Frameworks (for future):**
- **YOLOv8 Documentation:** https://docs.ultralytics.com/
- **TensorFlow Object Detection API:** https://github.com/tensorflow/models/tree/master/research/object_detection
- **Detectron2 (Meta AI):** https://detectron2.readthedocs.io/

---

### Research Papers (Foundational)

**Must-Read (for deep understanding):**

1. **"Rich feature hierarchies for accurate object detection and semantic segmentation" (R-CNN)**
   - Girshick et al., 2014 (CVPR)
   - Link: https://arxiv.org/abs/1311.2524
   - Why: Introduced region-based detection (Week 15 material)

2. **"You Only Look Once: Unified, Real-Time Object Detection" (YOLO)**
   - Redmon et al., 2016 (CVPR)
   - Link: https://arxiv.org/abs/1506.02640
   - Why: Single-shot detection breakthrough (Week 14 material)

3. **"Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"**
   - Ren et al., 2015 (NeurIPS)
   - Link: https://arxiv.org/abs/1506.01497
   - Why: Introduced learnable region proposals

4. **"Microsoft COCO: Common Objects in Context"**
   - Lin et al., 2014 (ECCV)
   - Link: https://arxiv.org/abs/1405.0312
   - Why: Standard dataset and evaluation metrics (mAP@[0.5:0.95])

**Optional (advanced topics):**
- **EfficientDet:** Tan et al., 2020 (CVPR) - Scalable detection architecture
- **DETR:** Carion et al., 2020 (ECCV) - Transformer-based detection
- **Mask R-CNN:** He et al., 2017 (ICCV) - Instance segmentation

---

### Video Tutorials & Lectures

**YouTube Channels:**

1. **Stanford CS231n: Convolutional Neural Networks for Visual Recognition**
   - Lecture 11: Detection and Segmentation
   - Link: https://www.youtube.com/watch?v=nDPWywWRIRo
   - Duration: 1 hour 15 min
   - Quality: Excellent theoretical foundation

2. **Two Minute Papers**
   - Search: "object detection YOLO" or "R-CNN"
   - Digestible 5-min summaries of research papers

3. **Yannic Kilcher**
   - Paper explanations for YOLO, R-CNN, DETR
   - Very detailed technical breakdowns

**Coursera / Udacity:**
- **DeepLearning.AI TensorFlow Developer Specialization**
  - Course 4: Sequences, Time Series and Prediction (includes detection)
- **Udacity Computer Vision Nanodegree**
  - Object Tracking and Localization module

---

### Interactive Tools & Demos

**Try Object Detection Live:**

1. **YOLOv8 Demo (Hugging Face):**
   - Link: https://huggingface.co/spaces/kadirnar/yolov8
   - Upload your own images, see detections instantly

2. **TensorFlow Detection Model Zoo:**
   - Link: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
   - Pre-trained models on COCO dataset

3. **Roboflow Universe:**
   - Link: https://universe.roboflow.com/
   - Browse 100,000+ public detection datasets

**Annotation Tools (for creating your own datasets):**
- **LabelImg:** https://github.com/heartexlabs/labelImg
  - Desktop tool for bounding box annotation
- **Roboflow Annotate:** https://roboflow.com/annotate
  - Web-based, collaborative annotation
- **CVAT (Computer Vision Annotation Tool):** https://github.com/opencv/cvat
  - Professional-grade, supports multiple formats

---

### Datasets for Practice

**Beginner-Friendly (small, well-labeled):**

1. **PASCAL VOC 2012**
   - 20 classes (person, car, cat, dog, etc.)
   - 11,530 images
   - Link: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

2. **Oxford-IIIT Pet Dataset**
   - 37 pet breeds
   - 7,349 images with bounding boxes
   - Link: https://www.robots.ox.ac.uk/~vgg/data/pets/

3. **TensorFlow Object Detection Dataset**
   - Multiple small datasets (raccoon, pedestrians, etc.)
   - Link: https://github.com/tensorflow/models/tree/master/research/object_detection/data

**Intermediate (standard benchmarks):**

4. **COCO (Common Objects in Context)**
   - 80 classes
   - 330,000 images (118,000 training)
   - Link: https://cocodataset.org/
   - Standard for comparing detection models

5. **KITTI (Autonomous Driving)**
   - 7 classes (car, pedestrian, cyclist, etc.)
   - 7,481 training images
   - Link: http://www.cvlibs.net/datasets/kitti/

**Advanced (specialized domains):**

6. **xView (Satellite Imagery)**
   - 60 classes
   - 1 million objects in satellite images
   - Link: http://xviewdataset.org/

7. **Open Images V7**
   - 600 classes
   - 9 million images (1.9M with bounding boxes)
   - Link: https://storage.googleapis.com/openimages/web/index.html

---

### Books & Textbooks

**Computer Vision Fundamentals:**

1. **"Computer Vision: Algorithms and Applications" (2nd Edition)**
   - Richard Szeliski, 2022
   - Chapter 6: Feature detection and matching
   - Chapter 7: Object recognition
   - Free online: https://szeliski.org/Book/

2. **"Deep Learning for Vision Systems"**
   - Mohamed Elgendy, 2020 (Manning)
   - Chapter 7: Object Detection (YOLO, R-CNN)
   - Beginner-friendly with code examples

**Deep Learning (for Week 13 foundations):**

3. **"Deep Learning" (Goodfellow, Bengio, Courville)**
   - Chapter 9: Convolutional Networks
   - Free online: https://www.deeplearningbook.org/

4. **"Dive into Deep Learning"**
   - Zhang et al., 2023
   - Chapter 13: Computer Vision (includes detection)
   - Interactive notebooks: https://d2l.ai/

---

### Blogs & Tutorials

**Highly Recommended:**

1. **"A Gentle Introduction to Object Detection" (Machine Learning Mastery)**
   - Link: https://machinelearningmastery.com/object-recognition-with-deep-learning/

2. **"Understanding YOLO" (What-When-How)**
   - Step-by-step breakdown of YOLO architecture

3. **"mAP (mean Average Precision) Explained"**
   - Link: https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173
   - Excellent visual explanations

4. **PyImageSearch Blog**
   - Author: Adrian Rosebrock
   - Many practical object detection tutorials
   - Link: https://pyimagesearch.com/

---

### Community & Forums

**Ask Questions:**
- **Stack Overflow:** Tag with `object-detection`, `computer-vision`
- **Reddit:** r/computervision, r/MachineLearning
- **Cross Validated:** For statistics/metrics questions (mAP, precision-recall)

**Stay Updated:**
- **Papers with Code:** https://paperswithcode.com/task/object-detection
  - Latest research with code implementations
  - Leaderboards for standard benchmarks
- **Arxiv Sanity:** http://www.arxiv-sanity.com/
  - Search "object detection" for recent papers

---

## Session Timeline

### Week 13 in Course Schedule

**Course Context:**
- **Week 10-11:** CNN Basics (Convolutions, Pooling, Architectures)
- **Week 12:** Transfer Learning (Feature Extraction, Fine-Tuning)
- **Week 13:** Object Detection Foundations ← YOU ARE HERE
- **Week 14:** YOLO (Single-shot detector)
- **Week 15:** R-CNN (Two-stage detector)

---

### Detailed Week 13 Schedule

**Session 1: Friday, November [Date TBD] (DO3 - 100 minutes)**

```
08:00-08:10 (10 min): Week 13 Introduction
├── Show YOLO/R-CNN demo videos (autonomous cars, retail analytics)
├── Explain Week 13-15 progression
├── Set expectations: "No deep learning this week - focus on foundations"
└── Discuss FT2 exam (Nov 14) and what to prioritize

08:10-08:30 (20 min): Notebook 01 - Live Walkthrough
├── Open in Google Colab (share link)
├── Run cells and explain classification vs detection
├── Discuss real-world applications
├── Show challenge scenarios (occlusion, scale)
└── Students: Open Notebook 01 on your own device

08:30-09:00 (30 min): Notebook 02 - Interactive Coding Session ⭐
├── Explain IoU concept (whiteboard + slides)
├── Live code calculate_iou() function together
├── Students code along in their Colab
├── Test on 3-4 example box pairs
├── Debug common mistakes together
└── Emphasize: "IoU is THE most important metric!"

09:00-09:10 (10 min): Break

09:10-09:30 (20 min): Notebook 03 & 04 - Quick Demo
├── Notebook 03: Show bounding box visualization examples
├── Notebook 04: Explain mAP conceptually (don't code together)
├── Run pre-written code, show outputs
└── "You'll complete these as homework"

09:30-09:40 (10 min): Notebook 05 - Motivation
├── Show sliding window animation
├── Reveal computational cost
├── "This is why we need YOLO next week!"
└── Build excitement for Week 14

09:40-09:50 (10 min): Homework Assignment & Wrap-Up
├── Assign Notebooks 03, 04, 05 for completion
├── Explain submission requirements
├── Point to v1_lecture_notes.md and this README
├── Remind about FT2 exam preparation
└── Q&A
```

---

**Session 2: Monday, November [Date TBD] (DO4 - 40 minutes)**

**Option A: Tutorial T13 Preparation**
```
Tutorial T13: Object Detection with YOLOv8 (Coming Soon)
├── Students complete setup (install ultralytics)
├── Download sample dataset
├── Pre-load YOLOv8 weights
└── Prepare for next week's hands-on session
```

**Option B: Week 13 Review & Office Hours**
```
If T13 not ready, use session for:
├── 10 min: Review Week 13 concepts (IoU, mAP)
├── 10 min: Live Q&A on homework notebooks
├── 10 min: FT2 exam preparation tips
└── 10 min: Preview Week 14 YOLO architecture
```

---

**Between Sessions: Homework (60-70 minutes)**

**Required:**
- [ ] Complete Notebook 03: Bounding Box Visualization
- [ ] Complete Notebook 04: Evaluation Metrics (mAP)
- [ ] Complete Notebook 05: Sliding Windows
- [ ] Answer all "TODO Exercise" questions in notebooks
- [ ] Submit screenshots + written answers (if graded)

**Optional (Recommended):**
- [ ] Read v1_comprehensive_lecture_notes.md (18 pages)
- [ ] Review IoU calculation - practice 5 examples by hand
- [ ] Explore additional resources (YouTube videos, blogs)
- [ ] Attempt one of the assignment ideas (for extra practice)

---

**Friday, November 14 (100 minutes)**

**FORMATIVE TEST 2 (FT2)**
- Modules 3-4: CNNs, Transfer Learning
- Module 5: Object Detection Introduction (Week 13 concepts)
- Format: 40 MCQs + 2 SAQs = 50 marks
- Expected Week 13 questions: 5-8 MCQs + possibly 1 SAQ

---

### Study Timeline for FT2

**Week 13 Focus Areas:**

**High Priority (Most Likely to Be Tested):**
- [ ] IoU calculation and interpretation (Notebook 02)
- [ ] Object detection definition and output format (Notebook 01)
- [ ] mAP concept and why it's used (Notebook 04)
- [ ] Difference between classification, localization, detection (Notebook 01)

**Medium Priority:**
- [ ] Bounding box coordinate systems (XYWH vs XYXY)
- [ ] Precision and Recall for detection
- [ ] Why sliding windows are impractical
- [ ] Real-world applications and challenges

**Lower Priority (Good to Know):**
- [ ] Non-Maximum Suppression algorithm details
- [ ] AP calculation step-by-step
- [ ] Specific IoU threshold values (0.5 vs 0.75)

**Study Plan:**
```
3 Days Before FT2:
└── Review all 5 notebooks (skim code, focus on concepts)

2 Days Before FT2:
├── Memorize IoU formula
├── Practice IoU calculation (5 examples by hand)
└── Review mAP definition

1 Day Before FT2:
├── Read Notebook 01 carefully (definitions)
├── Re-do Notebook 02 exercises (IoU)
└── Create 1-page cheat sheet (formulas, definitions)

Morning of FT2:
└── Review cheat sheet, relax, you're prepared!
```

---

## Checklist for Students

### Before Class (Preparation)

**Technical Setup:**
- [ ] Create or sign in to Google account (for Colab access)
- [ ] Bookmark this README: `week13-module5-object-localization/README.md`
- [ ] Test Colab access: Open https://colab.research.google.com/ and create blank notebook
- [ ] Verify you can run Python code in Colab (try `print("Hello")`)

**Knowledge Preparation:**
- [ ] Review Week 10-12 concepts (CNNs, transfer learning)
- [ ] Refresh NumPy basics: arrays, slicing, broadcasting
- [ ] Understand image representation: (Height, Width, Channels)
- [ ] Read this README overview section

**Mindset:**
- [ ] Understand that Week 13 is FOUNDATION for Weeks 14-15
- [ ] Accept that no deep learning this week is intentional
- [ ] Come with questions about object detection (write them down)

---

### During Class (Active Learning)

**Session 1 (Introduction & Hands-On):**

**First 10 Minutes:**
- [ ] Pay attention to YOLO/R-CNN demo videos (motivation)
- [ ] Understand Week 13→14→15 learning path
- [ ] Note which topics will be in FT2 exam

**Notebook 01 Walkthrough (20 min):**
- [ ] Open Notebook 01 in your own Colab (don't just watch instructor)
- [ ] Run cells yourself while instructor explains
- [ ] Ask questions if classification vs detection difference is unclear
- [ ] Take notes on real-world applications mentioned

**Notebook 02 Interactive Coding (30 min):**
- [ ] Open Notebook 02 in Colab
- [ ] Code along with instructor (type the code yourself!)
- [ ] Test your `calculate_iou()` function on examples
- [ ] Debug if your results don't match instructor's
- [ ] Ask for help if stuck (don't fall behind on this critical notebook)

**After Break:**
- [ ] Watch Notebook 03-05 demos attentively
- [ ] Note which notebooks are homework assignments
- [ ] Ask clarifying questions about mAP concept if confused

**Before Leaving Class:**
- [ ] Confirm homework due date
- [ ] Verify you can access all 5 notebooks
- [ ] Note instructor's office hours for help
- [ ] Download or bookmark v1_lecture_notes.md

---

### After Class (Homework & Practice)

**Immediate (Same Day or Next Day):**
- [ ] Re-run Notebooks 01-02 yourself (don't rely on class demo)
- [ ] Complete Notebook 03: Bounding Box Visualization
- [ ] Complete Notebook 04: Evaluation Metrics (mAP)
- [ ] Complete Notebook 05: Sliding Windows
- [ ] Answer all "TODO Exercise" questions in each notebook

**Deeper Learning (2-3 Days After Class):**
- [ ] Read v1_comprehensive_lecture_notes.md (18 pages)
- [ ] Practice IoU calculation by hand (5+ examples)
- [ ] Create visualization of your own image with bounding boxes
- [ ] Experiment: modify code in notebooks, see what happens
- [ ] Optional: Attempt one assignment idea (e.g., IoU Calculator)

**Submission (If Required):**
- [ ] Take screenshots of key outputs from each notebook
- [ ] Answer written questions (typed, not handwritten)
- [ ] Create 1-paragraph reflection: "What was most difficult/surprising?"
- [ ] Submit via LMS or email before deadline
- [ ] Keep local copies for your records

---

### Before FT2 Exam (Review)

**3-4 Days Before Exam:**
- [ ] Re-read this README (skip setup sections, focus on concepts)
- [ ] Skim all 5 notebooks (code + markdown cells)
- [ ] Review IoU calculation (Notebook 02)
- [ ] Review mAP definition (Notebook 04)

**2 Days Before Exam:**
- [ ] Memorize IoU formula: `IoU = Intersection / Union`
- [ ] Practice calculating IoU for 3-4 box pairs by hand
- [ ] Review coordinate systems: XYWH vs XYXY
- [ ] Understand: classification vs localization vs detection

**1 Day Before Exam:**
- [ ] Create 1-page cheat sheet (formulas, definitions, key concepts)
- [ ] Review practice questions in "Assessment Alignment" section
- [ ] Read Notebook 01 carefully (definitions and applications)
- [ ] Get good sleep (don't cram midnight before exam!)

**Morning of Exam:**
- [ ] Quick review of cheat sheet (10 minutes)
- [ ] Recall IoU interpretation: 0.0 (no overlap) to 1.0 (perfect)
- [ ] Remember: mAP = mean Average Precision (THE detection metric)
- [ ] Stay calm - you've prepared well!

---

### After FT2 Exam (Next Steps)

**Immediately After Exam:**
- [ ] Reflect: Which Week 13 concepts were tested?
- [ ] Note: What should I review before final exam?
- [ ] Relax and take a break!

**Preparing for Week 14 (YOLO):**
- [ ] Review Week 13 foundations (you'll need them for YOLO)
- [ ] Read ahead: Search "YOLO object detection tutorial" on YouTube
- [ ] Optional: Install ultralytics for Tutorial T13
- [ ] Be ready for hands-on YOLO implementation

**Long-Term Learning:**
- [ ] Keep all Week 13 notebooks (you'll reference them in future projects)
- [ ] Bookmark additional resources (papers, blogs, datasets)
- [ ] Consider a personal project: build a simple object detector
- [ ] Connect Week 13 concepts to real-world CV applications you care about

---

## Success Metrics

### How to Know You've Mastered Week 13

Object detection foundations are mastered when you can confidently complete these challenges:

---

### **Challenge 1: The 30-Second Explanation**

**Scenario:** A friend asks, "What's object detection?"

**Your Answer Should Include:**
- Definition: Finding all objects + their locations in an image
- Output format: List of [label, bounding box, confidence]
- Difference from classification: Multiple objects vs single label
- Real-world example: Self-driving cars detecting pedestrians

**Success Criteria:**
- ✅ Explanation is clear and accurate
- ✅ Non-technical person can understand
- ✅ Takes 20-40 seconds to deliver
- ✅ Includes at least one concrete example

---

### **Challenge 2: The IoU Speed Test**

**Scenario:** Given two bounding boxes on paper, calculate IoU by hand

**Example:**
```
Box 1: [100, 100, 200, 150]  # XYWH format
Box 2: [150, 125, 200, 150]

Calculate IoU in <3 minutes without calculator
```

**Success Criteria:**
- ✅ Convert to XYXY correctly: Box1 → [100, 100, 300, 250], Box2 → [150, 125, 350, 275]
- ✅ Find intersection coordinates: [150, 125, 300, 250]
- ✅ Calculate areas: Intersection=18,750, Union=71,250
- ✅ Final IoU: 0.263 (approximately 0.26)
- ✅ Completed in <3 minutes

---

### **Challenge 3: The Implementation Test**

**Scenario:** Implement `calculate_iou()` from memory in a blank Python file

**No Internet, No Notes - Just Your Memory:**

**Success Criteria:**
- ✅ Function signature: `def calculate_iou(box1, box2)`
- ✅ Handles XYWH input format
- ✅ Correctly computes intersection area (with zero-check)
- ✅ Correctly computes union area (Area1 + Area2 - Intersection)
- ✅ Returns IoU as float in [0.0, 1.0]
- ✅ Code runs without syntax errors
- ✅ Passes test: `calculate_iou([0,0,10,10], [0,0,10,10]) == 1.0`
- ✅ Completed in <10 minutes

---

### **Challenge 4: The Visualization Challenge**

**Scenario:** Given an image and detection results, create professional visualization

**Task:**
```python
# Given:
image = cv2.imread('test.jpg')
detections = [
    {'class': 'cat', 'bbox': [100, 50, 200, 150], 'confidence': 0.95},
    {'class': 'dog', 'bbox': [350, 200, 180, 220], 'confidence': 0.87}
]

# Create YOLO-style visualization with boxes and labels
```

**Success Criteria:**
- ✅ Both bounding boxes drawn correctly
- ✅ Labels include class name + confidence score
- ✅ Text has background for readability
- ✅ Color-coded boxes (different color per class)
- ✅ Professional appearance (like YOLO output)
- ✅ Code runs without errors
- ✅ Completed in <15 minutes

---

### **Challenge 5: The Metric Interpretation Challenge**

**Scenario:** Analyze detection results and explain performance

**Given Results:**
```
Model A:
- Class "cat": AP = 0.82
- Class "dog": AP = 0.76
- Class "person": AP = 0.91
- mAP@0.5 = 0.83

Model B:
- Class "cat": AP = 0.78
- Class "dog": AP = 0.88
- Class "person": AP = 0.85
- mAP@0.5 = 0.84
```

**Questions:**
1. Which model is better overall?
2. Which model is better for detecting dogs?
3. What does mAP@0.5 = 0.83 mean?
4. If the application requires very precise localization, how should you re-evaluate?

**Success Criteria:**
- ✅ Q1: Model B (higher mAP)
- ✅ Q2: Model B (0.88 vs 0.76 AP for dog class)
- ✅ Q3: "On average, the model achieves 83% precision/recall at IoU threshold 0.5"
- ✅ Q4: "Re-calculate mAP@0.75 to require tighter boxes"
- ✅ Explanations are clear and accurate

---

### **Challenge 6: The Debugging Challenge**

**Scenario:** Student's IoU function returns IoU=1.5 (impossible!)

**Their Code:**
```python
def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    inter_w = min(x1+w1, x2+w2) - max(x1, x2)
    inter_h = min(y1+h1, y2+h2) - max(y1, y2)
    intersection = inter_w * inter_h

    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2  # BUG HERE!

    return intersection / union
```

**Task:** Identify the bug and explain the fix

**Success Criteria:**
- ✅ Identified bug: Union should be `area1 + area2 - intersection`
- ✅ Explanation: "Without subtracting intersection, we count the overlap twice"
- ✅ Also noted missing check: "Should check if inter_w or inter_h is negative (no overlap)"
- ✅ Completed in <5 minutes

---

### **Challenge 7: The Application Challenge**

**Scenario:** Design object detection solution for a real-world problem

**Problem:**
```
A hospital wants to count white blood cells in microscopy images
to detect infections. Each image has 50-200 cells. Cells are small
(20-40 pixels) and sometimes overlap. Budget is limited.
```

**Task:** Recommend an approach using Week 13 knowledge

**Success Criteria:**
- ✅ Recognize this is object detection (not classification)
- ✅ Suggest: Use pre-trained detector with transfer learning
- ✅ Address small objects: "Need high IoU threshold (0.75) and multi-scale detection"
- ✅ Address overlap: "NMS will be critical to avoid duplicate counts"
- ✅ Address budget: "Start with YOLO (faster, cheaper) before trying R-CNN"
- ✅ Mention evaluation: "Use mAP to measure performance, precision to minimize false positives"

---

### **Self-Assessment Scorecard**

Rate yourself honestly after completing Week 13:

| Challenge | Status | Notes |
|-----------|--------|-------|
| 1. 30-Second Explanation | ☐ Mastered ☐ Partial ☐ Need Practice | |
| 2. IoU Speed Test | ☐ <3 min ☐ 3-5 min ☐ >5 min | |
| 3. Implementation Test | ☐ <10 min ☐ 10-15 min ☐ >15 min | |
| 4. Visualization Challenge | ☐ <15 min ☐ 15-20 min ☐ >20 min | |
| 5. Metric Interpretation | ☐ All correct ☐ 3/4 correct ☐ <3/4 | |
| 6. Debugging Challenge | ☐ <5 min ☐ 5-10 min ☐ >10 min | |
| 7. Application Challenge | ☐ Comprehensive ☐ Partial ☐ Incomplete | |

**Scoring:**
- **6-7 "Mastered"**: You're ready for Week 14 YOLO! Excellent work.
- **4-5 "Mastered"**: Good foundation. Review weak areas before Week 14.
- **2-3 "Mastered"**: Need more practice. Revisit notebooks and attempt challenges again.
- **0-1 "Mastered"**: Seek help from instructor or study group. Don't move to Week 14 yet.

---

### **Final Success Indicator**

**You've truly mastered Week 13 when:**

You can confidently say:
> "I understand how object detectors work under the hood. I know IoU is the core metric for matching predictions to ground truth. I can calculate mAP to evaluate any detector. I understand why YOLO and R-CNN are revolutionary (they solve the computational nightmare of sliding windows). I'm ready to implement and fine-tune production object detection systems."

**If you can say this and mean it - congratulations! You've built a solid foundation for advanced computer vision.** 🎉

---

## Support

### Getting Help During Week 13

---

### **Instructor Contact**

**Prof. Ramesh Babu**
- **Office:** [Room Number TBD]
- **Office Hours:** [Days/Times TBD]
- **Email:** [Email TBD]
- **Response Time:** Within 24-48 hours (weekdays)

**Best Times to Ask Questions:**
- During class sessions (real-time clarification)
- Immediately after class (quick questions)
- Office hours (in-depth discussion)

---

### **Online Resources**

**Course Materials:**
- **This README:** Primary reference for Week 13
- **Lecture Notes:** `v1_comprehensive_lecture_notes.md` (18 pages, detailed theory)
- **Notebooks:** 5 interactive Colab notebooks with code and explanations
- **Requirements:** `requirements.txt` for local setup

**External Help:**
- **Stack Overflow:** Tag questions with `object-detection`, `python`, `numpy`
- **Google Colab Forum:** https://discourse.colab.research.google.com/
- **Reddit:** r/learnprogramming, r/computervision

---

### **Peer Support**

**Study Groups (Recommended):**
- Form groups of 3-4 students
- Meet 1-2 times during Week 13
- Work through notebooks together
- Explain concepts to each other (best way to learn!)

**Collaboration Guidelines:**
- ✅ Discuss concepts and approaches
- ✅ Debug each other's code together
- ✅ Share resources and explanations
- ✅ Work through exercises collaboratively
- ⚠️ Write your own code (don't just copy)
- ⚠️ Submit your own work (if graded assignment)

---

### **Common Questions & Quick Answers**

**Q: Which notebook is most important?**
- **A:** Notebook 02 (IoU Calculation). IoU is the foundation for everything else. If you master only one notebook, make it this one.

**Q: Do I need a GPU for Week 13?**
- **A:** No! Week 13 uses only NumPy (no neural networks). GPU will be needed in Weeks 14-15 (YOLO/R-CNN), but Google Colab provides free GPU.

**Q: How long will Week 13 take?**
- **A:** 60-70 minutes for all 5 notebooks if you run them sequentially. Add 30-60 minutes for deeper understanding (reading lecture notes, practice problems).

**Q: Can I use my own dataset?**
- **A:** Yes! After completing Notebooks 01-05, try applying concepts to your own images. Great way to practice.

**Q: What if I don't understand mAP calculation?**
- **A:** Focus on the CONCEPT first (mAP measures detection quality across all classes). You don't need to memorize the full algorithm - understanding when and why to use it is more important.

**Q: Is Week 13 tested in FT2?**
- **A:** Yes, but lightly (5-8 MCQs + possibly 1 SAQ). Focus on IoU, object detection definition, and mAP concept. See "Assessment Alignment" section.

**Q: Can I skip Week 13 and go straight to YOLO?**
- **A:** Not recommended! YOLO uses IoU for anchor box matching, NMS for duplicate removal, and mAP for evaluation. Without Week 13, YOLO will be confusing.

**Q: My IoU function returns negative values. What's wrong?**
- **A:** Check "Troubleshooting" section, Issue #6. Common causes: wrong coordinate format, missing zero-check for non-overlapping boxes.

**Q: How do I prepare for FT2 exam regarding Week 13?**
- **A:** Memorize IoU formula, practice calculating IoU by hand (3-4 examples), understand mAP concept, review classification vs detection difference. See "Checklist Before FT2" section.

---

### **Troubleshooting Flowchart**

```
Problem?
    │
    ├─ Code Error?
    │   ├─ Import Error → See Troubleshooting Issue #1 (OpenCV)
    │   ├─ Plot Not Showing → See Issue #2 (Matplotlib inline)
    │   ├─ Runtime Disconnect → See Issue #3 (Colab reconnect)
    │   └─ Memory Error → See Issue #5 (Kernel restart)
    │
    ├─ Wrong Results?
    │   ├─ IoU Negative/Wrong → See Issue #6 (Coordinate format)
    │   ├─ mAP = 0 or >1 → See Issue #9 (Sort by confidence)
    │   └─ Colors Wrong → See Issue #8 (BGR vs RGB)
    │
    ├─ Conceptual Confusion?
    │   ├─ What is IoU? → Re-read Notebook 02, watch IoU visualization
    │   ├─ What is mAP? → Read v1_lecture_notes.md Section 4
    │   ├─ Why no deep learning? → Check README "Overview" section
    │   └─ Still confused → Ask instructor or study group
    │
    └─ Slow Performance?
        ├─ NumPy Operations Slow → See Issue #4 (Vectorization)
        └─ Notebook 05 Very Slow → Reduce scales, increase stride
```

---

### **When to Ask for Help**

**Ask Immediately (Don't Struggle Alone):**
- ❗ Code errors you can't fix after 15 minutes of trying
- ❗ Fundamental concept confusion (what is object detection?)
- ❗ Can't run notebooks at all (setup issues)
- ❗ Completely lost after class demo

**Try to Solve First, Then Ask:**
- ⏳ Wrong output but code runs (debug first, ask if stuck >30 min)
- ⏳ Slow performance (try troubleshooting tips first)
- ⏳ Want deeper understanding (read lecture notes first, ask to clarify)

**Optional (For Curious Students):**
- 💡 "Why is IoU better than Euclidean distance for box matching?"
- 💡 "Can I implement YOLO from scratch using Week 13 concepts?"
- 💡 "How do I create my own dataset for object detection?"

---

### **Emergency Help (Exam Week)**

**If FT2 is in 1-2 days and you're behind on Week 13:**

**Priority 1 (Must Do - 30 min):**
1. Run Notebook 01: Understand classification vs detection (10 min)
2. Run Notebook 02: Master IoU calculation (15 min)
3. Memorize IoU formula and interpretation (5 min)

**Priority 2 (Should Do - 20 min):**
4. Skim Notebook 04: Understand what mAP measures (10 min)
5. Review "Assessment Alignment" section in README (10 min)

**Priority 3 (If Time - 15 min):**
6. Read v1_lecture_notes.md Part 3 (IoU section)
7. Practice IoU calculation by hand (3 examples)

**Then:** Focus on Week 10-12 content (CNNs, Transfer Learning) - higher weight in FT2!

---

**Remember: You're not alone in this learning journey. Don't hesitate to ask for help - that's what instructors, TAs, and peers are here for!** 🤝

---

**Last Updated:** November 2025
**Version:** 1.0
**Status:** ✅ Ready for Week 13 Class
**Next Update:** After Tutorial T13 is added (Notebook 06)

---

## Let's Get Started! 🚀

**You're now ready to begin Week 13!**

1. **Open Notebook 01** to understand the object detection problem
2. **Master Notebook 02** to learn IoU (the most critical concept)
3. **Progress through Notebooks 03-05** to build complete foundation
4. **Prepare for Weeks 14-15** where you'll implement YOLO and R-CNN

**Remember:** Week 13 is not just another week - it's the foundation that will make you a confident computer vision engineer. Take your time, understand deeply, and enjoy the journey!

**Happy Learning!** 📚✨

---

**Questions? Issues? Feedback?**
Contact: Prof. Ramesh Babu | Office: [TBD] | Email: [TBD]
