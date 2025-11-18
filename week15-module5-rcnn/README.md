# Week 15: R-CNN Family - Two-Stage Object Detection

## Course Information
- **Course**: 21CSE558T - Deep Neural Network Architectures
- **Module**: 5 - Object Detection (Part 3 - Final Week)
- **Week**: 15 (Nov 18-21, 2025) - **FINAL WEEK OF COURSE**
- **Credits**: 3 (2L + 1T + 0P)
- **Institution**: SRM University, M.Tech Program

## Module 5 Complete Coverage

### Week 13: Object Detection Foundations
- Bounding boxes, IOU, mAP evaluation metrics
- Classical methods (sliding windows, selective search)
- Tutorial T13: Pre-trained YOLO inference

### Week 14: YOLO & SSD - Real-Time Detection
- Single-shot detection paradigm
- YOLO architecture and evolution (v1 → v8)
- Tutorial T14: Training custom YOLOv8 model

### Week 15: R-CNN Family - Two-Stage Detection (THIS WEEK)
- Two-stage detection paradigm
- R-CNN evolution (R-CNN → Fast → Faster)
- Tutorial T15: Pre-trained Faster R-CNN inference
- **Final course review and exam preparation**

---

## Table of Contents

1. [Overview](#overview)
2. [Learning Objectives](#learning-objectives)
3. [Prerequisites](#prerequisites)
4. [Key Concepts](#key-concepts)
5. [Notebooks Overview](#notebooks-overview)
6. [Installation Instructions](#installation-instructions)
7. [Learning Path](#learning-path)
8. [Detailed Notebook Descriptions](#detailed-notebook-descriptions)
9. [Technical Requirements](#technical-requirements)
10. [Troubleshooting](#troubleshooting)
11. [Assessment and Practice](#assessment-practice)
12. [Additional Resources](#additional-resources)
13. [Comparison with Week 14](#comparison-week14)
14. [Final Exam Preparation](#final-exam-prep)
15. [FAQs](#faqs)
16. [Project Ideas](#project-ideas)
17. [Credits and References](#credits-references)
18. [Next Steps](#next-steps)
19. [Contact Information](#contact-information)

---

## Overview {#overview}

### What You'll Learn This Week

Week 15 concludes Module 5 (Object Detection) and the entire course by covering **two-stage object detectors**, specifically the **R-CNN family**. This week focuses on understanding the evolution from R-CNN to Faster R-CNN, the Region Proposal Network (RPN), and comparing two-stage vs single-shot detection approaches.

**Key Topics:**
1. **R-CNN Evolution**: R-CNN (2014) → Fast R-CNN (2015) → Faster R-CNN (2015)
2. **Two-Stage Paradigm**: Region proposals + classification (vs YOLO's single-shot)
3. **Region Proposal Networks (RPN)**: Learned proposals using anchor boxes
4. **ROI Pooling/Align**: Extracting fixed-size features from variable-size regions
5. **Comparison**: When to use Faster R-CNN vs YOLO/SSD
6. **Practical Skills**: Using pre-trained Faster R-CNN for inference

### Week 15 Focus

Unlike Week 14 (which emphasized training YOLOv8), Week 15 focuses on:
- **Understanding architecture** over training from scratch
- **Using pre-trained models** (torchvision's Faster R-CNN)
- **Inference and evaluation** rather than custom training
- **Theoretical comparison** of detection paradigms
- **Final course review** and exam preparation

**Why This Approach?**
- Faster R-CNN training requires more resources (GPU, time)
- Pre-trained models work exceptionally well for most tasks
- Academic understanding is priority in final week
- Prepares students for final examination

---

## Learning Objectives {#learning-objectives}

### By the End of Week 15, You Will Be Able To:

**Conceptual Understanding:**
1. Explain the evolution of R-CNN family and key improvements
2. Describe the two-stage detection paradigm
3. Understand how Region Proposal Networks (RPN) work
4. Compare ROI Pooling and ROI Align
5. Analyze trade-offs between two-stage and single-shot detectors

**Technical Skills:**
6. Use pre-trained Faster R-CNN models for object detection
7. Implement inference pipelines using torchvision
8. Visualize detection results with bounding boxes
9. Evaluate detector performance using mAP
10. Choose appropriate detector for specific applications

**Practical Applications:**
11. Apply Faster R-CNN to real-world images
12. Compare YOLO vs Faster R-CNN on same dataset
13. Understand when to use which detection approach
14. Interpret detection results and confidence scores

**Course Completion:**
15. Integrate knowledge from all 5 modules
16. Prepare for final examination
17. Understand deep learning for computer vision holistically

---

## Prerequisites {#prerequisites}

### Required Knowledge

**From Previous Weeks:**
- **Week 13**: IOU calculation, mAP evaluation, bounding box formats
- **Week 14**: YOLO architecture, single-shot detection, NMS
- **Weeks 10-12**: CNNs, transfer learning, pre-trained models

**Technical Prerequisites:**
- Python programming (intermediate)
- PyTorch basics (tensors, models, inference)
- Object detection fundamentals (classification + localization)
- Understanding of CNNs and feature maps

### Required Software

**Core Libraries:**
- Python 3.8+ (3.10 recommended)
- PyTorch 2.0+
- torchvision 0.15+ (includes Faster R-CNN)
- OpenCV 4.6+
- Matplotlib, NumPy, Pillow

**Optional:**
- TensorFlow 2.10+ (for alternative implementations)
- Detectron2 (advanced R-CNN framework, optional)

**See `requirements.txt` for complete list**

### Hardware Requirements

**Minimum:**
- CPU: Any modern processor (inference possible on CPU)
- RAM: 8 GB
- Disk: 3 GB for models and notebooks

**Recommended:**
- GPU: NVIDIA with 4+ GB VRAM (T4, RTX 3060, etc.)
- RAM: 16 GB
- Disk: 5 GB

**Best for Students:**
- **Google Colab** (free tier with T4 GPU)
- Most libraries pre-installed
- No local setup required

---

## Key Concepts {#key-concepts}

### 1. Two-Stage Detection Paradigm

**Definition:**
Object detection in two sequential stages:
1. **Stage 1**: Generate region proposals (where objects might be)
2. **Stage 2**: Classify proposals and refine bounding boxes

**Contrast with Single-Shot (YOLO/SSD):**
- **Single-Shot**: One forward pass → predictions
- **Two-Stage**: Two forward passes → proposals → predictions

**Advantages:**
- Higher accuracy (especially for small objects)
- Better localization precision
- Separation of concerns (localization vs classification)

**Disadvantages:**
- Slower inference (5-7 FPS vs 30-120 FPS for YOLO)
- More complex architecture
- Harder to optimize end-to-end

### 2. R-CNN Family Evolution

**R-CNN (2014):**
- First deep learning object detector
- Selective search for proposals (~2,000 per image)
- CNN features + SVM classifier
- Speed: 47 seconds/image
- Accuracy: 53.7% mAP (PASCAL VOC)

**Fast R-CNN (2015):**
- Shared CNN computation (run once per image)
- ROI pooling for region features
- Multi-task loss (classification + regression)
- Speed: 2.3 seconds/image (20× faster)
- Accuracy: 66.9% mAP

**Faster R-CNN (2015):**
- Region Proposal Network (RPN) replaces selective search
- End-to-end trainable
- Anchor boxes for multi-scale detection
- Speed: 0.2 seconds/image (5 FPS)
- Accuracy: 73.2% mAP

**Timeline:**
```
2014: R-CNN → 47 sec/img → 53.7% mAP
2015: Fast R-CNN → 2.3 sec/img → 66.9% mAP
2015: Faster R-CNN → 0.2 sec/img → 73.2% mAP
2017: Mask R-CNN → Instance segmentation
2020+: Modern variants (Cascade R-CNN, etc.)
```

### 3. Region Proposal Network (RPN)

**Purpose:**
Generate region proposals using a learned neural network (instead of selective search)

**Architecture:**
1. Slide 3×3 window over CNN feature map
2. For each position, generate k anchor boxes (typically 9)
3. Predict objectness (object vs background) for each anchor
4. Predict box offsets (dx, dy, dw, dh) for each anchor

**Anchor Boxes:**
- 3 scales: 128², 256², 512² pixels
- 3 aspect ratios: 1:1, 1:2, 2:1
- Total: 3×3 = 9 anchors per position

**Output:**
- ~17,000 anchors generated
- Top 300 proposals selected (after NMS)
- Fed to detection head for classification

**Training:**
- Multi-task loss: objectness + box regression
- Positive: IOU ≥ 0.7 with ground truth
- Negative: IOU < 0.3 with ground truth

### 4. ROI Pooling vs ROI Align

**ROI Pooling (Fast R-CNN):**
- Converts variable-size ROIs to fixed size (e.g., 7×7)
- Quantizes coordinates to integer values
- Max pooling within each grid cell
- Issue: Quantization causes small misalignments

**ROI Align (Mask R-CNN):**
- No quantization (uses floating-point coordinates)
- Bilinear interpolation for exact sampling
- Better accuracy (especially for segmentation)
- Negligible speed penalty

**When to Use:**
- Detection: ROI Pooling sufficient
- Segmentation: ROI Align essential

### 5. Faster R-CNN Architecture

**Complete Pipeline:**

```
Input Image (any size)
    ↓
CNN Backbone (ResNet-50 with FPN)
    ↓
Feature Maps (multi-scale)
    ↓         ↓
    RPN       ↓
    ↓         ↓
Proposals    ↓
    ↓         ↓
ROI Align   ←┘
    ↓
Detection Head (FC layers)
    ↓
    ├─→ Classification (C classes)
    └─→ Box Regression (4×C)
    ↓
NMS → Final Detections
```

**Key Components:**
1. **Backbone**: ResNet-50 with Feature Pyramid Network (FPN)
2. **RPN**: Region proposals with anchor boxes
3. **ROI Align**: Fixed-size feature extraction
4. **Detection Head**: Classification + box refinement
5. **NMS**: Remove duplicate detections

---

## Notebooks Overview {#notebooks-overview}

### Week 15 Notebooks (5 Total)

| Notebook | Title | Duration | Difficulty | GPU |
|----------|-------|----------|------------|-----|
| 01 | R-CNN Family Evolution | ~10 min | Beginner | Optional |
| 02 | Region Proposals & Selective Search | ~15 min | Intermediate | Optional |
| 03 | **Faster R-CNN Pre-trained (Tutorial T15)** | ~20 min | Intermediate | Recommended |
| 04 | YOLO vs R-CNN Benchmark | ~15 min | Intermediate | Recommended |
| 05 | Choosing the Right Detector | ~15 min | Beginner | Optional |

**Total Time: ~75 minutes (1.25 hours)**

### Recommended Order

**For Beginners:**
1. Notebook 01 (Evolution overview)
2. Notebook 03 (Tutorial T15 - hands-on experience)
3. Notebook 05 (Decision framework)
4. Notebook 04 (Benchmarking)
5. Notebook 02 (Classical methods)

**For Advanced Students:**
1. All notebooks in order (01 → 05)
2. Experiment with different images in Notebook 03
3. Try fine-tuning in Notebook 03 (optional)

**For Final Exam Prep:**
- Focus on Notebooks 01, 03, 04, 05
- Practice questions in lecture notes
- Compare with Week 14 (YOLO)

---

## Installation Instructions {#installation-instructions}

### Option 1: Google Colab (Recommended)

**Advantages:**
- No local installation required
- Free GPU (T4) available
- Most libraries pre-installed
- Easy sharing and collaboration

**Steps:**
1. Upload notebooks to Google Drive
2. Open with Google Colab
3. Enable GPU: Runtime → Change runtime type → GPU
4. Install missing packages:
   ```python
   !pip install ultralytics  # For Notebook 04 (YOLO comparison)
   ```

### Option 2: Local Installation

**Step 1: Create Virtual Environment**
```bash
# Using venv
python3 -m venv week15-env
source week15-env/bin/activate  # Linux/Mac
# OR
week15-env\Scripts\activate  # Windows

# Using conda
conda create -n week15 python=3.10
conda activate week15
```

**Step 2: Install Dependencies**
```bash
# Option A: Install all dependencies
pip install -r requirements.txt

# Option B: Minimal install (inference only)
pip install torch torchvision opencv-python matplotlib pillow

# Option C: Install individually
pip install torch>=2.0.0
pip install torchvision>=0.15.0
pip install opencv-python>=4.6.0
pip install matplotlib>=3.5.0
pip install pillow>=9.0.0
```

**Step 3: Verify Installation**
```python
import torch
import torchvision
print(f"PyTorch: {torch.__version__}")
print(f"torchvision: {torchvision.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

# Load Faster R-CNN model (test)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
print("✓ Faster R-CNN model loaded successfully!")
```

### Option 3: Docker (Advanced)

```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
RUN pip install torchvision opencv-python matplotlib pillow jupyter
WORKDIR /workspace
```

```bash
docker build -t week15-rcnn .
docker run --gpus all -p 8888:8888 -v $(pwd):/workspace week15-rcnn jupyter notebook --allow-root
```

### Pre-trained Model Downloads

**Automatic Download:**
Models download automatically on first use (~160 MB for ResNet-50 variant)

**Models Available in torchvision:**
1. **fasterrcnn_resnet50_fpn**: 160 MB, 37.0% mAP (COCO)
2. **fasterrcnn_mobilenet_v3_large_fpn**: 19 MB, 32.8% mAP (COCO)
3. **maskrcnn_resnet50_fpn**: 170 MB (includes segmentation)

**Manual Download (Optional):**
- Models cached in: `~/.cache/torch/hub/checkpoints/`
- URL: https://download.pytorch.org/models/

---

## Learning Path {#learning-path}

### Beginner Path (75 minutes)

**Session 1: Understanding R-CNN Evolution (25 min)**
1. Read lecture notes sections 1-4
2. Complete Notebook 01 (R-CNN Evolution)
3. Understand two-stage paradigm

**Session 2: Hands-On Faster R-CNN (20 min)**
1. Complete Notebook 03 (Tutorial T15)
2. Run inference on sample images
3. Visualize detections

**Session 3: Comparison and Decision (30 min)**
1. Complete Notebook 05 (Decision framework)
2. Complete Notebook 04 (YOLO vs R-CNN)
3. Review practice questions

### Intermediate Path (2 hours)

**Session 1: Theory Deep Dive (45 min)**
1. Read entire lecture notes (20 pages)
2. Complete Notebook 01 (Evolution)
3. Complete Notebook 02 (Region Proposals)

**Session 2: Practical Implementation (45 min)**
1. Complete Notebook 03 (Tutorial T15)
2. Experiment with different images
3. Analyze detection results

**Session 3: Benchmarking and Comparison (30 min)**
1. Complete Notebook 04 (YOLO vs R-CNN)
2. Complete Notebook 05 (Decision framework)
3. Synthesize learnings

### Advanced Path (4+ hours)

**All intermediate content PLUS:**
1. Implement custom fine-tuning (Notebook 03 optional section)
2. Experiment with Detectron2 (optional framework)
3. Compare Mask R-CNN for instance segmentation
4. Read original research papers
5. Implement evaluation metrics from scratch

### Final Exam Preparation Path (3 hours)

**Week 13 Review (60 min):**
- IOU, mAP, bounding boxes
- Evaluation metrics
- Classical methods

**Week 14 Review (60 min):**
- YOLO architecture
- Single-shot detection
- Real-time considerations

**Week 15 Review (60 min):**
- R-CNN evolution
- Two-stage paradigm
- Comparison framework

**Practice Questions:**
- Complete all practice questions in lecture notes
- Focus on comparative questions (YOLO vs R-CNN)

---

## Detailed Notebook Descriptions {#detailed-notebook-descriptions}

### Notebook 01: R-CNN Family Evolution

**File**: `notebooks/01_rcnn_family_evolution.ipynb`

**Duration**: ~10 minutes
**Difficulty**: Beginner
**GPU**: Optional

**Topics Covered:**
1. R-CNN (2014) architecture and limitations
2. Fast R-CNN (2015) improvements
3. Faster R-CNN (2015) RPN innovation
4. Timeline visualization
5. Performance comparisons

**Learning Outcomes:**
- Understand evolution of R-CNN family
- Identify key improvements in each version
- Visualize speed vs accuracy trade-offs

**Key Code Examples:**
- Timeline visualization
- Architecture diagrams
- Speed comparison charts

**Practice Exercises:**
1. Calculate speedup from R-CNN to Faster R-CNN
2. Explain why Fast R-CNN is faster than R-CNN
3. Identify the bottleneck removed by Faster R-CNN

---

### Notebook 02: Region Proposals & Selective Search

**File**: `notebooks/02_region_proposals_selective_search.ipynb`

**Duration**: ~15 minutes
**Difficulty**: Intermediate
**GPU**: Optional

**Topics Covered:**
1. Selective search algorithm
2. Region proposal generation
3. Why RPN replaced selective search
4. Anchor boxes concept

**Learning Outcomes:**
- Understand classical region proposal methods
- Implement selective search basics
- Compare selective search vs RPN

**Key Code Examples:**
- Selective search implementation (OpenCV)
- Visualizing region proposals
- Anchor box generation

**Practice Exercises:**
1. Generate region proposals using selective search
2. Count proposals for different images
3. Compare proposal quality (IOU with ground truth)

---

### Notebook 03: Faster R-CNN Pre-trained (Tutorial T15)

**File**: `notebooks/03_faster_rcnn_pretrained.ipynb`

**Duration**: ~20 minutes
**Difficulty**: Intermediate
**GPU**: Recommended

**Topics Covered:**
1. Loading pre-trained Faster R-CNN (torchvision)
2. Inference on custom images
3. Visualizing detections
4. Confidence threshold tuning
5. Optional: Fine-tuning on custom dataset

**Learning Outcomes:**
- Use torchvision's Faster R-CNN models
- Implement complete inference pipeline
- Visualize and interpret results

**Key Code Examples:**
```python
# Load pre-trained model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# Inference
with torch.no_grad():
    predictions = model([image_tensor])

# Results: boxes, labels, scores
boxes = predictions[0]['boxes']
labels = predictions[0]['labels']
scores = predictions[0]['scores']
```

**Practice Exercises:**
1. Run inference on 5 different images
2. Experiment with confidence thresholds (0.5, 0.7, 0.9)
3. Compare ResNet-50 vs MobileNet variants
4. (Optional) Fine-tune on custom dataset

**This is Tutorial T15 - Complete for Credit**

---

### Notebook 04: YOLO vs R-CNN Benchmark

**File**: `notebooks/04_yolo_vs_rcnn_benchmark.ipynb`

**Duration**: ~15 minutes
**Difficulty**: Intermediate
**GPU**: Recommended

**Topics Covered:**
1. Direct comparison: YOLOv8 vs Faster R-CNN
2. Speed benchmarking (FPS)
3. Accuracy comparison (mAP)
4. Qualitative analysis (visual results)

**Learning Outcomes:**
- Benchmark two detection paradigms
- Understand speed vs accuracy trade-offs
- Decide which detector for specific tasks

**Key Code Examples:**
- Timing inference (both models)
- Side-by-side visualization
- Quantitative comparison tables

**Practice Exercises:**
1. Run both models on same 10 images
2. Compare inference times
3. Identify cases where Faster R-CNN performs better
4. Identify cases where YOLO performs better

---

### Notebook 05: Choosing the Right Detector

**File**: `notebooks/05_choosing_detector_decision_tree.ipynb`

**Duration**: ~15 minutes
**Difficulty**: Beginner
**GPU**: Optional

**Topics Covered:**
1. Decision framework for detector selection
2. Application scenarios (autonomous vehicles, surveillance, etc.)
3. Hardware considerations
4. Module 5 complete review
5. Final exam preparation

**Learning Outcomes:**
- Choose appropriate detector for applications
- Understand trade-offs in real-world deployments
- Synthesize learnings from Weeks 13-15

**Key Content:**
- Decision tree flowchart
- Application case studies
- Performance tables
- Final exam tips

**Practice Exercises:**
1. Choose detector for 5 different scenarios
2. Justify choices with technical reasoning
3. Create custom decision matrix

**Final Course Review:**
- Modules 1-5 integration
- Key concepts summary
- Exam preparation checklist

---

## Technical Requirements {#technical-requirements}

### Software Versions

**Tested Configurations:**

| Component | Minimum | Recommended | Tested |
|-----------|---------|-------------|--------|
| Python | 3.8 | 3.10 | 3.10.12 |
| PyTorch | 2.0.0 | 2.1.0+ | 2.1.0 |
| torchvision | 0.15.0 | 0.16.0+ | 0.16.0 |
| OpenCV | 4.6.0 | 4.8.0+ | 4.8.1 |
| NumPy | 1.21.0 | 1.24.0+ | 1.24.3 |
| Matplotlib | 3.5.0 | 3.7.0+ | 3.7.2 |

### Hardware Configurations

**Minimum (CPU Only):**
- Processor: Any modern CPU (Intel i5, AMD Ryzen 5, etc.)
- RAM: 8 GB
- Storage: 3 GB free
- OS: Windows 10, macOS 10.15+, Ubuntu 20.04+
- **Performance**: 2-5 seconds per image (inference)

**Recommended (GPU):**
- GPU: NVIDIA with 4+ GB VRAM (GTX 1660, RTX 3060, T4)
- RAM: 16 GB
- Storage: 5 GB free
- CUDA: 11.7+
- **Performance**: 0.15-0.2 seconds per image (inference)

**Optimal (for fine-tuning):**
- GPU: NVIDIA with 8+ GB VRAM (RTX 3070, A100)
- RAM: 32 GB
- Storage: 10 GB free
- **Performance**: Training possible (optional in Week 15)

### Google Colab Specifications

**Free Tier:**
- GPU: NVIDIA T4 (16 GB VRAM)
- RAM: 12-15 GB
- Storage: 100 GB (temporary)
- Session: 12 hours max
- **Sufficient for all Week 15 notebooks**

**Pro/Pro+ (Optional):**
- Better GPUs (V100, A100)
- Longer sessions (24 hours)
- Priority access
- **Not required for this course**

---

## Troubleshooting {#troubleshooting}

### Common Issues and Solutions

#### Issue 1: Model Download Fails

**Symptoms:**
```
RuntimeError: Error downloading model weights
```

**Solutions:**
1. Check internet connection
2. Manually download from https://download.pytorch.org/models/
3. Place in `~/.cache/torch/hub/checkpoints/`
4. Retry loading model

#### Issue 2: CUDA Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Process one image at a time (batch size = 1)
2. Use smaller model: `fasterrcnn_mobilenet_v3_large_fpn`
3. Reduce image resolution before inference
4. Clear GPU cache: `torch.cuda.empty_cache()`
5. Use CPU instead: `model.to('cpu')`

#### Issue 3: Slow Inference on CPU

**Symptoms:**
- Inference takes 2-5 seconds per image

**Solutions:**
1. Enable GPU in Colab (Runtime → Change runtime type)
2. Use MobileNet variant (faster on CPU)
3. Reduce image resolution
4. Process in batches (if memory allows)

#### Issue 4: torchvision Model Not Found

**Symptoms:**
```
AttributeError: module 'torchvision.models.detection' has no attribute 'fasterrcnn_resnet50_fpn'
```

**Solutions:**
1. Upgrade torchvision: `pip install --upgrade torchvision`
2. Check version: `torchvision.__version__` (should be 0.15+)
3. Reinstall: `pip uninstall torchvision && pip install torchvision`

#### Issue 5: Matplotlib/OpenCV Display Issues

**Symptoms:**
- Images not displaying in Jupyter

**Solutions:**
1. Add: `%matplotlib inline` at notebook start
2. Use: `plt.show()` after plotting
3. For OpenCV: Convert BGR to RGB before displaying

#### Issue 6: Import Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'torch'
```

**Solutions:**
1. Verify installation: `pip list | grep torch`
2. Reinstall: `pip install torch torchvision`
3. Check virtual environment is activated
4. Restart Jupyter kernel

---

## Assessment and Practice {#assessment-practice}

### Tutorial T15: Faster R-CNN Hands-On

**Notebook**: `03_faster_rcnn_pretrained.ipynb`
**Marks**: Part of Tutorial assessment (15% of final grade)

**Requirements:**
1. Complete inference on at least 3 images
2. Visualize detections with bounding boxes
3. Experiment with confidence thresholds
4. Compare ResNet-50 vs MobileNet variants
5. Submit outputs and observations

**Submission:**
- Jupyter notebook with outputs
- Screenshot of detections
- Short report (200-300 words) on observations

### Practice Questions

**Provided in Lecture Notes (Section 10.2):**
- 15 practice questions covering:
  - Conceptual understanding (R-CNN evolution, RPN, two-stage paradigm)
  - Technical questions (calculations, training, architecture)
  - Implementation questions (PyTorch code, fine-tuning)

**Additional Exercises:**
1. Explain why Faster R-CNN is slower than YOLO
2. Calculate total anchors for 1024×768 image with VGG16
3. Compare ROI Pooling vs ROI Align
4. Implement NMS from scratch
5. Choose detector for 5 different applications

### Final Exam Topics

**Module 5 Coverage (Weeks 13-15):**
- Object detection fundamentals (Week 13)
- YOLO & SSD (Week 14)
- R-CNN family (Week 15)

**Expected Question Types:**
- Multiple choice (definitions, architectures)
- Short answer (explain RPN, compare detectors)
- Numerical problems (IOU, mAP, anchor calculations)
- Code snippets (inference, training setup)
- Application scenarios (choose detector for task)

**Preparation Tips:**
1. Complete all 15 notebooks (Weeks 13-15)
2. Practice all questions in lecture notes
3. Compare YOLO vs Faster R-CNN visually
4. Memorize key numbers (FPS, mAP benchmarks)
5. Understand trade-offs (speed vs accuracy)

---

## Additional Resources {#additional-resources}

### Research Papers

1. **R-CNN (2014)**
   Girshick et al., "Rich feature hierarchies for accurate object detection"
   https://arxiv.org/abs/1311.2524

2. **Fast R-CNN (2015)**
   Ross Girshick, "Fast R-CNN"
   https://arxiv.org/abs/1504.08083

3. **Faster R-CNN (2015)**
   Ren et al., "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"
   https://arxiv.org/abs/1506.01497

4. **Mask R-CNN (2017)**
   He et al., "Mask R-CNN"
   https://arxiv.org/abs/1703.06870

### Official Documentation

1. **PyTorch torchvision**
   https://pytorch.org/vision/stable/models/detection.html
   Pre-trained models and APIs

2. **Detectron2**
   https://github.com/facebookresearch/detectron2
   Advanced R-CNN implementations (optional)

3. **COCO Dataset**
   https://cocodataset.org/
   Standard benchmark dataset

### Video Tutorials

1. **Stanford CS231n**
   Lecture 11: Detection and Segmentation
   http://cs231n.stanford.edu/

2. **Two Minute Papers**
   "R-CNN Explained"
   https://youtube.com/twominutepapers

3. **PyTorch Official**
   "Object Detection with Faster R-CNN"
   https://pytorch.org/tutorials/

### Blogs and Articles

1. **Towards Data Science**
   "Understanding Faster R-CNN"
   Multiple in-depth articles

2. **PyImageSearch**
   "Object Detection with Faster R-CNN and PyTorch"
   Practical tutorials

### Datasets for Practice

1. **COCO**: 80 classes, 330K images
2. **PASCAL VOC**: 20 classes, classic benchmark
3. **Open Images**: 600 classes, large scale
4. **Custom**: Use your own images!

---

## Comparison with Week 14 {#comparison-week14}

### Week 14 vs Week 15: Key Differences

| Aspect | Week 14 (YOLO) | Week 15 (R-CNN) |
|--------|----------------|-----------------|
| **Paradigm** | Single-shot | Two-stage |
| **Speed** | 30-120 FPS | 5-7 FPS |
| **Accuracy** | Good (improving) | High (localization) |
| **Training Focus** | Custom training (Tutorial T14) | Pre-trained inference (Tutorial T15) |
| **Complexity** | Simpler architecture | More complex (RPN + detector) |
| **Real-Time** | Yes | Challenging |
| **GPU Requirement** | Essential for training | Optional for inference |
| **Use Cases** | Real-time, edge devices | Accuracy-critical, offline |

### Complementary Learning

**Week 14 Strengths:**
- Real-time detection
- Practical training experience
- Deployment to edge devices

**Week 15 Strengths:**
- Higher accuracy
- Better small object detection
- Academic/theoretical foundation

**Together, They Cover:**
- Full spectrum of object detection approaches
- Trade-offs in real-world applications
- Complete understanding of detection paradigms

---

## Final Exam Preparation {#final-exam-prep}

### Module 5 Complete Review Checklist

**Week 13: Foundations**
- [ ] IOU calculation (formula and implementation)
- [ ] mAP evaluation (precision, recall, AP)
- [ ] Bounding box formats (XYXY, XYWH, COCO)
- [ ] NMS (Non-Maximum Suppression)
- [ ] Classical methods (sliding windows, selective search)

**Week 14: YOLO & SSD**
- [ ] YOLO architecture (grid, anchors, predictions)
- [ ] YOLO evolution (v1 → v8)
- [ ] Single-shot detection paradigm
- [ ] Training custom YOLO models
- [ ] Real-time considerations (FPS, hardware)

**Week 15: R-CNN Family**
- [ ] R-CNN evolution (R-CNN → Fast → Faster)
- [ ] Two-stage detection paradigm
- [ ] RPN architecture and anchor boxes
- [ ] ROI Pooling vs ROI Align
- [ ] Comparison framework (YOLO vs Faster R-CNN)

### Final Exam Topics (Module 5)

**Conceptual Questions (30%):**
- Explain object detection problem
- Compare single-shot vs two-stage
- Describe RPN architecture
- Evolution of R-CNN family

**Numerical Problems (25%):**
- Calculate IOU for given boxes
- Compute mAP from precision-recall curve
- Count anchors for image size
- Box coordinate transformations

**Code Questions (25%):**
- Implement IOU function
- Load and use pre-trained Faster R-CNN
- Visualize detections
- NMS implementation

**Application Scenarios (20%):**
- Choose detector for given application
- Justify choice with technical reasoning
- Compare trade-offs
- Hardware considerations

### Study Strategy

**Week 1 (Now - Nov 18):**
- Complete all 5 Week 15 notebooks
- Read lecture notes thoroughly
- Complete practice questions

**Week 2 (Nov 18-21):**
- Review Weeks 13-14 materials
- Compare YOLO vs Faster R-CNN on same images
- Practice numerical calculations (IOU, mAP)

**Final Days:**
- Memorize key numbers (FPS, mAP benchmarks)
- Review all practice questions
- Synthesize learnings from all 3 weeks

---

## FAQs {#faqs}

### General Questions

**Q1: Do I need to train Faster R-CNN from scratch?**
A: No. Week 15 focuses on using pre-trained models. Training is optional and not required for assessment.

**Q2: Can I run notebooks on CPU?**
A: Yes, but inference will be slower (2-5 sec/image vs 0.15 sec on GPU). Use Google Colab for free GPU.

**Q3: How is Week 15 different from Week 14?**
A: Week 14 focused on YOLO training. Week 15 focuses on Faster R-CNN inference and comparison.

**Q4: Is GPU required for Week 15?**
A: Recommended but not essential. All notebooks work on CPU (slower).

**Q5: Which model should I use for Tutorial T15?**
A: `fasterrcnn_resnet50_fpn` (recommended) or `fasterrcnn_mobilenet_v3_large_fpn` (faster on CPU).

### Technical Questions

**Q6: How do I visualize Faster R-CNN detections?**
A: See Notebook 03 (Tutorial T15) for complete code example with matplotlib.

**Q7: What confidence threshold should I use?**
A: Start with 0.7. Adjust based on results (higher = fewer detections, lower = more false positives).

**Q8: Can I use Faster R-CNN for custom objects?**
A: Yes, via fine-tuning. See Notebook 03 optional section (not required for course).

**Q9: How do I compare YOLO and Faster R-CNN?**
A: See Notebook 04 for benchmarking framework (speed, accuracy, visual comparison).

**Q10: What's the difference between torchvision and Detectron2?**
A: torchvision is simpler (recommended for beginners). Detectron2 is more advanced (optional).

### Exam Questions

**Q11: What will be covered in the final exam?**
A: All 5 modules, with emphasis on Modules 4-5 (CNNs, Transfer Learning, Object Detection).

**Q12: Are Week 15 notebooks enough for exam prep?**
A: No. Review all 3 weeks (13-15) and practice questions in lecture notes.

**Q13: Will there be coding questions in the exam?**
A: Likely yes. Practice loading models, inference, and visualization.

**Q14: Should I memorize all numbers (mAP, FPS)?**
A: Know approximate ranges (YOLO: 30-120 FPS, Faster R-CNN: 5-7 FPS, mAP trends).

**Q15: How should I prepare for comparative questions?**
A: Understand trade-offs (speed vs accuracy), use decision framework from Notebook 05.

---

## Project Ideas {#project-ideas}

### Beginner Projects

1. **Custom Image Detection**
   Run Faster R-CNN on personal photos, compare with YOLO results

2. **Confidence Threshold Analysis**
   Experiment with thresholds (0.3 to 0.9), analyze precision-recall trade-off

3. **Model Variant Comparison**
   Compare ResNet-50 vs MobileNet Faster R-CNN on same dataset

### Intermediate Projects

4. **Speed Benchmarking Tool**
   Create tool to benchmark YOLO vs Faster R-CNN on custom images

5. **Detection Visualizer**
   Build interactive tool to visualize detections from multiple models

6. **Video Object Detection**
   Apply Faster R-CNN to video (frame-by-frame), analyze FPS

### Advanced Projects

7. **Fine-Tuning for Custom Dataset**
   Collect custom dataset, fine-tune Faster R-CNN (optional, challenging)

8. **Hybrid Detection System**
   YOLO for initial detection, Faster R-CNN for refinement

9. **Detector Selection System**
   Build automated tool to recommend detector based on requirements

10. **Comparative Analysis Paper**
    Write comprehensive analysis of YOLO vs Faster R-CNN for specific application

---

## Credits and References {#credits-references}

### Course Materials

**Created By:**
- Instructor: [Your Name]
- Institution: SRM University, M.Tech Program
- Course: 21CSE558T - Deep Neural Network Architectures
- Date: November 2025

**Acknowledgments:**
- PyTorch team for torchvision pre-trained models
- Ross Girshick and collaborators for R-CNN research
- COCO dataset team for benchmarking standards
- SRM University for course support

### Research Citations

1. Girshick et al. (2014), "Rich feature hierarchies for accurate object detection", CVPR
2. Girshick (2015), "Fast R-CNN", ICCV
3. Ren et al. (2015), "Faster R-CNN", NeurIPS
4. He et al. (2017), "Mask R-CNN", ICCV

### Tools and Libraries

- PyTorch: https://pytorch.org/
- torchvision: https://pytorch.org/vision/
- OpenCV: https://opencv.org/
- Matplotlib: https://matplotlib.org/
- Google Colab: https://colab.research.google.com/

---

## Next Steps {#next-steps}

### Immediate (This Week)

1. **Complete Tutorial T15** (Notebook 03)
2. **Review lecture notes** (20 pages)
3. **Practice questions** (Section 10.2)
4. **Compare with Week 14** (YOLO vs R-CNN)

### Short-Term (Next Week)

1. **Module 5 Review** (Weeks 13-15)
2. **Final exam preparation**
3. **Complete all practice exercises**
4. **Synthesize learnings**

### Long-Term (After Course)

1. **Explore Detectron2** (advanced framework)
2. **Read original research papers**
3. **Try fine-tuning on custom dataset**
4. **Build portfolio projects**
5. **Stay updated with latest research**

### Career Development

**Skills Acquired:**
- Object detection (YOLO, Faster R-CNN)
- PyTorch/TensorFlow expertise
- Computer vision fundamentals
- Model evaluation and benchmarking

**Next Learning:**
- Instance segmentation (Mask R-CNN)
- 3D object detection
- Video understanding
- Transformer-based detection (DETR)

**Industry Applications:**
- Autonomous vehicles
- Surveillance systems
- Medical imaging
- Retail analytics

---

## Contact Information {#contact-information}

### Instructor

**Name**: [Instructor Name]
**Email**: [instructor@srmuniversity.edu.in]
**Office Hours**: [Schedule]
**Office Location**: [Building/Room]

### Course Resources

**Course Website**: [URL if available]
**Discussion Forum**: [Platform/URL]
**Assignment Submissions**: [Platform]

### Technical Support

**Google Colab Issues**: https://research.google.com/colaboratory/faq.html
**PyTorch Forums**: https://discuss.pytorch.org/
**Stack Overflow**: Tag questions with `pytorch`, `object-detection`, `faster-rcnn`

### Emergency Contact

For urgent course-related queries:
**Department Office**: [Contact info]
**Hours**: Monday-Friday, 9 AM - 5 PM

---

## Document Information

**Version**: 1.0
**Last Updated**: November 18, 2025
**Status**: Complete
**Lines**: ~530
**Maintainer**: Course Instructor

**Changelog:**
- v1.0 (Nov 18, 2025): Initial release
- Covers Week 15: R-CNN Family (Final week)
- Includes 5 notebooks, lecture notes, requirements.txt

**License**: Educational use only (SRM University)

---

## Quick Links

- [Lecture Notes](v1_comprehensive_lecture_notes.md)
- [Requirements](requirements.txt)
- [Notebook 01: R-CNN Evolution](notebooks/01_rcnn_family_evolution.ipynb)
- [Notebook 02: Region Proposals](notebooks/02_region_proposals_selective_search.ipynb)
- [Notebook 03: Tutorial T15](notebooks/03_faster_rcnn_pretrained.ipynb)
- [Notebook 04: YOLO vs R-CNN](notebooks/04_yolo_vs_rcnn_benchmark.ipynb)
- [Notebook 05: Decision Framework](notebooks/05_choosing_detector_decision_tree.ipynb)

---

**End of README - Week 15: R-CNN Family - Two-Stage Object Detection**

**Good luck with your final week and final examination!**
