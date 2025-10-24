# Week 10 CNN Notebooks - Complete Learning Series

**Course:** Deep Neural Network Architectures (21CSE558T)
**Week:** 10 - Module 4: CNN Basics
**Date:** DO3 (October 27, 2025) - Saturday
**Duration:** 2-hour lecture broken into 10 interactive notebooks

---

## 📚 Notebook Series Overview

This directory contains **10 Jupyter notebooks** (00-09) that provide a comprehensive, hands-on introduction to Convolutional Neural Networks (CNNs). Each notebook is designed to be self-contained yet builds progressively on previous concepts.

### Teaching Philosophy: **80-15-5 Rule**
- **80%** Conceptual understanding (intuition, visualizations, stories)
- **15%** Strategic calculations (key formulas, step-by-step math)
- **5%** Minimal code (focused implementations)

---

## 🗺️ Complete Notebook Map

```
Week 10 CNN Journey
│
├── 📓 Notebook 0: Setup & Prerequisites (10 min)
│   ├── Environment setup
│   ├── Helper functions
│   ├── NumPy review
│   └── Week 9 connection (LBP, GLCM → CNNs)
│
├── 📓 Notebook 1: Convolution Concept & Intuition (15-20 min)
│   ├── Sliding window analogy
│   ├── Pattern detection intuition
│   ├── Detective Kavya's story
│   └── Manual vs learned features
│
├── 📓 Notebook 2: 1D Convolution Math & Code (20-25 min) [HIGH PRIORITY]
│   ├── Mathematical formula
│   ├── Hand calculation (step-by-step)
│   ├── NumPy implementation
│   ├── Signal processing (ECG smoothing)
│   └── Custom filter design
│
├── 📓 Notebook 3: 2D Convolution for Images (25-30 min) [HIGH PRIORITY]
│   ├── 2D convolution mathematics
│   ├── Image edge detection
│   ├── Sobel, Laplacian filters
│   ├── Feature maps visualization
│   └── Character: Meera's X-ray analysis
│
├── 📓 Notebook 4: Convolution Parameters (20 min)
│   ├── Stride (movement step size)
│   ├── Padding (valid vs same)
│   ├── Kernel size trade-offs
│   └── Output dimension formula: (W - F + 2P) / S + 1
│
├── 📓 Notebook 5: Hierarchical Feature Learning (20 min)
│   ├── Feature hierarchy: edges → textures → parts → objects
│   ├── Receptive field concept
│   ├── Connection to Week 9 (LBP/GLCM)
│   └── Automatic feature learning
│
├── 📓 Notebook 6: Pooling Mechanisms (15-20 min)
│   ├── Max pooling vs average pooling
│   ├── Translation invariance
│   ├── Dimension reduction
│   └── Global Average Pooling (GAP)
│
├── 📓 Notebook 7: Complete CNN Architecture (25-30 min) [HIGH PRIORITY]
│   ├── [Conv → ReLU → Pool] × N pattern
│   ├── Dimension tracing (28×28 → 14×14 → 7×7)
│   ├── Parameter calculation
│   ├── LeNet-5 walkthrough
│   ├── CNN vs MLP comparison
│   └── Keras implementation preview
│
├── 📓 Notebook 8: 3D Convolution Preview (10-15 min)
│   ├── 2D vs 3D convolution
│   ├── Video analysis (action recognition)
│   ├── Medical imaging (CT/MRI scans)
│   └── Spatiotemporal features
│
└── 📓 Notebook 9: Review & Tutorial T10 Preview (15-20 min)
    ├── Complete concept map
    ├── Formula cheat sheet
    ├── Self-assessment questions
    ├── Common pitfalls
    ├── Tutorial T10 preview (Fashion-MNIST)
    └── Preparation checklist
```

---

## 🎯 Learning Path

### Recommended Order

1. **Start here:** Notebook 0 (Setup)
2. **Foundation:** Notebooks 1-3 (Concept → 1D → 2D)
3. **Parameters:** Notebook 4 (Stride, padding, kernels)
4. **Deep Understanding:** Notebooks 5-6 (Hierarchy, pooling)
5. **Putting it Together:** Notebook 7 (Complete CNN)
6. **Advanced Topic:** Notebook 8 (3D convolution)
7. **Consolidation:** Notebook 9 (Review & preview)

### High-Priority Notebooks

If time is limited, focus on:
- ✅ **Notebook 2**: 1D Convolution (foundational math)
- ✅ **Notebook 3**: 2D Convolution (core CNN operation)
- ✅ **Notebook 7**: Complete CNN Architecture (practical implementation)

---

## 📖 Character-Driven Stories

The notebooks feature recurring characters to make concepts memorable:

- **Character: Detective Kavya**: Pattern recognition expert (Week 9 callback)
- **Character: Dr. Priya**: Medical imaging researcher
- **Character: Arjun**: Biomedical engineering student
- **Character: Meera**: Radiology AI specialist
- **Character: Dr. Rajesh**: Clinic director

**Note:** All character names follow the "Character:" prefix convention as per course guidelines.

---

## 🛠️ Technical Requirements

### Required Packages

```bash
pip install numpy matplotlib scipy scikit-image tensorflow opencv-python
```

### Python Version
- Python 3.8+ recommended
- Compatible with Google Colab, JupyterLab, VS Code

### Hardware
- CPU sufficient for all notebooks
- GPU optional (helpful for Notebook 7)

---

## 📝 Key Concepts Covered

### Mathematical Foundations
1. **1D Convolution Formula:**
   ```
   output[n] = Σ input[n+k] · kernel[k]
   ```

2. **2D Convolution Formula:**
   ```
   output[i,j] = ΣΣ input[i+m, j+n] · kernel[m,n]
   ```

3. **Output Dimension Formula:**
   ```
   Output = ⌊(W - F + 2P) / S⌋ + 1
   ```
   - W = input width
   - F = filter size
   - P = padding
   - S = stride

4. **Parameter Count (Conv Layer):**
   ```
   Params = (Fh × Fw × Cin + 1) × Cout
   ```

### Core CNN Concepts
- Sliding window operation
- Feature maps and filters
- Stride and padding
- Pooling (max, average, global)
- Hierarchical feature learning
- Translation invariance
- Receptive fields

---

## 🔗 Connection to Course Flow

### Backward Links (What Built This)
```
Module 1-2: Neural networks, backprop, optimization
    ↓
Module 3: Image processing, manual feature extraction
    ↓
Week 9: CNN motivation (WHY?) - LBP, GLCM, shape features
    ↓
Week 10: CNN mechanics (HOW?) ← CURRENT NOTEBOOKS
```

### Forward Links (What Comes Next)
```
Week 10: Basic CNN mechanics ← TODAY (these notebooks)
    ↓
Week 11: CNN layers, regularization, famous architectures
    ↓
Week 12: Transfer learning, pre-trained models
    ↓
Tutorial T10: Building CNN in Keras (DO4, Oct 29, Monday)
```

---

## 🚀 Quick Start Guide

### Option 1: Local Jupyter
```bash
cd notebooks/
jupyter notebook
# Open 00_setup_and_prerequisites.ipynb
```

### Option 2: JupyterLab
```bash
cd notebooks/
jupyter lab
```

### Option 3: VS Code
1. Install Python extension
2. Install Jupyter extension
3. Open any `.ipynb` file
4. Select Python kernel

### Option 4: Google Colab
1. Upload notebooks to Google Drive
2. Open with Google Colab
3. Run cells sequentially

---

## 📊 Progress Tracking

### Completion Checklist

- [ ] Notebook 0: Environment setup verified
- [ ] Notebook 1: Understand convolution intuition
- [ ] Notebook 2: Calculate 1D convolution by hand
- [ ] Notebook 3: Apply 2D filters to images
- [ ] Notebook 4: Master dimension calculations
- [ ] Notebook 5: Grasp feature hierarchy
- [ ] Notebook 6: Implement pooling operations
- [ ] Notebook 7: Build complete CNN in Keras
- [ ] Notebook 8: Understand 3D convolution use cases
- [ ] Notebook 9: Self-assessment complete

### Learning Outcomes

By the end of this notebook series, you will be able to:

1. ✅ **Explain** convolution as a sliding window operation
2. ✅ **Calculate** 1D and 2D convolution outputs manually
3. ✅ **Implement** convolution using NumPy and TensorFlow
4. ✅ **Design** CNN architectures with appropriate parameters
5. ✅ **Compute** output dimensions and parameter counts
6. ✅ **Understand** hierarchical feature learning (edges → objects)
7. ✅ **Apply** pooling for dimension reduction
8. ✅ **Build** complete CNN in Keras
9. ✅ **Compare** CNN vs MLP for image tasks
10. ✅ **Connect** Week 9 manual features to Week 10 learned features

---

## 📚 Pedagogical Features

### Interactive Elements
- **Hand calculations**: Step-by-step math with verification
- **Visualizations**: Every concept has a visual representation
- **Code examples**: Minimal, focused implementations
- **Exercises**: Practice problems at the end of each notebook

### Learning Aids
- **Summary sections**: Key takeaways in every notebook
- **Formula cheat sheets**: Quick reference (Notebook 9)
- **Common pitfalls**: What to avoid (Notebook 9)
- **Self-assessment**: Test your understanding (Notebook 9)

### Story-Based Learning
- **Narrative continuity**: Characters appear across notebooks
- **Real-world problems**: ECG analysis, X-ray enhancement
- **Problem → Solution**: Each story demonstrates practical application

---

## 🎓 Assessment Integration

### Unit Test 2 (Oct 31, 2025) Coverage

**MCQ Questions (1 mark each):**
- Convolution parameter effects
- Pooling operation types
- Output dimension calculations
- Receptive field concepts

**5-Mark Questions:**
- Calculate convolution output (show steps)
- Design CNN for specific problem
- Explain biological motivation
- Compare CNN vs MLP

**10-Mark Questions:**
- Complete 2D convolution calculation
- Design and justify CNN architecture
- Implement CNN in Keras with explanation

### Tutorial T10 Preparation

These notebooks prepare you for:
- **Task**: Build CNN for Fashion-MNIST classification
- **Goal**: 90%+ accuracy
- **Skills**: Data loading, model building, training, evaluation
- **Bonus**: Filter visualization

---

## 🔧 Troubleshooting

### Common Issues

**Issue 1: Import errors**
```python
# Solution: Install missing packages
pip install numpy matplotlib scipy scikit-image tensorflow
```

**Issue 2: Matplotlib plots not showing**
```python
# Solution: Add magic command in Jupyter
%matplotlib inline
```

**Issue 3: TensorFlow warnings**
```python
# Solution: Suppress info messages (optional)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
```

**Issue 4: Kernel crashes**
- Reduce image sizes in visualizations
- Close other applications
- Restart kernel and clear outputs

---

## 📖 Additional Resources

### Recommended Reading
1. **Goodfellow et al. (2016)**: Deep Learning, Chapter 9
2. **Chollet (2021)**: Deep Learning with Python, Chapter 5
3. **Stanford CS231n**: CNN for Visual Recognition (notes)
4. **3Blue1Brown**: "But what is a convolution?" (YouTube)

### Week 9 Connection
- `week9/do4-oct-17/lecture_notes.md`: CNN motivation
- Manual feature extraction: LBP, GLCM, shape features
- Why CNNs exist (sets up Week 10's "how")

### Week 10 Lecture Materials
- `v3_comprehensive_lecture_notes.md`: Full 2-hour lecture
- `week10-plan.md`: Weekly schedule and objectives
- `NOTEBOOK_SERIES_GUIDE.md`: This file

---

## 📞 Support and Questions

### During Class
- Ask questions in Notebooks 1-8
- Bring doubts to Notebook 9 review

### Tutorial T10 (DO4, Oct 29, Monday)
- Hands-on implementation
- Instructor guidance
- Debugging help

### Office Hours
- Review difficult notebooks
- Clarify mathematical concepts
- Architecture design guidance

---

## 🎉 Success Metrics

### You're ready for Tutorial T10 if you can:
- ✅ Calculate output dimensions from parameters
- ✅ Explain why CNNs work for images
- ✅ Trace feature map sizes through a network
- ✅ Count parameters in conv and dense layers
- ✅ Understand pooling vs stride trade-offs

### Red Flags (Need Review):
- ❌ Confusion about 2D vs 3D convolution
- ❌ Cannot apply output formula
- ❌ Unclear on padding types (valid vs same)
- ❌ Don't see connection to Week 9

---

## 📝 Notebook Statistics

| Notebook | Duration | Lines of Code | Visualizations | Exercises |
|----------|----------|---------------|----------------|-----------|
| 00 - Setup | 10 min | ~50 | 2 | 1 |
| 01 - Concept | 15-20 min | ~100 | 5 | 3 |
| 02 - 1D Conv | 20-25 min | ~150 | 6 | 3 |
| 03 - 2D Conv | 25-30 min | ~200 | 8 | 3 |
| 04 - Parameters | 20 min | ~80 | 3 | 4 |
| 05 - Hierarchy | 20 min | ~60 | 4 | 2 |
| 06 - Pooling | 15-20 min | ~100 | 4 | 3 |
| 07 - Complete CNN | 25-30 min | ~120 | 5 | 4 |
| 08 - 3D Conv | 10-15 min | ~40 | 2 | 2 |
| 09 - Review | 15-20 min | ~30 | 3 | 10 |
| **Total** | **~3 hours** | **~930** | **42** | **35** |

---

## 🌟 Final Notes

### Best Practices
1. **Sequential learning**: Complete notebooks in order
2. **Run all cells**: Execute every code cell
3. **Understand before moving**: Don't skip concepts
4. **Practice exercises**: Reinforce learning
5. **Visualize everything**: Look at all plots carefully

### Time Management
- **Minimum** (exam prep): Notebooks 0, 2, 3, 7, 9 → ~2 hours
- **Recommended** (full understanding): All notebooks → ~3 hours
- **Optimal** (mastery): All notebooks + extra practice → ~4 hours

### After Completion
1. Review Notebook 9 formula sheet before exam
2. Complete Tutorial T10 hands-on implementation
3. Practice dimension calculations
4. Design your own CNN architectures

---

**Last Updated:** October 23, 2025
**Version:** 1.0
**Maintainer:** Course Instructor

**For questions or feedback:**
- During class: Raise hand or use chat
- Office hours: TBD
- Email: [course email]

---

*Week 10 - Deep Neural Network Architectures (21CSE558T)*
*SRM University - M.Tech Program*

**Happy Learning! 🚀**
