# Famous CNN Architectures - Analysis Worksheet

**Course:** 21CSE558T - Deep Neural Network Architectures
**Module 4:** CNNs (Week 2 of 3)
**Date:** November 1, 2025
**Name:** _________________________
**Roll No:** _________________________

---

## Instructions

This worksheet helps you practice analyzing and comparing famous CNN architectures. Work through each problem, showing calculations and reasoning.

**Due:** Before Tutorial T11 (Monday, November 3, 2025)

---

## Problem 1: Timeline and Historical Context (10 marks)

### Q1.1 (4 marks)
Fill in the timeline with correct years and key innovations:

| Year | Architecture | Key Innovation | ImageNet Top-5 Accuracy |
|------|--------------|----------------|-------------------------|
| ____ | LeNet-5 | _________________ | N/A (pre-ImageNet) |
| ____ | AlexNet | _________________ | _____% |
| ____ | VGG-16 | _________________ | _____% |
| 2015 | ResNet | Skip connections | 3.6% |

### Q1.2 (3 marks)
**Explain the "14-year gap" between LeNet-5 and AlexNet:**

```
What held CNNs back (1998-2012)?
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________

What changed by 2012?
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
```

### Q1.3 (3 marks)
**Multiple Choice:** AlexNet's breakthrough impact was primarily due to: (Circle ALL correct)

a) Larger dataset (ImageNet)
b) GPU acceleration
c) ReLU activation function
d) Deeper network (8 layers)
e) Dropout regularization
f) Using Python instead of MATLAB

**Number of correct answers:** ________

---

## Problem 2: Parameter Calculation (15 marks)

### Q2.1 (5 marks)
**Calculate parameters for AlexNet Conv1:**

Given:
- Input: 224×224×3 RGB image
- Filter: 11×11, 96 filters, stride=4
- Bias: Yes

**Formula:** Parameters = (K_h × K_w × C_in × C_out) + C_out

**Calculate:**
```
Parameters = (11 × 11 × _____ × _____) + _____
           = ___________ + _____
           = ___________ parameters
```

### Q2.2 (5 marks)
**Calculate parameters for VGG Conv3-256 (from 128 channels):**

Given:
- Input channels: 128
- Filter: 3×3, 256 filters
- Bias: Yes

**Calculate:**
```
Parameters = (3 × 3 × _____ × _____) + _____
           = ___________ + _____
           = ___________ parameters
```

### Q2.3 (5 marks)
**Calculate parameters for AlexNet FC6:**

Given:
- Input: 6×6×256 feature maps (flattened)
- Output: 4,096 neurons
- Bias: Yes

**Calculate:**
```
Flattened input size = 6 × 6 × 256 = _________

Parameters = (_________ × 4,096) + 4,096
           = ___________ + 4,096
           = ___________ parameters
```

**Express in millions:** _________ M parameters

---

## Problem 3: Architecture Comparison (20 marks)

### Q3.1 (12 marks)
**Complete the comparison table:**

| Feature | LeNet-5 | AlexNet | VGG-16 |
|---------|---------|---------|--------|
| Input size | ___×___×___ | ___×___×___ | ___×___×___ |
| Number of conv layers | _____ | _____ | _____ |
| Number of FC layers | _____ | _____ | _____ |
| Total parameters | _____ | _____ | _____ |
| Filter sizes used | _____ | _____ | _____ |
| Pooling type | _____ | _____ | _____ |
| Activation function | _____ | _____ | _____ |
| Dropout used? | Yes/No | Yes/No | Yes/No |
| Data augmentation? | Yes/No | Yes/No | Yes/No |

### Q3.2 (8 marks)
**Parameter distribution analysis:**

Calculate the percentage of parameters in FC layers for each architecture:

**LeNet-5:**
```
Conv parameters: ~10K
FC parameters: ~50K
Total: 60K

FC percentage = (50K / 60K) × 100 = ________%
```

**AlexNet:**
```
Conv parameters: 3.7M
FC parameters: 58.6M
Total: 60M

FC percentage = (_____ / _____) × 100 = ________%
```

**VGG-16:**
```
Conv parameters: 14.7M
FC parameters: 123.6M
Total: 138M

FC percentage = (_____ / _____) × 100 = ________%
```

**What pattern do you observe?**
```



```

**Why is this a problem?**
```



```

---

## Problem 4: Receptive Field Analysis (15 marks)

### Q4.1 (6 marks)
**Why does VGG use 3×3 filters exclusively?**

Compare two approaches for achieving a 5×5 receptive field:

**Approach A: One 5×5 conv layer**
```
Receptive field: 5×5
Parameters per filter: 5 × 5 = _____
Number of ReLU activations: _____
```

**Approach B: Two 3×3 conv layers stacked**
```
Receptive field: 5×5 (same!)
Parameters per filter: (3 × 3) + (3 × 3) = _____
Number of ReLU activations: _____
```

**Which is better and why?**
```




```

### Q4.2 (5 marks)
**Calculate receptive field for three stacked 3×3 convolutions:**

```
After 1st conv (3×3): Receptive field = 3×3
After 2nd conv (3×3): Receptive field = ___×___
After 3rd conv (3×3): Receptive field = ___×___
```

**This is equivalent to one ___×___ convolution!**

**How many parameters saved?**
```
One 7×7 filter: 7 × 7 = _____ parameters
Three 3×3 filters: (3 × 3) × 3 = _____ parameters
Savings: _____ parameters (____% reduction)
```

### Q4.3 (4 marks)
**Sketch the receptive field growth:**

Draw how a 3×3 filter's receptive field grows when stacked:

```
Layer 1 input:          Layer 1 output affects:      Layer 2 output affects:
[  ][  ][  ]           [  ][  ][  ][  ][  ]        [  ][  ][  ][  ][  ][  ][  ]
[  ][x][  ]            [  ][  ][  ][  ][  ]        [  ][  ][  ][  ][  ][  ][  ]
[  ][  ][  ]           [  ][  ][x][  ][  ]        [  ][  ][  ][x][  ][  ][  ]
                       [  ][  ][  ][  ][  ]        [  ][  ][  ][  ][  ][  ][  ]
                       [  ][  ][  ][  ][  ]        [  ][  ][  ][  ][  ][  ][  ]

   3×3 input              5×5 affected                 7×7 affected
```

---

## Problem 5: Output Dimension Tracking (15 marks)

### Q5.1 (8 marks)
**Track dimensions through AlexNet's first 3 layers:**

**Formula:** Output = ⌊(Input - Kernel + 2×Pad) / Stride⌋ + 1

**Layer 1: Conv(96, 11×11, stride=4, pad=0)**
```
Input: 224×224×3
Output_h = ⌊(224 - 11 + 0) / 4⌋ + 1 = ⌊_____ / 4⌋ + 1 = _____ + 1 = _____
Output_w = _____
Output_channels = _____
Result: _____×_____×_____
```

**Layer 2: MaxPool(3×3, stride=2)**
```
Input: _____×_____×_____
Output_h = ⌊(_____ - 3) / 2⌋ + 1 = _____
Output_w = _____
Result: _____×_____×_____
```

**Layer 3: Conv(256, 5×5, stride=1, pad=2)**
```
Input: _____×_____×_____
Output_h = ⌊(_____ - 5 + 2×2) / 1⌋ + 1 = _____
Output_w = _____
Output_channels = _____
Result: _____×_____×_____
```

### Q5.2 (7 marks)
**Track dimensions through VGG-16 Block 1:**

```
Input: 224×224×3

Conv3-64 (3×3, pad=1, stride=1):
→ Output: _____×_____×_____

Conv3-64 (3×3, pad=1, stride=1):
→ Output: _____×_____×_____

MaxPool (2×2, stride=2):
→ Output: _____×_____×_____
```

**Verify the pattern for Block 2 (starts with 112×112×64):**

```
Conv3-128 → _____×_____×_____
Conv3-128 → _____×_____×_____
MaxPool   → _____×_____×_____
```

---

## Problem 6: Activation Functions (10 marks)

### Q6.1 (6 marks)
**Compare activation functions:**

| Activation | Formula | Range | Gradient (x=0) | Gradient (x=5) | Vanishes? |
|------------|---------|-------|----------------|----------------|-----------|
| Sigmoid | 1/(1+e^(-x)) | (0, 1) | _____ | _____ | Yes/No |
| Tanh | (e^x-e^(-x))/(e^x+e^(-x)) | (-1, 1) | _____ | _____ | Yes/No |
| ReLU | max(0, x) | [0, ∞) | _____ | _____ | Yes/No |

**Calculate sigmoid gradient at x=5:**
```
sigmoid(5) = 1/(1+e^(-5)) ≈ 0.993
gradient = sigmoid(5) × (1 - sigmoid(5))
         = 0.993 × (1 - 0.993)
         = 0.993 × 0.007
         = ___________
```

**Is this good for backpropagation?** ___________

### Q6.2 (4 marks)
**Why did ReLU revolutionize deep learning?**

List 4 advantages of ReLU over sigmoid/tanh:

1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________

---

## Problem 7: Design Decision Analysis (15 marks)

### Q7.1 (8 marks)
**For each scenario, choose the best architecture and justify:**

**Scenario A: Handwritten digit recognition (MNIST)**
- Dataset: 60K grayscale 28×28 images, 10 classes
- Deployment: Raspberry Pi (limited memory)
- Requirement: 98%+ accuracy

**Best architecture:** _________________

**Justification:**
```




```

**Scenario B: ImageNet classification**
- Dataset: 1.2M RGB 224×224 images, 1000 classes
- Deployment: Cloud server (powerful GPU)
- Requirement: Maximum accuracy

**Best architecture:** _________________

**Justification:**
```




```

**Scenario C: Transfer learning for bird species (200 classes)**
- Dataset: 5K RGB images (small!)
- Deployment: Web application
- Requirement: Good accuracy, reasonable speed

**Best architecture:** _________________

**Justification:**
```




```

### Q7.2 (7 marks)
**Analyze this architecture choice:**

A student wants to classify 20 flower species with 2,000 training images (100 per class). They decide to train VGG-16 from scratch.

**Is this a good decision?** Yes / No (Circle one)

**Identify 3 problems with this approach:**
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________

**Propose a better solution:**
```





```

---

## Problem 8: Innovation Impact (10 marks)

### Q8.1 (5 marks)
**Rank these innovations by impact (1 = most important, 5 = least important):**

____ ReLU activation function
____ GPU training acceleration
____ Dropout regularization
____ Data augmentation
____ 3×3 filter standardization

**Justify your #1 choice:**
```




```

### Q8.2 (5 marks)
**Match each innovation to its primary benefit:**

| Innovation | Benefit |
|------------|---------|
| 1. ReLU | A. Prevents overfitting by forcing redundant representations |
| 2. Dropout | B. Enables training very deep networks without vanishing gradients |
| 3. Max Pooling | C. Artificially increases dataset size and variation |
| 4. Data Augmentation | D. Provides translation invariance and reduces spatial dimensions |
| 5. 3×3 Filters | E. Fewer parameters and more non-linearity than large filters |

**Answers:**
1 → _____, 2 → _____, 3 → _____, 4 → _____, 5 → _____

---

## Problem 9: Debugging Architecture Issues (10 marks)

### Q9.1 (5 marks)
**This AlexNet implementation has errors. Find and fix them:**

```python
# INCORRECT CODE
model = Sequential([
    Conv2D(96, (11,11), strides=4, activation='sigmoid',  # Error 1?
           input_shape=(224,224,3)),
    AveragePooling2D((3,3), strides=2),                   # Error 2?

    Conv2D(256, (5,5), activation='relu'),
    MaxPooling2D((3,3), strides=2),

    Conv2D(384, (3,3), activation='relu'),
    Conv2D(384, (3,3), activation='relu'),
    Conv2D(256, (3,3), activation='relu'),
    MaxPooling2D((3,3), strides=2),

    Flatten(),
    Dense(4096, activation='relu'),                       # Error 3?
    Dense(4096, activation='relu'),                       # Error 4?
    Dense(1000, activation='softmax')
])
```

**Errors found:**
1. Line ___: ______________________________________
2. Line ___: ______________________________________
3. Line ___: ______________________________________
4. Line ___: ______________________________________

### Q9.2 (5 marks)
**Corrected code:**

```python
model = Sequential([
    # YOUR CORRECTED CODE HERE







])
```

---

## Problem 10: Modern Optimization (Bonus: 5 marks)

**Challenge:** Redesign VGG-16 to reduce parameters while maintaining accuracy.

**Original VGG-16 final layers:**
```
...
Conv3-512 (7×7×512 output)
Flatten → 25,088 values
FC(4096) → ~102M parameters
FC(4096) → ~17M parameters
FC(1000) → ~4M parameters
Total FC: ~123M parameters
```

**Your optimized design:**
```
...
Conv3-512 (7×7×512 output)
_________________________________
_________________________________
_________________________________
Total FC parameters: ________
```

**Parameter reduction:** ________× fewer

**Technique used:** _______________________________

**Expected accuracy impact:** _____________________

---

## Submission Checklist

Before submitting, ensure:
- [ ] All calculations shown with work
- [ ] All comparisons completed
- [ ] All justifications provided
- [ ] Errors corrected in Problem 9
- [ ] Name and roll number at top

---

## Answer Key Summary (For Instructor)

**Problem 1:** 1998, 2012, 2014; AlexNet 84%, VGG 93%; GPU, ReLU, Big Data
**Problem 2:** Conv1: 34,944; Conv3-256: 295,168; FC6: 37,752,832
**Problem 3:** FC percentages: 83%, 94%, 89% (trend: FC dominates)
**Problem 4:** Two 3×3 = 18 params vs One 5×5 = 25 params (28% savings)
**Problem 5:** AlexNet: 55×55×96 → 27×27×96 → 27×27×256
**Problem 6:** Sigmoid gradient at x=5: 0.0066 (vanishing!); ReLU: no saturation
**Problem 7:** MNIST→LeNet; ImageNet→VGG; Transfer→Pre-trained VGG/ResNet
**Problem 8:** Rankings vary; ReLU typically #1 (enables depth)
**Problem 9:** Errors: sigmoid→relu, AvgPool→MaxPool, missing Dropout(0.5) ×2
**Problem 10:** Use GlobalAveragePooling2D → 234× reduction

---

**Total: 100 marks (+5 bonus)**

**Due Date:** Before Tutorial T11 (Monday, November 3, 2025)
