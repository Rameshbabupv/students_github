# Famous CNN Architectures - Quick Reference Cheat Sheet

**Course:** 21CSE558T - Deep Neural Network Architectures
**Module 4:** CNNs (Week 2 of 3)
**Date:** November 1, 2025

---

## 🏛️ The Big Three: LeNet → AlexNet → VGG

| Architecture | Year | Layers | Parameters | Input Size | Key Innovation |
|--------------|------|--------|------------|------------|----------------|
| **LeNet-5** | 1998 | 7 | 60K | 32×32×1 | First practical CNN |
| **AlexNet** | 2012 | 8 | 60M | 224×224×3 | ReLU, Dropout, GPU |
| **VGG-16** | 2014 | 16 | 138M | 224×224×3 | 3×3 everywhere, Depth |

---

## 📊 Architecture Comparison at a Glance

### LeNet-5 (1998) - The Pioneer

**Structure:**
```
Input (32×32×1)
  ↓
Conv(6, 5×5) → Sigmoid → AvgPool(2×2)
  ↓
Conv(16, 5×5) → Sigmoid → AvgPool(2×2)
  ↓
Conv(120, 5×5) → Sigmoid
  ↓
FC(84) → Sigmoid
  ↓
FC(10) → Softmax
```

**Stats:**
- Parameters: 60,000
- FLOPs: ~0.4M
- Model size: 240 KB
- Application: Handwritten digits (MNIST)
- Accuracy: 99% on MNIST

**Key Features:**
- ✅ Proved CNNs work for vision
- ✅ Hierarchical feature learning
- ✅ End-to-end trainable
- ❌ Sigmoid (vanishing gradients)
- ❌ Too small for complex images

---

### AlexNet (2012) - The Breakthrough

**Structure:**
```
Input (224×224×3)
  ↓
Conv(96, 11×11, s=4) → ReLU → MaxPool → LRN
  ↓
Conv(256, 5×5) → ReLU → MaxPool → LRN
  ↓
Conv(384, 3×3) → ReLU
  ↓
Conv(384, 3×3) → ReLU
  ↓
Conv(256, 3×3) → ReLU → MaxPool
  ↓
FC(4096) → ReLU → Dropout(0.5)
  ↓
FC(4096) → ReLU → Dropout(0.5)
  ↓
FC(1000) → Softmax
```

**Stats:**
- Parameters: 60 million
- FLOPs: ~720M
- Model size: 240 MB
- Application: ImageNet (1000 classes)
- Accuracy: 84% top-5 on ImageNet

**Key Innovations:**
- ✅ **ReLU activation** (6× faster training)
- ✅ **Dropout regularization** (0.5 rate)
- ✅ **GPU training** (7× speedup)
- ✅ **Data augmentation** (crops, flips)
- ✅ **Max pooling** (vs average)
- ✅ **Overlapping pooling** (3×3, stride=2)

**Parameter Distribution:**
- Conv layers: 3.7M (6%)
- FC layers: 58.6M (94%) ← Problem!

---

### VGG-16 (2014) - Simplicity Through Depth

**Structure:**
```
Input (224×224×3)
  ↓
[Conv(64, 3×3) → ReLU] × 2 → MaxPool
  ↓
[Conv(128, 3×3) → ReLU] × 2 → MaxPool
  ↓
[Conv(256, 3×3) → ReLU] × 3 → MaxPool
  ↓
[Conv(512, 3×3) → ReLU] × 3 → MaxPool
  ↓
[Conv(512, 3×3) → ReLU] × 3 → MaxPool
  ↓
FC(4096) → ReLU → Dropout(0.5)
  ↓
FC(4096) → ReLU → Dropout(0.5)
  ↓
FC(1000) → Softmax
```

**Stats:**
- Parameters: 138 million
- FLOPs: ~15.5B
- Model size: 552 MB
- Application: ImageNet (1000 classes)
- Accuracy: 93% top-5 on ImageNet

**Key Innovations:**
- ✅ **Only 3×3 filters** (uniformity)
- ✅ **Deeper network** (16-19 layers)
- ✅ **Simple, modular blocks**
- ✅ **Small filters stacked** = large receptive field

**Why 3×3 Filters?**
```
Two 3×3 filters = One 5×5 filter (same receptive field)
Parameters: 18 vs 25 (28% fewer!)
Non-linearity: 2 ReLU vs 1 (more expressive)
```

**Parameter Distribution:**
- Conv layers: 14.7M (11%)
- FC layers: 123.6M (89%) ← Still a problem!

---

## 🔄 Evolution Timeline

**1998 → 2012 (14 years):**
- LeNet proved concept but limited
- "AI Winter" for neural networks
- Traditional CV methods dominated

**2012: The Breakthrough**
- AlexNet wins ImageNet by huge margin
- 84% accuracy (vs 74% traditional methods)
- CNNs become mainstream overnight

**2012 → 2014 (2 years):**
- Rapid innovation period
- VGG, GoogLeNet, others
- Accuracy jumps to 93%

**2015+:**
- ResNet enables 100+ layer networks
- Superhuman performance achieved
- Modern architectures (Transformers, etc.)

---

## 📐 Key Design Patterns

### Pattern 1: Progressive Filter Growth
```
LeNet:   6 → 16 → 120
AlexNet: 96 → 256 → 384 → 384 → 256
VGG:     64 → 128 → 256 → 512 → 512

Rule: Double filters when spatial size halves
```

### Pattern 2: Spatial Reduction
```
All architectures:
224×224 → 112×112 → 56×56 → 28×28 → 14×14 → 7×7

Mechanism: Pooling (2×2) halves dimensions
```

### Pattern 3: Hierarchical Features
```
Early layers:  Edges, colors, simple patterns
Middle layers: Textures, combinations
Deep layers:   Parts, objects, concepts
```

### Pattern 4: Conv → Pool Blocks
```
Standard pattern (all architectures):
[Conv → Activation → (Conv) → Pool] × N → FC → Output
```

---

## 🎯 Activation Functions Evolution

| Era | Activation | Formula | Pros | Cons |
|-----|------------|---------|------|------|
| 1998 | Sigmoid | σ(x) = 1/(1+e^(-x)) | Smooth | Vanishing gradients |
| 1998 | Tanh | tanh(x) = (e^x - e^(-x))/(e^x + e^(-x)) | Zero-centered | Vanishing gradients |
| 2012+ | ReLU | max(0, x) | Fast, no saturation | Dead neurons |

**Why ReLU Won:**
- No vanishing gradients for x > 0
- 6× faster training than sigmoid
- Simple computation
- Sparse activation (biological)

---

## 💾 Parameter & Memory Comparison

### Parameters
| Architecture | Conv Params | FC Params | Total | FC % |
|--------------|-------------|-----------|-------|------|
| LeNet-5 | 10K | 50K | 60K | 83% |
| AlexNet | 3.7M | 58.6M | 60M | 94% |
| VGG-16 | 14.7M | 123.6M | 138M | 89% |

**Key Insight:** FC layers dominate parameter count!

### Computation (FLOPs)
| Architecture | FLOPs | Relative Cost |
|--------------|-------|---------------|
| LeNet-5 | 0.4M | 1× |
| AlexNet | 720M | 1,800× |
| VGG-16 | 15.5B | 38,750× |

### Model Size (FP32)
| Architecture | Size | Mobile-Friendly? |
|--------------|------|------------------|
| LeNet-5 | 240 KB | ✅ Yes |
| AlexNet | 240 MB | ⚠️ Maybe |
| VGG-16 | 552 MB | ❌ No |

---

## 🧮 Parameter Calculation Formulas

### Convolutional Layer
```
Parameters = (K_h × K_w × C_in × C_out) + C_out
              └─────────────┬────────────┘   └──┬──┘
                      weights                 biases

Example: Conv(256, 3×3) from 128 channels
= (3 × 3 × 128 × 256) + 256
= 294,912 + 256
= 295,168 parameters
```

### Fully Connected Layer
```
Parameters = (input_size × output_size) + output_size

Example: FC(4096) from flattened 6×6×256
= (9,216 × 4,096) + 4,096
= 37,752,832 parameters (~38M!)
```

### Output Size (Conv/Pool)
```
Output_size = ⌊(Input_size - Kernel_size + 2×Padding) / Stride⌋ + 1

Example: 224×224 input, Conv(11×11, stride=4, pad=0)
= ⌊(224 - 11 + 0) / 4⌋ + 1
= ⌊213 / 4⌋ + 1
= 53 + 1
= 54 (Actually 55 with padding adjustments)
```

---

## 🎓 Receptive Field Concept

**Receptive Field:** Region of input image that affects one output pixel

### LeNet-5
```
Layer 1 (5×5 conv): RF = 5×5
Layer 2 (5×5 conv): RF = 13×13
Final: Sees most of 32×32 input
```

### Why Stack Small Filters?
```
One 7×7 conv:
- Receptive field: 7×7
- Parameters per filter: 49

Three 3×3 convs:
- Receptive field: 7×7 (same!)
- Parameters per filter: 27 (45% fewer!)
- Non-linearity: 3 ReLU vs 1 (more expressive!)
```

**VGG Insight:** Small filters stacked deeply = efficiency + expressiveness

---

## 🚀 Historical Impact

### ImageNet Competition Results
| Year | Winner | Top-5 Error | Key Innovation |
|------|--------|-------------|----------------|
| 2010 | Traditional CV | 28.2% | Hand-crafted features |
| 2011 | Traditional CV | 25.8% | Better features |
| 2012 | **AlexNet** | **15.3%** | **Deep learning revolution** |
| 2013 | ZFNet | 11.7% | Hyperparameter tuning |
| 2014 | **VGG** | **7.3%** | **Depth + simplicity** |
| 2014 | GoogLeNet | 6.7% | Inception modules |
| 2015 | **ResNet** | **3.6%** | **Skip connections** |
| 2015 | Human | ~5% | (Baseline) |

**2015: First superhuman performance!**

---

## 🔧 Regularization Techniques by Architecture

| Technique | LeNet-5 | AlexNet | VGG-16 |
|-----------|---------|---------|--------|
| Dropout | ❌ | ✅ (0.5) | ✅ (0.5) |
| Data Augmentation | ❌ | ✅ | ✅ |
| BatchNorm | ❌ | ❌ | ❌ |
| L2 Regularization | ⚠️ | ✅ | ✅ |
| Early Stopping | ⚠️ | ✅ | ✅ |

**Note:** BatchNorm came in 2015 (after VGG), now standard in all modern CNNs

---

## 📱 When to Use Which Architecture?

### LeNet-5 (or variants)
**Use When:**
- ✅ Simple grayscale images (28×28, 32×32)
- ✅ 10-100 classes
- ✅ Digit recognition, simple OCR
- ✅ Learning CNN fundamentals
- ✅ CPU-only deployment

**Don't Use When:**
- ❌ Complex natural images (224×224 RGB)
- ❌ 1000+ classes
- ❌ Need high accuracy

---

### AlexNet
**Use When:**
- ✅ Quick prototyping
- ✅ Baseline model comparison
- ✅ Educational purposes
- ✅ Medium-sized datasets (10K-100K)

**Don't Use When:**
- ❌ Need best accuracy (use VGG/ResNet)
- ❌ Mobile deployment (too large)
- ❌ Production systems (outdated)

**Modern Alternative:** ResNet-18

---

### VGG-16
**Use When:**
- ✅ Transfer learning (pre-trained weights)
- ✅ Feature extraction (conv blocks)
- ✅ Style transfer tasks
- ✅ High accuracy needed
- ✅ Have powerful GPU

**Don't Use When:**
- ❌ Limited computational budget
- ❌ Mobile/edge deployment
- ❌ Real-time inference needed
- ❌ Training from scratch on small data

**Modern Alternative:** ResNet-50, EfficientNet

---

## 🎯 Design Principles Learned

### 1. Depth Matters
```
Shallow networks: Limited feature complexity
Deep networks: Hierarchical, rich features
Trend: 8 layers (AlexNet) → 16 layers (VGG) → 152 layers (ResNet)
```

### 2. Small Filters Are Better
```
Large filters (11×11, 7×7): Expensive, less expressive
Small filters (3×3) stacked: Cheap, more expressive
Modern standard: 3×3 or 1×1 only
```

### 3. ReLU Revolutionized Training
```
Before (Sigmoid/Tanh): Slow, vanishing gradients
After (ReLU): 6× faster, stable training
Impact: Made deep networks practical
```

### 4. FC Layers Are the Bottleneck
```
Problem: 90%+ parameters in FC layers
Solution (2015+): Global Average Pooling
Reduction: 234× fewer parameters!
```

### 5. Regularization Is Essential
```
Without: Overfitting on large networks
With: Dropout + Augmentation = Generalization
Modern: + BatchNorm + Early Stopping
```

---

## 🔮 What Came Next? (Preview of Week 12)

### ResNet (2015) - Skip Connections
```
Problem: VGG-30, VGG-50 won't train (gradients vanish)
Solution: Residual connections (shortcuts)
Result: Can train 152+ layer networks!
```

### Modern Architectures
- **Inception/GoogLeNet:** Multi-scale features
- **MobileNet:** Efficient for mobile (< 5MB)
- **EfficientNet:** Optimal accuracy-efficiency
- **Vision Transformers:** State-of-art (2020+)

---

## 💡 Quick Decision Guide

**Starting a new project?**
```
Is your problem similar to ImageNet?
├─ YES → Use pre-trained VGG/ResNet (transfer learning)
└─ NO  → Continue

Is your dataset < 10K images?
├─ YES → Must use transfer learning OR simple custom network
└─ NO  → Continue

Do you need mobile deployment?
├─ YES → Use MobileNet/EfficientNet
└─ NO  → Continue

Training from scratch?
├─ YES → Start with AlexNet-style, scale up if underfitting
└─ NO  → Use pre-trained ResNet-50 (best default choice)
```

---

## 📚 Code Examples (Keras)

### LeNet-5 (Simplified)
```python
model = Sequential([
    Conv2D(6, (5,5), activation='relu', input_shape=(32,32,1)),
    AveragePooling2D((2,2)),
    Conv2D(16, (5,5), activation='relu'),
    AveragePooling2D((2,2)),
    Flatten(),
    Dense(120, activation='relu'),
    Dense(84, activation='relu'),
    Dense(10, activation='softmax')
])
```

### AlexNet (Simplified)
```python
model = Sequential([
    Conv2D(96, (11,11), strides=4, activation='relu', input_shape=(224,224,3)),
    MaxPooling2D((3,3), strides=2),
    Conv2D(256, (5,5), activation='relu'),
    MaxPooling2D((3,3), strides=2),
    Conv2D(384, (3,3), activation='relu'),
    Conv2D(384, (3,3), activation='relu'),
    Conv2D(256, (3,3), activation='relu'),
    MaxPooling2D((3,3), strides=2),
    Flatten(),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(1000, activation='softmax')
])
```

### VGG-16 (Using Keras)
```python
from tensorflow.keras.applications import VGG16

# Load pre-trained weights
model = VGG16(weights='imagenet', include_top=True)

# Or custom top for your classes
base = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(10, activation='softmax')(x)
model = Model(inputs=base.input, outputs=x)
```

---

## 🎓 Study Tips

### For Understanding
1. Draw architectures on paper (block diagrams)
2. Calculate parameters for each layer
3. Trace one example through the network
4. Compare architectures side-by-side

### For Exams
- **Memorize:** Timeline (1998, 2012, 2014, 2015)
- **Understand:** Why each innovation mattered
- **Calculate:** Parameter counts, output dimensions
- **Compare:** Advantages/disadvantages of each

### For Projects
- **Start simple:** LeNet-style for simple problems
- **Use pre-trained:** VGG/ResNet for most tasks
- **Optimize later:** MobileNet/EfficientNet for deployment

---

## ⚠️ Common Mistakes to Avoid

❌ **Training VGG from scratch on small dataset**
- 138M parameters will overfit badly
- Use transfer learning instead

❌ **Using sigmoid/tanh in 2025**
- Vanishing gradients, slow training
- Use ReLU or variants

❌ **Forgetting dropout in FC layers**
- AlexNet/VGG need dropout(0.5) in FC
- Without it, overfitting guaranteed

❌ **Not using data augmentation**
- Essential for AlexNet/VGG
- Without it, test accuracy drops 10-15%

❌ **Comparing architectures unfairly**
- Different training epochs
- Different regularization
- Different hardware
- Always control variables!

---

## 📊 Summary Comparison Table

| Aspect | LeNet-5 | AlexNet | VGG-16 |
|--------|---------|---------|--------|
| **Best For** | Simple tasks | Prototyping | Transfer learning |
| **Strengths** | Fast, small | Proven, balanced | Simple, accurate |
| **Weaknesses** | Limited capacity | Dated | Huge, slow |
| **Modern Use** | Education | Baseline | Feature extractor |
| **Replaced By** | Custom CNNs | ResNet-18 | ResNet-50 |

---

**Print this cheat sheet and bring it to Tutorial T11 (Monday, Nov 3)!**

**Next:** Complete architecture analysis worksheet to practice these concepts 🚀
