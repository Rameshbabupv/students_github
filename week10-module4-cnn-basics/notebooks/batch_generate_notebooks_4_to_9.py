#!/usr/bin/env python3
"""
Batch generate notebooks 4-9 for Week 10 CNN lecture.
Creates: Parameters, Hierarchy, Pooling, Complete CNN, 3D, Review
"""

import json

def cell(ctype, src):
    """Quick cell creator."""
    c = {"cell_type": ctype, "metadata": {}, "source": src if isinstance(src, list) else [src]}
    if ctype == "code":
        c["execution_count"] = None
        c["outputs"] = []
    return c

def save_nb(name, cells):
    """Save notebook."""
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"codemirror_mode": {"name": "ipython", "version": 3},
                            "file_extension": ".py", "mimetype": "text/x-python",
                            "name": "python", "nbconvert_exporter": "python",
                            "pygments_lexer": "ipython3", "version": "3.8.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    with open(name, 'w') as f:
        json.dump(nb, f, indent=1)
    print(f"✅ Generated {name}")

# NOTEBOOK 4: Convolution Parameters
nb4 = [
    cell("markdown", """# Notebook 4: Convolution Parameters (Stride, Padding, Kernel Size)

**Week 10 - Module 4: CNN Basics**
**DO3 (October 27, 2025) - Saturday**
**Duration:** 20 minutes

## Learning Objectives

1. ✅ **Understand** stride and its effect on output size
2. ✅ **Apply** padding to maintain dimensions
3. ✅ **Calculate** output dimensions using the formula
4. ✅ **Compare** different kernel sizes and their trade-offs

---"""),

    cell("code", """import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

plt.rcParams['figure.figsize'] = (14, 6)
print("✅ Setup complete!")"""),

    cell("markdown", """## 1. Parameter #1: Stride

**Stride** = How many pixels to move when sliding the kernel

- **Stride = 1**: Slide 1 pixel at a time (most common)
- **Stride = 2**: Slide 2 pixels (faster, smaller output)
- **Stride = 3**: Slide 3 pixels (even smaller output)

### Output Size Formula (Stride):

$$
\\text{Output Size} = \\left\\lfloor \\frac{W - F}{S} \\right\\rfloor + 1
$$

Where:
- $W$ = input width
- $F$ = filter size
- $S$ = stride

---"""),

    cell("code", """# Demonstrate different strides
input_1d = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
kernel_1d = np.array([1, 1, 1]) / 3

def convolve_with_stride(signal, kernel, stride):
    \"\"\"Manual convolution with stride.\"\"\"
    n = len(signal)
    k = len(kernel)
    output_size = (n - k) // stride + 1
    output = np.zeros(output_size)
    for i in range(output_size):
        pos = i * stride
        output[i] = np.sum(signal[pos:pos+k] * kernel)
    return output

# Test different strides
for stride in [1, 2, 3]:
    output = convolve_with_stride(input_1d, kernel_1d, stride)
    print(f"Stride {stride}: Output size = {len(output)}, values = {output}")"""),

    cell("markdown", """## 2. Parameter #2: Padding

**Padding** = Adding border pixels to input

**Why padding?**
- Maintain spatial dimensions
- Preserve border information
- Control output size

**Types:**
- **Valid**: No padding (output shrinks)
- **Same**: Zero padding (output = input size)

---"""),

    cell("code", """# Visualize padding
img_small = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
kernel_3x3 = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])

# Valid mode (no padding)
valid = convolve2d(img_small, kernel_3x3, mode='valid')
# Same mode (with padding)
same = convolve2d(img_small, kernel_3x3, mode='same')

print(f"Input shape: {img_small.shape}")
print(f"Valid mode output: {valid.shape}")
print(f"Same mode output: {same.shape}")"""),

    cell("markdown", """## 3. Output Dimension Formula (Complete)

### The Master Formula:

$$
\\text{Output} = \\left\\lfloor \\frac{W - F + 2P}{S} \\right\\rfloor + 1
$$

Where:
- $W$ = input width/height
- $F$ = filter size
- $P$ = padding
- $S$ = stride

### Example Calculations:

**Case 1:** Input=28×28, Filter=5×5, Stride=1, Padding=0
$$
\\text{Output} = \\frac{28 - 5 + 0}{1} + 1 = 24
$$

**Case 2:** Input=32×32, Filter=3×3, Stride=2, Padding=1
$$
\\text{Output} = \\frac{32 - 3 + 2}{2} + 1 = 16
$$

---"""),

    cell("code", """def calculate_output_size(input_size, filter_size, stride, padding):
    \"\"\"Calculate output dimensions.\"\"\"
    return ((input_size - filter_size + 2*padding) // stride) + 1

# Test cases
cases = [
    (28, 5, 1, 0),  # MNIST with 5x5 filter
    (32, 3, 1, 1),  # CIFAR with 3x3 filter, padding
    (224, 7, 2, 3), # ImageNet first layer
]

print("Input | Filter | Stride | Padding | Output")
print("-" * 50)
for w, f, s, p in cases:
    out = calculate_output_size(w, f, s, p)
    print(f"{w:5} | {f:6} | {s:6} | {p:7} | {out:6}")"""),

    cell("markdown", """## 4. Parameter #3: Kernel Size

**Common kernel sizes:**
- **3×3**: Most common (VGG, ResNet)
- **5×5**: Wider receptive field
- **1×1**: Channel mixing (no spatial)
- **7×7**: First layer of deep networks

### Trade-offs:

| Kernel Size | Pros | Cons |
|-------------|------|------|
| **1×1** | Fast, fewer params | No spatial info |
| **3×3** | Balanced, stackable | Smaller receptive field |
| **5×5** | Larger receptive field | More parameters |
| **7×7** | Very wide context | Expensive |

---"""),

    cell("markdown", """## Summary

### 🎯 Key Formulas

1. **Output Size:**
   $$\\text{Output} = \\left\\lfloor \\frac{W - F + 2P}{S} \\right\\rfloor + 1$$

2. **Common Configurations:**
   - **Same padding**: $P = (F-1)/2$ (for stride=1)
   - **Valid padding**: $P = 0$

3. **Parameter Count:**
   - For filter $F \\times F \\times C_{in} \\times C_{out}$
   - Params = $F^2 \\times C_{in} \\times C_{out} + C_{out}$ (with bias)

### 🔮 Next

**Notebook 5:** Hierarchical Feature Learning (edges → shapes → objects)

---

*Week 10 - Deep Neural Network Architectures (21CSE558T)*
*SRM University - M.Tech Program*""")
]

# NOTEBOOK 5: Hierarchical Feature Learning
nb5 = [
    cell("markdown", """# Notebook 5: Hierarchical Feature Learning

**Week 10 - Module 4: CNN Basics**
**DO3 (October 27, 2025) - Saturday**
**Duration:** 20 minutes

## Learning Objectives

1. ✅ **Understand** feature hierarchy in CNNs (edges → textures → parts → objects)
2. ✅ **Visualize** how features evolve across layers
3. ✅ **Connect** to Week 9 manual features (LBP, GLCM)
4. ✅ **Explain** receptive field concept

---"""),

    cell("code", """import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (14, 8)
print("✅ Setup complete!")"""),

    cell("markdown", """## 1. The Hierarchy: From Week 9 to Week 10

### Week 9: Manual Feature Design
- **LBP**: Local texture patterns (3×3 neighborhoods)
- **GLCM**: Co-occurrence statistics
- **Shape features**: Geometric properties

**Problem:** WE designed these features manually

### Week 10: Learned Feature Hierarchy
- **Layer 1**: Edges, colors (like manual edge detectors)
- **Layer 2**: Textures, patterns (like LBP/GLCM)
- **Layer 3**: Object parts (eyes, wheels)
- **Layer 4**: Complete objects (faces, cars)

**Solution:** CNN LEARNS this hierarchy automatically!

---"""),

    cell("markdown", """## 2. Feature Hierarchy Visualization

```
Input Image (224×224×3 RGB)
    ↓
Layer 1: Edge Detectors (Conv 3×3, 64 filters)
    → Detects: Horizontal, vertical, diagonal edges
    → Like: Sobel, Prewitt filters from Week 9
    ↓
Layer 2: Texture Detectors (Conv 3×3, 128 filters)
    → Detects: Patterns, textures
    → Like: LBP, GLCM from Week 9
    ↓
Layer 3: Part Detectors (Conv 3×3, 256 filters)
    → Detects: Eyes, wheels, windows
    → Combines: Edges + textures
    ↓
Layer 4: Object Detectors (Conv 3×3, 512 filters)
    → Detects: Faces, cars, animals
    → Combines: Multiple parts
```

---"""),

    cell("code", """# Simulate feature hierarchy
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# Layer 1: Edges
edges = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
axes[0].imshow(edges, cmap='gray')
axes[0].set_title('Layer 1:\\nEdges', fontsize=13, fontweight='bold')
axes[0].axis('off')

# Layer 2: Textures
texture = np.random.rand(5, 5)
axes[1].imshow(texture, cmap='gray')
axes[1].set_title('Layer 2:\\nTextures', fontsize=13, fontweight='bold')
axes[1].axis('off')

# Layer 3: Parts
part = np.zeros((10, 10))
part[2:8, 2:8] = 1  # Simple shape
axes[2].imshow(part, cmap='gray')
axes[2].set_title('Layer 3:\\nParts', fontsize=13, fontweight='bold')
axes[2].axis('off')

# Layer 4: Objects
obj = np.zeros((15, 15))
obj[3:12, 3:12] = 1  # Larger shape
axes[3].imshow(obj, cmap='gray')
axes[3].set_title('Layer 4:\\nObjects', fontsize=13, fontweight='bold')
axes[3].axis('off')

plt.tight_layout()
plt.show()

print("🎯 Feature Hierarchy:")
print("  Layer 1 (simple) → Layer 4 (complex)")"""),

    cell("markdown", """## 3. Receptive Field Concept

**Receptive Field** = Region of input that affects a neuron

- **Layer 1**: 3×3 receptive field
- **Layer 2**: 5×5 receptive field (combines 3×3 from layer 1)
- **Layer 3**: 7×7 receptive field
- **Layer 4**: 9×9 receptive field

**Key Insight:** Stacking convolutions increases receptive field WITHOUT large kernels!

---"""),

    cell("markdown", """## Summary

### 🎯 Hierarchy

1. **Layer 1**: Edges, colors → replaces Sobel, Prewitt
2. **Layer 2**: Textures → replaces LBP, GLCM
3. **Layer 3**: Parts → combines features
4. **Layer 4**: Objects → final detection

### 🔮 Next

**Notebook 6:** Pooling Mechanisms (dimension reduction, translation invariance)

---

*Week 10 - Deep Neural Network Architectures (21CSE558T)*
*SRM University - M.Tech Program*""")
]

# Save notebooks 4 and 5
save_nb('04_convolution_parameters.ipynb', nb4)
save_nb('05_hierarchical_feature_learning.ipynb', nb5)

# NOTEBOOK 6: Pooling Mechanisms
nb6 = [
    cell("markdown", """# Notebook 6: Pooling Layers

**Week 10 - Module 4: CNN Basics**
**DO3 (October 27, 2025) - Saturday**
**Duration:** 15-20 minutes

## Learning Objectives

1. ✅ **Understand** why pooling is needed
2. ✅ **Implement** max pooling and average pooling
3. ✅ **Calculate** pooling output dimensions
4. ✅ **Compare** pooling vs stride

---"""),

    cell("code", """import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (12, 6)
print("✅ Setup complete!")"""),

    cell("markdown", """## 1. Why Pooling?

**Three Main Reasons:**

1. **Translation Invariance**: Small shifts don't matter
2. **Dimension Reduction**: Reduce computational load
3. **Overfitting Prevention**: Fewer parameters to learn

### Example:

If a cat's eye moves 1 pixel, we still want to detect it!

---"""),

    cell("markdown", """## 2. Max Pooling

**Max Pooling** = Take maximum value in window

**Example (2×2 max pooling):**

```
Input (4×4):          Output (2×2):
[1, 3, 2, 4]          [3, 4]
[5, 6, 7, 8]     →    [6, 8]
[9, 2, 3, 4]
[1, 5, 6, 7]          [9, 7]
```

**Calculation:**
- Top-left: max(1,3,5,6) = 6
- Top-right: max(2,4,7,8) = 8
- Bottom-left: max(9,2,1,5) = 9
- Bottom-right: max(3,4,6,7) = 7

---"""),

    cell("code", """def max_pool2d(image, pool_size=2, stride=2):
    \"\"\"Implement 2D max pooling.\"\"\"
    h, w = image.shape
    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1
    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            start_i = i * stride
            start_j = j * stride
            window = image[start_i:start_i+pool_size, start_j:start_j+pool_size]
            output[i, j] = np.max(window)

    return output

# Test
test_img = np.array([[1, 3, 2, 4],
                     [5, 6, 7, 8],
                     [9, 2, 3, 4],
                     [1, 5, 6, 7]])

pooled = max_pool2d(test_img, pool_size=2, stride=2)
print("Input (4×4):")
print(test_img)
print("\\nMax Pooled (2×2):")
print(pooled)"""),

    cell("markdown", """## 3. Average Pooling

**Average Pooling** = Take average value in window

Same example:
```
Input (4×4):          Output (2×2):
[1, 3, 2, 4]          [3.75, 5.25]
[5, 6, 7, 8]     →
[9, 2, 3, 4]          [4.25, 5.0]
[1, 5, 6, 7]
```

**When to use which?**
- **Max Pooling**: Most common, preserves strongest features
- **Average Pooling**: Smoother, used in final layers

---"""),

    cell("code", """def avg_pool2d(image, pool_size=2, stride=2):
    \"\"\"Implement 2D average pooling.\"\"\"
    h, w = image.shape
    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1
    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            start_i = i * stride
            start_j = j * stride
            window = image[start_i:start_i+pool_size, start_j:start_j+pool_size]
            output[i, j] = np.mean(window)

    return output

avg_pooled = avg_pool2d(test_img, pool_size=2, stride=2)
print("Average Pooled (2×2):")
print(avg_pooled)
print("\\nMax vs Average:")
print(f"Max: {pooled[0,0]:.2f} | Avg: {avg_pooled[0,0]:.2f}")"""),

    cell("markdown", """## 4. Global Average Pooling (GAP)

**GAP** = Average over entire spatial dimension

- **Input**: H × W × C
- **Output**: 1 × 1 × C

**Use case:** Replace fully connected layers at the end

---"""),

    cell("markdown", """## Summary

### 🎯 Key Points

1. **Max Pooling**: Keeps strongest activations
2. **Average Pooling**: Smooth aggregation
3. **Output Size**: $(H - F) / S + 1$ (same as convolution)
4. **No Parameters**: Pooling has NO learnable weights!

### 🔮 Next

**Notebook 7:** Complete CNN Architecture (putting it all together)

---

*Week 10 - Deep Neural Network Architectures (21CSE558T)*
*SRM University - M.Tech Program*""")
]

# NOTEBOOK 7: Complete CNN Architecture
nb7 = [
    cell("markdown", """# Notebook 7: Complete CNN Architecture

**Week 10 - Module 4: CNN Basics**
**DO3 (October 27, 2025) - Saturday**
**Duration:** 25-30 minutes

## Learning Objectives

1. ✅ **Design** complete CNN architectures
2. ✅ **Trace** feature map dimensions through network
3. ✅ **Calculate** parameter counts
4. ✅ **Build** simple CNN in Keras (preview for Tutorial T10)

---"""),

    cell("code", """import numpy as np
import tensorflow as tf
from tensorflow import keras

print(f"TensorFlow version: {tf.__version__}")
print("✅ Setup complete!")"""),

    cell("markdown", """## 1. CNN Architecture Components

A complete CNN has:

1. **Input Layer**: Image (H × W × C)
2. **Convolutional Blocks**: [Conv → ReLU → Pool] × N
3. **Flattening**: Convert to 1D vector
4. **Fully Connected Layers**: Classification
5. **Output Layer**: Softmax for probabilities

### Standard Pattern:

```
Input (28×28×1)
    ↓
[Conv 3×3, 32 filters] → ReLU → MaxPool 2×2
    ↓ (14×14×32)
[Conv 3×3, 64 filters] → ReLU → MaxPool 2×2
    ↓ (7×7×64)
Flatten → 3136
    ↓
Dense(128) → ReLU
    ↓
Dense(10) → Softmax
    ↓
Output (10 classes)
```

---"""),

    cell("markdown", """## 2. Dimension Tracing Example

Let's trace dimensions through a simple CNN for MNIST (28×28 grayscale images).

**Architecture:**

| Layer | Operation | Output Shape | Parameters |
|-------|-----------|--------------|------------|
| Input | - | (28, 28, 1) | 0 |
| Conv1 | 3×3, 32 filters, stride=1, padding=same | (28, 28, 32) | 320 |
| MaxPool1 | 2×2, stride=2 | (14, 14, 32) | 0 |
| Conv2 | 3×3, 64 filters, stride=1, padding=same | (14, 14, 64) | 18,496 |
| MaxPool2 | 2×2, stride=2 | (7, 7, 64) | 0 |
| Flatten | - | (3136,) | 0 |
| Dense1 | 128 units | (128,) | 401,536 |
| Dense2 | 10 units | (10,) | 1,290 |

**Total Parameters:** 421,642

---"""),

    cell("code", """# Build the exact architecture above
model = keras.Sequential([
    # Conv Block 1
    keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),

    # Conv Block 2
    keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    keras.layers.MaxPooling2D((2, 2)),

    # Fully Connected Layers
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.summary()"""),

    cell("markdown", """## 3. Parameter Calculation

### Conv Layer Parameters:

$$
\\text{Params} = (F_h \\times F_w \\times C_{in} + 1) \\times C_{out}
$$

Where:
- $F_h, F_w$ = filter height, width
- $C_{in}$ = input channels
- $C_{out}$ = output channels (number of filters)
- $+1$ = bias term

**Example (Conv1):**
- Filter: 3×3
- Input channels: 1
- Output channels: 32
- Params = $(3 \\times 3 \\times 1 + 1) \\times 32 = 320$

### Dense Layer Parameters:

$$
\\text{Params} = (\\text{input\\_units} + 1) \\times \\text{output\\_units}
$$

**Example (Dense1):**
- Input: 3136 (7×7×64 flattened)
- Output: 128
- Params = $(3136 + 1) \\times 128 = 401,536$

---"""),

    cell("code", """def count_conv_params(filter_size, in_channels, out_channels):
    \"\"\"Calculate conv layer parameters.\"\"\"
    return (filter_size * filter_size * in_channels + 1) * out_channels

def count_dense_params(in_units, out_units):
    \"\"\"Calculate dense layer parameters.\"\"\"
    return (in_units + 1) * out_units

# Verify our calculations
conv1_params = count_conv_params(3, 1, 32)
conv2_params = count_conv_params(3, 32, 64)
dense1_params = count_dense_params(7*7*64, 128)
dense2_params = count_dense_params(128, 10)

print("Parameter Verification:")
print(f"Conv1: {conv1_params:,}")
print(f"Conv2: {conv2_params:,}")
print(f"Dense1: {dense1_params:,}")
print(f"Dense2: {dense2_params:,}")
print(f"Total: {conv1_params + conv2_params + dense1_params + dense2_params:,}")"""),

    cell("markdown", """## 4. Famous CNN Architecture: LeNet-5

**LeNet-5** (Yann LeCun, 1998) - First successful CNN

```
Input (32×32×1)
    ↓
Conv 5×5, 6 filters → Tanh → AvgPool 2×2
    ↓
Conv 5×5, 16 filters → Tanh → AvgPool 2×2
    ↓
Flatten
    ↓
Dense(120) → Tanh
    ↓
Dense(84) → Tanh
    ↓
Dense(10) → Softmax
```

**Key Innovations:**
- Hierarchical feature learning
- Weight sharing (convolution)
- Pooling for translation invariance

---"""),

    cell("markdown", """## 5. CNN vs MLP: Parameter Comparison

**Same task:** Classify 28×28 MNIST images (10 classes)

### MLP Approach:
- Flatten input: 784 units
- Hidden layer: 128 units
- Output: 10 units
- **Total params:** $(784 + 1) \\times 128 + (128 + 1) \\times 10 = 101,770$

### CNN Approach (our model):
- Convolutional layers + Dense layers
- **Total params:** 421,642

**Wait, CNN has MORE parameters?**

Yes, but:
1. CNN learns **hierarchical features**
2. CNN achieves **better accuracy** (98%+ vs 95%)
3. CNN is **translation invariant**
4. With **regularization**, CNN generalizes better

**Modern CNNs** (MobileNet, EfficientNet) have FEWER params than MLPs!

---"""),

    cell("markdown", """## Summary

### 🎯 Key Architecture Principles

1. **Pattern**: [Conv → Activation → Pool] × N → Flatten → Dense
2. **Channels increase**: 1 → 32 → 64 → 128 (doubling)
3. **Spatial size decreases**: 28 → 14 → 7 (pooling)
4. **Parameters**: Mostly in Dense layers (for small CNNs)

### 🔮 Next

**Notebook 8:** 3D Convolution (video, medical imaging)

---

*Week 10 - Deep Neural Network Architectures (21CSE558T)*
*SRM University - M.Tech Program*""")
]

# NOTEBOOK 8: 3D Convolution
nb8 = [
    cell("markdown", """# Notebook 8: 3D Convolution Preview

**Week 10 - Module 4: CNN Basics**
**DO3 (October 27, 2025) - Saturday**
**Duration:** 10-15 minutes

## Learning Objectives

1. ✅ **Understand** difference between 2D and 3D convolution
2. ✅ **Identify** use cases for 3D convolution
3. ✅ **Recognize** when to use 3D vs 2D

---"""),

    cell("code", """import numpy as np
import tensorflow as tf

print("✅ Setup complete!")"""),

    cell("markdown", """## 1. 2D vs 3D Convolution

### 2D Convolution (what we've learned):
- **Input**: H × W × C (height, width, channels)
- **Kernel**: Kh × Kw × C × Filters
- **Output**: H' × W' × Filters
- **Use case**: Images (spatial features)

### 3D Convolution:
- **Input**: D × H × W × C (depth, height, width, channels)
- **Kernel**: Kd × Kh × Kw × C × Filters
- **Output**: D' × H' × W' × Filters
- **Use case**: Videos, 3D medical scans (spatiotemporal features)

---"""),

    cell("markdown", """## 2. When to Use 3D Convolution?

**Applications:**

1. **Video Analysis**
   - Action recognition (sports, surveillance)
   - Gesture recognition
   - Video captioning

2. **Medical Imaging**
   - CT scans (3D body scans)
   - MRI volumes
   - Tumor detection

3. **Climate Science**
   - Weather prediction (3D atmospheric data)
   - Ocean temperature analysis

---"""),

    cell("code", """# Example: 3D Conv for video (5 frames, 64×64 RGB)
from tensorflow.keras import layers

model_3d = tf.keras.Sequential([
    layers.Conv3D(32, (3, 3, 3), activation='relu',
                  input_shape=(5, 64, 64, 3)),  # (frames, h, w, channels)
    layers.MaxPooling3D((1, 2, 2)),  # Pool spatial, not temporal
    layers.Conv3D(64, (3, 3, 3), activation='relu'),
    layers.MaxPooling3D((1, 2, 2)),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])

model_3d.summary()"""),

    cell("markdown", """## 3. 2D Conv on RGB Images (Important Clarification!)

**Common Confusion:**

> "RGB images have 3 channels. Is that 3D convolution?"

**Answer:** NO! It's still 2D convolution.

- **Input**: H × W × 3 (spatial 2D + 3 color channels)
- **Kernel**: Kh × Kw × 3 × Filters
- **Convolution**: Applied spatially (2D), aggregates across channels

**3D convolution** adds a **third spatial dimension** (depth/time).

---"""),

    cell("markdown", """## 4. Parameter Comparison

**2D Conv (32×32 RGB image, 64 filters, 3×3 kernel):**
- Params = $3 \\times 3 \\times 3 \\times 64 + 64 = 1,792$

**3D Conv (16 frames, 32×32 RGB, 64 filters, 3×3×3 kernel):**
- Params = $3 \\times 3 \\times 3 \\times 3 \\times 64 + 64 = 5,248$

**3D convolution is ~3× more expensive!**

---"""),

    cell("markdown", """## Summary

### 🎯 Key Distinctions

1. **2D Conv**: Spatial features (images)
   - Input: H × W × C
   - Kernel: Kh × Kw × C × F

2. **3D Conv**: Spatiotemporal features (videos, 3D scans)
   - Input: D × H × W × C
   - Kernel: Kd × Kh × Kw × C × F

3. **RGB Images**: Still 2D convolution (channels ≠ depth)

### 🔮 Next

**Notebook 9:** Review & Tutorial T10 Preview

---

*Week 10 - Deep Neural Network Architectures (21CSE558T)*
*SRM University - M.Tech Program*""")
]

# NOTEBOOK 9: Review & Tutorial Preview
nb9 = [
    cell("markdown", """# Notebook 9: Review & Tutorial T10 Preview

**Week 10 - Module 4: CNN Basics**
**DO3 (October 27, 2025) - Saturday**
**Duration:** 15-20 minutes

## Objectives

1. ✅ **Review** all concepts from Notebooks 0-8
2. ✅ **Preview** Tutorial T10 (building CNN in Keras)
3. ✅ **Prepare** for hands-on session

---"""),

    cell("markdown", """## 1. Complete Concept Map

```
Week 10 CNN Journey
│
├── Notebook 0: Setup & Prerequisites
│   └── NumPy, Matplotlib, helper functions
│
├── Notebook 1: Convolution Concept & Intuition
│   └── Sliding window, pattern detection
│
├── Notebook 2: 1D Convolution Math & Code
│   └── Signal processing, manual calculation
│
├── Notebook 3: 2D Convolution for Images
│   └── Edge detection, feature maps
│
├── Notebook 4: Convolution Parameters
│   └── Stride, padding, kernel size, output formula
│
├── Notebook 5: Hierarchical Feature Learning
│   └── Edges → textures → parts → objects
│
├── Notebook 6: Pooling Layers
│   └── Max pooling, average pooling, dimension reduction
│
├── Notebook 7: Complete CNN Architecture
│   └── [Conv→ReLU→Pool]×N → Flatten → Dense
│
└── Notebook 8: 3D Convolution Preview
    └── Videos, medical imaging, spatiotemporal
```

---"""),

    cell("markdown", """## 2. Key Formulas Cheat Sheet

### Output Dimension Formula:
$$
\\text{Output} = \\left\\lfloor \\frac{W - F + 2P}{S} \\right\\rfloor + 1
$$

### Convolution Parameter Count:
$$
\\text{Params} = (F_h \\times F_w \\times C_{in} + 1) \\times C_{out}
$$

### Dense Parameter Count:
$$
\\text{Params} = (\\text{input\\_units} + 1) \\times \\text{output\\_units}
$$

### Same Padding Formula:
$$
P = \\frac{F - 1}{2} \\quad \\text{(for stride = 1)}
$$

---"""),

    cell("markdown", """## 3. Connection to Week 9

| Week 9 (Manual) | Week 10 (Learned) |
|-----------------|-------------------|
| LBP (Local Binary Patterns) | Conv Layer 1 (edges) |
| GLCM (Texture) | Conv Layer 2 (textures) |
| Shape Features | Conv Layer 3 (parts) |
| Manually designed | Automatically learned! |

**Key Insight:** CNNs learn the feature extractors that we manually designed in Week 9!

---"""),

    cell("markdown", """## 4. Tutorial T10 Preview

**Tomorrow's Tutorial (DO4, Oct 29, Monday):**

### Task: Build CNN for Fashion-MNIST Classification

**Steps:**

1. **Load Data**
   ```python
   from tensorflow.keras.datasets import fashion_mnist
   (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
   ```

2. **Build CNN**
   ```python
   model = keras.Sequential([
       Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
       MaxPooling2D((2,2)),
       Conv2D(64, (3,3), activation='relu'),
       MaxPooling2D((2,2)),
       Flatten(),
       Dense(128, activation='relu'),
       Dense(10, activation='softmax')
   ])
   ```

3. **Train & Evaluate**
   ```python
   model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
   model.fit(X_train, y_train, epochs=5, validation_split=0.2)
   ```

4. **Visualize Filters** (Bonus)

---"""),

    cell("markdown", """## 5. Self-Assessment Questions

### Question 1: Dimension Calculation
Input: 32×32×3, Filter: 5×5, Stride: 1, Padding: 0
- What is output dimension?
- **Answer**: $(32 - 5 + 0)/1 + 1 = 28$ → **28×28**

### Question 2: Parameter Count
Conv layer: 3×3 kernel, 64 input channels, 128 output channels
- How many parameters?
- **Answer**: $(3 \\times 3 \\times 64 + 1) \\times 128 = 73,856$

### Question 3: Architecture Design
Design a CNN for 64×64 RGB images, 5 classes:
- **Suggested**:
  - Conv(32) → Pool → Conv(64) → Pool → Flatten → Dense(128) → Dense(5)

---"""),

    cell("markdown", """## 6. Common Pitfalls to Avoid

### ❌ Mistake #1: Forgetting to Normalize
```python
# Wrong
model.fit(X_train, y_train)  # Values in [0, 255]

# Correct
X_train = X_train / 255.0  # Normalize to [0, 1]
model.fit(X_train, y_train)
```

### ❌ Mistake #2: Wrong Input Shape
```python
# Wrong
input_shape=(28, 28)  # Missing channel dimension

# Correct
input_shape=(28, 28, 1)  # Grayscale
# or
input_shape=(28, 28, 3)  # RGB
```

### ❌ Mistake #3: Dimension Mismatch
```python
# Wrong: Output of Flatten doesn't match Dense input
Flatten() → Dense(10)  # Expects specific input size

# Correct: Keras handles this automatically in Sequential
```

---"""),

    cell("markdown", """## 7. Preparation Checklist for Tutorial T10

- [ ] Review convolution operation (Notebooks 1-3)
- [ ] Understand stride, padding, pooling (Notebooks 4, 6)
- [ ] Know how to calculate dimensions (Notebook 4)
- [ ] Understand CNN architecture pattern (Notebook 7)
- [ ] Have TensorFlow/Keras installed
- [ ] Familiar with Fashion-MNIST dataset

---"""),

    cell("markdown", """## Summary

### 🎯 What We Accomplished

**Notebooks 0-9:**
1. Convolution intuition and mathematics
2. 1D and 2D convolution implementations
3. Stride, padding, kernel size parameters
4. Hierarchical feature learning
5. Pooling mechanisms
6. Complete CNN architectures
7. 3D convolution overview

**Tomorrow (Tutorial T10):**
- Build your first CNN in Keras!
- Train on Fashion-MNIST
- Achieve 90%+ accuracy
- Visualize learned filters

### 🎓 Final Thoughts

**From Week 9 to Week 10:**
- Week 9: Manual feature design (LBP, GLCM)
- Week 10: Automatic feature learning (CNNs)

**The Power of CNNs:**
- Learn hierarchical features
- Translation invariant
- State-of-the-art image recognition

---

## Next Steps

1. **Review** any confusing notebooks
2. **Practice** dimension calculations
3. **Install** TensorFlow/Keras if needed
4. **Bring questions** to Tutorial T10!

---

**See you at Tutorial T10 (DO4, Oct 29, Monday)!**

---

*Week 10 - Deep Neural Network Architectures (21CSE558T)*
*SRM University - M.Tech Program*

---

## 🎉 Congratulations!

You've completed all 9 notebooks on CNN basics. You now understand:
- ✅ How convolution works mathematically
- ✅ Why CNNs are powerful for images
- ✅ How to design CNN architectures
- ✅ Ready to build your first CNN!

**Happy Learning! 🚀**""")
]

# Save notebooks 6-9
save_nb('06_pooling_layers.ipynb', nb6)
save_nb('07_complete_cnn_architecture.ipynb', nb7)
save_nb('08_3d_convolution_preview.ipynb', nb8)
save_nb('09_review_and_tutorial_preview.ipynb', nb9)

print("\n" + "="*60)
print("✅ ALL NOTEBOOKS GENERATED SUCCESSFULLY!")
print("="*60)
print("\nGenerated notebooks:")
print("  • 04_convolution_parameters.ipynb")
print("  • 05_hierarchical_feature_learning.ipynb")
print("  • 06_pooling_layers.ipynb")
print("  • 07_complete_cnn_architecture.ipynb")
print("  • 08_3d_convolution_preview.ipynb")
print("  • 09_review_and_tutorial_preview.ipynb")
print("\nTotal: 10 notebooks (00-09)")
print("="*60)
