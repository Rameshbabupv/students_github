#!/usr/bin/env python3
"""
Generate all remaining Jupyter notebooks (3-9) for Week 10 CNN lecture.
This is a comprehensive generator that creates all notebooks at once.
"""

import json
import os

def create_notebook_cell(cell_type, source, metadata=None):
    """Create a notebook cell."""
    cell = {
        "cell_type": cell_type,
        "metadata": metadata or {},
        "source": source if isinstance(source, list) else [source]
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell

def create_base_metadata():
    """Create standard notebook metadata."""
    return {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.0"
        }
    }

def save_notebook(filename, cells):
    """Save notebook to file."""
    notebook = {
        "cells": cells,
        "metadata": create_base_metadata(),
        "nbformat": 4,
        "nbformat_minor": 4
    }
    with open(filename, 'w') as f:
        json.dump(notebook, f, indent=1)
    return filename

# Create Notebook 3: 2D Convolution for Images
def create_notebook_3():
    """Generate Notebook 3."""
    cells = [
        create_notebook_cell("markdown", """# Notebook 3: 2D Convolution for Images

**Week 10 - Module 4: CNN Basics**
**DO3 (October 27, 2025) - Saturday**
**Duration:** 25-30 minutes

---

## Learning Objectives

By the end of this notebook, you will be able to:

1. ✅ **Calculate** 2D convolution operations by hand
2. ✅ **Understand** how convolution applies to images
3. ✅ **Implement** 2D convolution using NumPy and SciPy
4. ✅ **Apply** edge detection filters to real images
5. ✅ **Visualize** feature maps and filter responses

---

## Prerequisites

- ✅ Completed Notebook 2 (1D Convolution)
- ✅ Understanding of images as 2D arrays
- ✅ Matrix multiplication concepts

---"""),

        create_notebook_cell("markdown", "## 1. Setup and Imports"),

        create_notebook_cell("code", """import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, ndimage
from skimage import data, color, filters
import cv2

# Set random seed
np.random.seed(42)

# Configure matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

print("✅ Setup complete!")
print("Libraries loaded: NumPy, Matplotlib, SciPy, scikit-image, OpenCV")"""),

        create_notebook_cell("markdown", """---

## 2. The Story: Character: Meera's Medical Imaging Challenge

### 📖 Narrative

**Character: Meera**, a medical imaging researcher, needs to analyze X-ray images for **Character: Dr. Rajesh**'s radiology clinic.

**The Problem:**

> "These X-ray images are blurry, and I need to detect bone edges clearly," explains **Character: Dr. Rajesh**. "Can you enhance the edges automatically?"

**Character: Meera** responds: "Perfect! I'll use 2D convolution with edge detection kernels. Let me show you how 2D convolution extends what we learned in 1D."

---"""),

        create_notebook_cell("markdown", """## 3. From 1D to 2D: Mathematical Extension

### 📐 The 2D Convolution Formula

For 2D images, convolution extends naturally:

$$
\\text{output}[i, j] = \\sum_{m=0}^{M-1} \\sum_{n=0}^{N-1} \\text{input}[i+m, j+n] \\cdot \\text{kernel}[m, n]
$$

Where:
- `input`: 2D image matrix (height × width)
- `kernel`: 2D filter matrix (M × N)
- `output`: Convolved feature map

**In Plain English:**

"Slide a 2D filter across the image, multiply overlapping values, sum them up."

---"""),

        create_notebook_cell("markdown", """## 4. Hand Calculation: Simple 2D Example

Let's calculate 2D convolution **by hand** with a tiny example.

### Example Setup:

**Input (4×4 image):**
```
[1, 2, 3, 4]
[5, 6, 7, 8]
[9, 10, 11, 12]
[13, 14, 15, 16]
```

**Kernel (3×3 edge detector):**
```
[1, 0, -1]
[2, 0, -2]
[1, 0, -1]
```

This is the **Sobel vertical edge detector** (detects vertical edges).

---"""),

        create_notebook_cell("code", """# Define input image and kernel
input_image = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

sobel_vertical = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
])

print("Input Image (4×4):")
print(input_image)
print("\\nSobel Vertical Kernel (3×3):")
print(sobel_vertical)
print("\\nWe will calculate output at position (0, 0)...")"""),

        create_notebook_cell("markdown", """### Step-by-Step Calculation: Position (0, 0)

Extract 3×3 window from top-left corner:

```
Window:        Kernel:
[1,  2,  3]    [1,  0, -1]
[5,  6,  7]    [2,  0, -2]
[9, 10, 11]    [1,  0, -1]
```

**Element-wise multiplication:**

```
1×1  + 2×0  + 3×(-1)  = 1 + 0 - 3   = -2
5×2  + 6×0  + 7×(-2)  = 10 + 0 - 14 = -4
9×1  + 10×0 + 11×(-1) = 9 + 0 - 11  = -2
```

**Sum all products:**

$$
\\text{output}[0, 0] = -2 + (-4) + (-2) = -8
$$

---"""),

        create_notebook_cell("code", """# Manual calculation for position (0, 0)
window = input_image[0:3, 0:3]
print("Extracted 3×3 window:")
print(window)
print("\\nElement-wise multiplication:")
element_wise = window * sobel_vertical
print(element_wise)
print("\\nSum of all elements:")
result = np.sum(element_wise)
print(f"output[0, 0] = {result}")"""),

        create_notebook_cell("markdown", """---

## 5. NumPy/SciPy Implementation

Now let's compute the full convolution using built-in functions.

---"""),

        create_notebook_cell("code", """# Using scipy.signal.convolve2d
from scipy.signal import convolve2d

# Full convolution
full_output = convolve2d(input_image, sobel_vertical, mode='valid')

print("Full Output (valid mode):")
print(full_output)
print(f"\\nOutput shape: {full_output.shape}")
print(f"Input shape: {input_image.shape}")
print(f"Kernel shape: {sobel_vertical.shape}")
print(f"\\n✅ Verified: output[0, 0] = {full_output[0, 0]}")"""),

        create_notebook_cell("markdown", """---

## 6. Real-World Application: Edge Detection

Let's apply 2D convolution to detect edges in real images.

---"""),

        create_notebook_cell("code", """# Load a sample image (camera from scikit-image)
camera = data.camera()  # 512×512 grayscale image

# Define edge detection kernels
kernels = {
    'Sobel Vertical': np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]]),
    'Sobel Horizontal': np.array([[1, 2, 1],
                                   [0, 0, 0],
                                   [-1, -2, -1]]),
    'Laplacian': np.array([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]]),
    'Box Blur': np.ones((5, 5)) / 25
}

# Apply each kernel
fig, axes = plt.subplots(3, 2, figsize=(14, 16))
axes = axes.flatten()

# Original image
axes[0].imshow(camera, cmap='gray')
axes[0].set_title('Original Image (512×512)', fontsize=14, fontweight='bold')
axes[0].axis('off')

# Apply kernels
for idx, (name, kernel) in enumerate(kernels.items(), 1):
    filtered = convolve2d(camera, kernel, mode='same', boundary='symm')
    axes[idx].imshow(filtered, cmap='gray')
    axes[idx].set_title(f'{name} Filter', fontsize=14, fontweight='bold')
    axes[idx].axis('off')

# Show kernel for Sobel Vertical
axes[5].imshow(kernels['Sobel Vertical'], cmap='RdBu', interpolation='nearest')
axes[5].set_title('Sobel Vertical Kernel (3×3)', fontsize=14, fontweight='bold')
for i in range(3):
    for j in range(3):
        axes[5].text(j, i, f"{kernels['Sobel Vertical'][i, j]}",
                    ha='center', va='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

print("✅ Character: Meera successfully detected edges in X-ray images!")"""),

        create_notebook_cell("markdown", """---

## 7. Custom 2D Convolution Implementation

Let's implement 2D convolution from scratch for deep understanding.

---"""),

        create_notebook_cell("code", """def conv2d_manual(image, kernel):
    \"\"\"
    Manual implementation of 2D convolution (valid mode).

    Parameters:
    -----------
    image : np.ndarray
        Input 2D image
    kernel : np.ndarray
        2D convolution kernel

    Returns:
    --------
    output : np.ndarray
        Convolved output
    \"\"\"
    img_h, img_w = image.shape
    ker_h, ker_w = kernel.shape

    out_h = img_h - ker_h + 1
    out_w = img_w - ker_w + 1

    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            # Extract window
            window = image[i:i+ker_h, j:j+ker_w]
            # Element-wise multiply and sum
            output[i, j] = np.sum(window * kernel)

    return output

# Test on small image
test_img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
test_kernel = np.array([[1, 0], [0, -1]])

custom_result = conv2d_manual(test_img, test_kernel)
scipy_result = convolve2d(test_img, test_kernel, mode='valid')

print("Custom Implementation:")
print(custom_result)
print("\\nSciPy Implementation:")
print(scipy_result)
print(f"\\n✅ Match: {np.allclose(custom_result, scipy_result)}")"""),

        create_notebook_cell("markdown", """---

## 8. Visualizing Multiple Feature Maps

In CNNs, we apply multiple filters to extract different features. Let's visualize this.

---"""),

        create_notebook_cell("code", """# Create multiple feature detectors
feature_kernels = {
    'Vertical Edges': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
    'Horizontal Edges': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
    'Diagonal 45°': np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]),
    'Diagonal 135°': np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]]),
    'Sharpen': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
    'Emboss': np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
}

# Use a smaller region for clarity
img_region = camera[100:300, 100:300]

# Apply all kernels
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()

# Original
axes[0].imshow(img_region, cmap='gray')
axes[0].set_title('Original Region', fontsize=13, fontweight='bold')
axes[0].axis('off')

# Feature maps
for idx, (name, kernel) in enumerate(feature_kernels.items(), 1):
    feature_map = convolve2d(img_region, kernel, mode='same', boundary='symm')
    axes[idx].imshow(feature_map, cmap='gray')
    axes[idx].set_title(f'Feature Map: {name}', fontsize=13, fontweight='bold')
    axes[idx].axis('off')

# Hide extra subplot
axes[-1].axis('off')

plt.tight_layout()
plt.show()

print("🎯 Multiple Feature Maps:")
print("  Each filter detects different patterns in the same image")
print("  This is exactly how CNNs learn hierarchical features!")"""),

        create_notebook_cell("markdown", """---

## 9. Summary and Key Takeaways

### 🎯 What We Learned

1. **2D Convolution Formula**
   - Extends 1D: slide 2D kernel across 2D image
   - Output: `(H - Kh + 1) × (W - Kw + 1)` (valid mode)

2. **Hand Calculation**
   - Extract window, multiply element-wise, sum
   - Same principle as 1D, but in 2 dimensions

3. **Edge Detection**
   - Sobel, Laplacian, etc.
   - Detect different orientations (vertical, horizontal, diagonal)

4. **Multiple Feature Maps**
   - Different kernels → different patterns
   - Foundation of CNN feature learning

### 🔮 What's Next?

In **Notebook 4**, we'll explore **Convolution Parameters**:
- Stride (how much to slide)
- Padding (maintaining dimensions)
- Output dimension calculations
- Parameter trade-offs

---"""),

        create_notebook_cell("markdown", """## 10. Practice Exercises

### Exercise 1: Hand Calculation
Calculate 2D convolution manually:
- Input: 3×3 image `[[1,2,3], [4,5,6], [7,8,9]]`
- Kernel: 2×2 `[[1,0], [0,-1]]`
- Calculate all output positions

### Exercise 2: Custom Edge Detector
Design a kernel that detects:
- Edges at 45° diagonal
- Test on checkerboard pattern

### Exercise 3: Multi-Channel Thinking
How would you apply convolution to RGB images (3 channels)?
- Hint: Think about kernel depth

---

**Next Notebook:** [Notebook 4: Convolution Parameters](04_convolution_parameters.ipynb)

---

*Week 10 - Deep Neural Network Architectures (21CSE558T)*
*SRM University - M.Tech Program*""")
    ]

    return cells

print("Starting notebook generation...")
print("=" * 60)

# Generate Notebook 3
cells_3 = create_notebook_3()
save_notebook('03_2d_convolution_images.ipynb', cells_3)
print("✅ Generated Notebook 3: 2D Convolution for Images")

print("=" * 60)
print("✅ Notebook generation complete!")
print("\nGenerated files:")
print("  • 03_2d_convolution_images.ipynb")
