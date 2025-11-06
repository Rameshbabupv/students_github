#!/usr/bin/env python3
"""
Generate Famous CNN Architectures Jupyter Notebooks
Course: 21CSE558T - Deep Neural Network Architectures
Module 4, Week 11, DO3 Nov-1 Saturday
"""

import json
import os

# Base notebook structure
def create_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
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
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

def markdown_cell(content):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": content.split('\n')
    }

def code_cell(content):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": content.split('\n')
    }

# Notebook 1: LeNet-5
notebook1_cells = [
    markdown_cell("""# 01 - LeNet-5: The Pioneer (1998)

**Course:** 21CSE558T - Deep Neural Network Architectures
**Module 4:** CNNs (Week 2 of 3)
**Date:** November 1, 2025
**Duration:** 30-40 minutes

---

## Learning Objectives

1. Understand LeNet-5 architecture and its historical significance
2. Implement LeNet-5 from scratch using Keras
3. Train on MNIST and analyze performance
4. Compare with modern CNN techniques
5. Recognize LeNet's limitations and legacy

---

## Story: Character: Dr. Yann LeCun's Check Reading Problem

**1998, AT&T Bell Labs:** Banks process millions of handwritten checks daily. Manual reading is slow and expensive.

**Character: Dr. Yann LeCun** (real scientist) developed LeNet-5:
- Read handwritten digits automatically
- 99% accuracy (superhuman on MNIST!)
- Deployed in banks worldwide
- Processed millions of checks per day

**This was the birth of practical computer vision with CNNs.**

---"""),

    code_cell("""# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, AveragePooling2D, Flatten, Dense, Activation
)
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")"""),

    markdown_cell("""## Part 1: Load and Explore MNIST Dataset

**MNIST:**
- 70,000 handwritten digit images
- 28×28 grayscale pixels
- 10 classes (digits 0-9)
- Standard benchmark for decades"""),

    code_cell("""# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f"Training samples: {x_train.shape[0]}")
print(f"Test samples: {x_test.shape[0]}")
print(f"Image shape: {x_train.shape[1:]}"))

# Visualize samples
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(x_train[i], cmap='gray')
    ax.set_title(f'Label: {y_train[i]}')
    ax.axis('off')
plt.suptitle('MNIST Sample Images', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()"""),

    markdown_cell("""## Part 2: Data Preprocessing

LeNet-5 expects:
- 32×32 images (MNIST is 28×28, needs padding)
- Normalized pixel values [0, 1]
- One-hot encoded labels"""),

    code_cell("""# Preprocess data
# Pad 28×28 to 32×32 (LeNet-5 original input)
x_train_padded = np.pad(x_train, ((0,0), (2,2), (2,2)), mode='constant')
x_test_padded = np.pad(x_test, ((0,0), (2,2), (2,2)), mode='constant')

# Normalize to [0, 1]
x_train_norm = x_train_padded.astype('float32') / 255.0
x_test_norm = x_test_padded.astype('float32') / 255.0

# Add channel dimension
x_train_final = x_train_norm[..., np.newaxis]
x_test_final = x_test_norm[..., np.newaxis]

# One-hot encode labels
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

print(f"Final training shape: {x_train_final.shape}")
print(f"Final test shape: {x_test_final.shape}")"""),

    markdown_cell("""## Part 3: LeNet-5 Architecture

**Original LeNet-5 (1998):**
```
Input: 32×32×1
  ↓
C1: Conv(6, 5×5) → Sigmoid → AvgPool(2×2)
  ↓
C2: Conv(16, 5×5) → Sigmoid → AvgPool(2×2)
  ↓
C3: Conv(120, 5×5) → Sigmoid
  ↓
F6: FC(84) → Sigmoid
  ↓
Output: FC(10) → Softmax
```

**Total Parameters:** ~60,000"""),

    code_cell("""# Build LeNet-5 (Original with Sigmoid)
def build_lenet5_original():
    model = Sequential([
        # C1: Convolution + Sigmoid + Pooling
        Conv2D(6, (5,5), activation='sigmoid', input_shape=(32, 32, 1)),
        AveragePooling2D((2,2), strides=2),

        # C2: Convolution + Sigmoid + Pooling
        Conv2D(16, (5,5), activation='sigmoid'),
        AveragePooling2D((2,2), strides=2),

        # C3: Convolution + Sigmoid (acts like FC)
        Conv2D(120, (5,5), activation='sigmoid'),

        # Flatten
        Flatten(),

        # F6: Fully Connected + Sigmoid
        Dense(84, activation='sigmoid'),

        # Output: Fully Connected + Softmax
        Dense(10, activation='softmax')
    ], name='LeNet5_Original')

    return model

model_original = build_lenet5_original()
model_original.summary()

# Count parameters
total_params = model_original.count_params()
print(f"\\nTotal parameters: {total_params:,}")"""),

    markdown_cell("""## Part 4: Modern LeNet-5 (with ReLU)

**Problem with Sigmoid:**
- Vanishing gradients
- Slow convergence
- Saturation issues

**Modern version:** Replace Sigmoid with ReLU"""),

    code_cell("""# Build LeNet-5 (Modern with ReLU)
def build_lenet5_modern():
    model = Sequential([
        # C1: Convolution + ReLU + Pooling
        Conv2D(6, (5,5), activation='relu', input_shape=(32, 32, 1)),
        AveragePooling2D((2,2), strides=2),

        # C2: Convolution + ReLU + Pooling
        Conv2D(16, (5,5), activation='relu'),
        AveragePooling2D((2,2), strides=2),

        # C3: Convolution + ReLU
        Conv2D(120, (5,5), activation='relu'),

        # Flatten
        Flatten(),

        # F6: Fully Connected + ReLU
        Dense(84, activation='relu'),

        # Output: Fully Connected + Softmax
        Dense(10, activation='softmax')
    ], name='LeNet5_Modern')

    return model

model_modern = build_lenet5_modern()
model_modern.summary()"""),

    markdown_cell("""## Part 5: Training Comparison

Train both versions and compare:
- Original (Sigmoid)
- Modern (ReLU)"""),

    code_cell("""# Compile both models
model_original.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model_modern.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train original (Sigmoid) - 10 epochs
print("=" * 60)
print("Training LeNet-5 Original (Sigmoid)")
print("=" * 60)
history_original = model_original.fit(
    x_train_final, y_train_cat,
    validation_split=0.1,
    epochs=10,
    batch_size=128,
    verbose=1
)

# Train modern (ReLU) - 10 epochs
print("\\n" + "=" * 60)
print("Training LeNet-5 Modern (ReLU)")
print("=" * 60)
history_modern = model_modern.fit(
    x_train_final, y_train_cat,
    validation_split=0.1,
    epochs=10,
    batch_size=128,
    verbose=1
)"""),

    markdown_cell("""## Part 6: Results Comparison"""),

    code_cell("""# Compare training curves
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Accuracy
axes[0].plot(history_original.history['accuracy'], 'b-', label='Train (Sigmoid)', linewidth=2)
axes[0].plot(history_original.history['val_accuracy'], 'b--', label='Val (Sigmoid)', linewidth=2)
axes[0].plot(history_modern.history['accuracy'], 'g-', label='Train (ReLU)', linewidth=2)
axes[0].plot(history_modern.history['val_accuracy'], 'g--', label='Val (ReLU)', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Loss
axes[1].plot(history_original.history['loss'], 'b-', label='Train (Sigmoid)', linewidth=2)
axes[1].plot(history_original.history['val_loss'], 'b--', label='Val (Sigmoid)', linewidth=2)
axes[1].plot(history_modern.history['loss'], 'g-', label='Train (ReLU)', linewidth=2)
axes[1].plot(history_modern.history['val_loss'], 'g--', label='Val (ReLU)', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.suptitle('LeNet-5: Sigmoid vs ReLU', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Final test accuracy
test_loss_orig, test_acc_orig = model_original.evaluate(x_test_final, y_test_cat, verbose=0)
test_loss_mod, test_acc_mod = model_modern.evaluate(x_test_final, y_test_cat, verbose=0)

print("=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)
print(f"LeNet-5 Original (Sigmoid): {test_acc_orig:.2%}")
print(f"LeNet-5 Modern (ReLU):      {test_acc_mod:.2%}")
print(f"Improvement:                +{(test_acc_mod - test_acc_orig):.2%}")
print("=" * 60)"""),

    markdown_cell("""## Summary: Key Takeaways

### LeNet-5 Innovations (1998)
- ✅ First practical CNN for vision
- ✅ Hierarchical feature learning
- ✅ End-to-end trainable
- ✅ Proved CNNs work for real problems

### Limitations
- ❌ Sigmoid activation (vanishing gradients)
- ❌ Average pooling (less sharp than max)
- ❌ Small capacity (60K params)
- ❌ Limited to simple images (28×28 grayscale)

### Legacy
- Conv → Pool pattern (still used!)
- Inspired AlexNet (2012)
- Foundation of modern CNNs

### Modern Improvements
- ReLU instead of Sigmoid (6× faster)
- Max pooling instead of average
- Batch normalization
- Dropout regularization
- Data augmentation

**LeNet-5 was the seed. The revolution would come 14 years later with AlexNet!**

**Next:** Notebook 02 - AlexNet (2012 breakthrough)""")
]

notebook1 = create_notebook(notebook1_cells)

# Save Notebook 1
output_dir = "/Users/rameshbabu/data/projects/srm/lectures/Deep_Neural_Network_Architectures/course_planning/weekly_plans/week11-12-module4-cnn-layers/Do3-Nov-1-Saturday/wip/notebooks"
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, "01_lenet5_implementation.ipynb"), 'w') as f:
    json.dump(notebook1, f, indent=2)

print("✅ Created: 01_lenet5_implementation.ipynb")
print(f"   Cells: {len(notebook1_cells)}")
print(f"   Size: ~{len(json.dumps(notebook1)) // 1024}KB")
