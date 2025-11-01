# 7 SAQ Question Bank for Formative Test 2 (FT2)

**Course:** 21CSE558T - Deep Neural Network Architectures
**Coverage:** Modules 3-4 (Image Processing & CNNs)
**Format:** 5 marks each
**Total Questions:** 7 (3 from Module 3, 4 from Module 4)
**Date:** November 14, 2025

---

## Module 3: Image Processing & Deep Neural Networks (3 questions)

### **Q1.** Feature Extraction Comparison

You are building an image classification system and need to choose between traditional feature extraction methods (like LBP or GLCM) and deep learning automatic feature extraction. Explain the key differences and when you would choose each approach.

**Module:** 3 | **Week:** 7-8 | **Difficulty:** Moderate | **CO:** CO-3

**Expected Answer (5 marks):**

**Traditional Feature Extraction (LBP, GLCM):**
- Manually designed features based on domain knowledge
- Computationally efficient and interpretable
- Works well with small datasets
- Limited to predefined patterns (texture, edges, shapes)
- Requires domain expertise to select appropriate features

**Deep Learning Automatic Feature Extraction:**
- Learns features automatically from data through training
- Can discover complex, hierarchical patterns
- Requires large datasets and computational resources
- End-to-end learning without manual feature engineering
- Better generalization for complex tasks

**When to choose Traditional:**
- Small datasets (few hundred images)
- Limited computational resources
- Need interpretable features
- Well-defined texture/shape problems

**When to choose Deep Learning:**
- Large datasets (thousands+ images)
- Complex patterns beyond manual feature design
- GPU resources available
- State-of-the-art accuracy required

**Marking Scheme:**
- Explanation of traditional features (1.5 marks)
- Explanation of deep learning features (1.5 marks)
- When to choose traditional (1 mark)
- When to choose deep learning (1 mark)

---

### **Q2.** Image Segmentation Techniques

You have a medical image dataset where you need to separate tumor regions from healthy tissue. Compare thresholding-based segmentation with region-based segmentation approaches. Explain which would be more suitable for this task and why.

**Module:** 3 | **Week:** 7-8 | **Difficulty:** Moderate | **CO:** CO-3

**Expected Answer (5 marks):**

**Thresholding-based Segmentation:**
- Separates pixels based on intensity values
- Simple and fast (Otsu's method for automatic threshold)
- Works well when tumor has distinct intensity from background
- Limitation: Struggles with varying illumination or overlapping intensity ranges
- Example: Binary thresholding creates two classes (tumor/not tumor)

**Region-based Segmentation:**
- Groups pixels based on similarity criteria (intensity, color, texture)
- Methods: Region growing, region splitting/merging, watershed
- Considers spatial coherence and connectivity
- Better handles gradual transitions and complex boundaries
- More robust to noise than simple thresholding

**For Medical Tumor Segmentation:**

**Region-based is more suitable because:**
- Tumors have complex, irregular boundaries
- Intensity may vary within tumor region
- Background tissue has varying intensity (not uniform)
- Region growing can start from seed point in tumor
- Spatial coherence ensures connected tumor region
- Can incorporate texture and boundary information

**However, thresholding may work if:**
- Tumor has very distinct intensity (high contrast)
- Image quality is consistent
- Used as pre-processing before region-based refinement

**Marking Scheme:**
- Thresholding explanation (1 mark)
- Region-based explanation (1.5 marks)
- Suitability for medical imaging (1.5 marks)
- Justification with reasoning (1 mark)

---

### **Q3.** Edge Detection Method Selection

You are processing images from two different scenarios: (1) Low-noise indoor photographs, (2) High-noise outdoor surveillance footage. Explain which edge detection method (Sobel vs. Canny) you would choose for each scenario and justify your choices.

**Module:** 3 | **Week:** 7 | **Difficulty:** Easy | **CO:** CO-3

**Expected Answer (5 marks):**

**Sobel Edge Detector:**
- Uses gradient-based approach with two 3×3 kernels (Gx, Gy)
- Simple, fast, computationally efficient
- Detects edges by calculating gradient magnitude
- Produces thick edges (multiple pixels wide)
- Sensitive to noise (no built-in noise reduction)
- Good for real-time applications

**Canny Edge Detector:**
- Multi-stage algorithm: Gaussian smoothing → gradient → non-maximum suppression → hysteresis thresholding
- Built-in Gaussian blur for noise reduction
- Produces thin, well-localized edges (single pixel wide)
- Uses double thresholding to reduce false edges
- More robust to noise but computationally expensive

**Scenario 1: Low-noise indoor photographs**
- **Choice: Sobel**
- **Justification:** Since images are low-noise, Sobel's simplicity and speed are advantageous. Canny's noise reduction overhead is unnecessary. Sobel's faster processing suitable for batch processing of clean images.

**Scenario 2: High-noise outdoor surveillance**
- **Choice: Canny**
- **Justification:** Surveillance footage has noise from weather, lighting variations, compression artifacts. Canny's Gaussian smoothing reduces noise before edge detection. Hysteresis thresholding eliminates weak false edges from noise. Better edge localization despite computational cost.

**Marking Scheme:**
- Sobel explanation (1 mark)
- Canny explanation (1.5 marks)
- Scenario 1 choice and justification (1 mark)
- Scenario 2 choice and justification (1.5 marks)

---

---

## Module 4: Convolutional Neural Networks & Transfer Learning (4 questions)

### **Q4.** Batch Normalization Placement in CNNs

You are building a CNN and see two different implementations online. Implementation A places BatchNormalization AFTER the activation function (Conv2D → ReLU → BatchNorm), while Implementation B places it BEFORE activation (Conv2D → BatchNorm → ReLU). Explain which is the modern best practice and why it matters for training.

**Module:** 4 | **Week:** 11 | **Difficulty:** Moderate | **CO:** CO-4

**Expected Answer (5 marks):**

**Implementation A (Conv → ReLU → BatchNorm):**
- Old practice from early BatchNorm papers
- Normalizes after activation (post-activation values)
- ReLU zeros out negative values before normalization
- Less effective because normalization acts on already transformed values

**Implementation B (Conv → BatchNorm → ReLU):**
- Modern best practice
- Normalizes pre-activation values (before ReLU)
- BatchNorm operates on full range of values (including negatives)
- More effective normalization of the linear transformation output

**Why Implementation B is Better:**

1. **Better Gradient Flow:**
   - BatchNorm normalizes the full distribution before activation
   - Prevents activation saturation issues
   - Gradients flow more uniformly through the network

2. **Theoretical Justification:**
   - BatchNorm was designed to normalize covariate shift in layer inputs
   - Pre-activation values are the actual inputs to the non-linearity
   - Normalizing before activation addresses the root cause

3. **Empirical Performance:**
   - Modern architectures (ResNet, DenseNet, EfficientNet) use Conv → BN → ReLU
   - Faster convergence and better final accuracy
   - More stable training with higher learning rates

4. **Practical Impact:**
   - Implementation A: Slower convergence, may need lower learning rate
   - Implementation B: Faster training, better regularization effect

**Correct Pattern:**
```
Conv2D(32, (3,3), padding='same')  # No activation!
BatchNormalization()
Activation('relu')
```

**Marking Scheme:**
- Implementation A explanation (1 mark)
- Implementation B explanation (1 mark)
- Why B is better (2 marks)
- Practical impact (1 mark)

---

### **Q5.** Dropout Placement Strategy in CNNs

You build a CNN for CIFAR-10 classification with three convolutional blocks. Your colleague suggests placing Dropout(0.5) after every layer including after the final Dense(10) output layer. Explain what is wrong with this approach and describe the correct dropout placement strategy.

**Module:** 4 | **Week:** 11 | **Difficulty:** Easy | **CO:** CO-4

**Expected Answer (5 marks):**

**What is Wrong:**

1. **Dropout After Output Layer is WRONG:**
   - Dropout after Dense(10, softmax) makes predictions random!
   - During inference, output probabilities become unreliable
   - Final layer should NEVER have dropout after it
   - Defeats the purpose of having a stable classification output

2. **Same Dropout Rate (0.5) Everywhere is WRONG:**
   - Early layers learn low-level features (edges, textures) - need less regularization
   - Later layers learn high-level features - need more regularization
   - Using 0.5 everywhere over-regularizes early layers

**Correct Dropout Placement Strategy:**

**Progressive Dropout Rates:**
- After first pooling: Dropout(0.2) - light regularization
- After second pooling: Dropout(0.3) - moderate regularization
- Before final dense layer: Dropout(0.5) - heavy regularization
- **NEVER** after output layer

**Example Correct Architecture:**
```
Block 1:
  Conv2D → BatchNorm → ReLU
  Conv2D → BatchNorm → ReLU
  MaxPooling2D
  Dropout(0.2)  ← Light

Block 2:
  Conv2D → BatchNorm → ReLU
  Conv2D → BatchNorm → ReLU
  MaxPooling2D
  Dropout(0.3)  ← Moderate

Block 3:
  Conv2D → BatchNorm → ReLU
  GlobalAveragePooling2D

Output:
  Dropout(0.5)  ← Heavy (before output, not after!)
  Dense(10, activation='softmax')
  (NO DROPOUT HERE!)
```

**Why Progressive Rates Work:**
- Early features are general (useful across tasks) - preserve them
- Late features are task-specific - regularize heavily to prevent overfitting
- Gradual increase prevents under-regularization and over-regularization

**Marking Scheme:**
- Identifying dropout after output error (1.5 marks)
- Identifying same rate everywhere error (1 mark)
- Correct progressive strategy (1.5 marks)
- Justification (1 mark)

---

### **Q6.** Data Augmentation for Image Classification

You train a CNN on CIFAR-10 (airplanes, cars, animals) with the following augmentation: rotation_range=180, vertical_flip=True, horizontal_flip=True. Your validation accuracy is poor despite good training accuracy. Explain what is wrong with this augmentation strategy and propose appropriate augmentation for CIFAR-10.

**Module:** 4 | **Week:** 11 | **Difficulty:** Moderate | **CO:** CO-4

**Expected Answer (5 marks):**

**What is Wrong:**

1. **rotation_range=180 (±180°) is WRONG:**
   - Airplanes upside-down don't look like airplanes
   - Cars rotated 180° are unrealistic
   - Model learns unrealistic variations
   - Confuses the model during training

2. **vertical_flip=True is WRONG for CIFAR-10:**
   - Airplanes don't fly upside-down in normal scenarios
   - Flipped animals/vehicles change semantic meaning
   - Introduces label noise (flipped airplane ≠ airplane in real world)
   - Only horizontal flip makes sense (left/right mirror is realistic)

**Why Validation Accuracy is Poor:**
- Training data includes unrealistic augmentations
- Model learns to recognize upside-down objects
- Real validation/test images are right-side up
- Mismatch between augmented training and real validation data
- Model wastes capacity learning impossible variations

**Appropriate Augmentation for CIFAR-10:**

```python
ImageDataGenerator(
    rotation_range=15,        # ±15° slight angle variations (realistic)
    width_shift_range=0.1,    # 10% horizontal position shift
    height_shift_range=0.1,   # 10% vertical position shift
    horizontal_flip=True,     # ✓ Mirror is realistic
    vertical_flip=False,      # ✗ Upside-down not realistic
    zoom_range=0.1            # 90-110% zoom (distance variation)
)
```

**Justification:**
- **rotation_range=15:** Small rotations mimic camera angle variations
- **width/height_shift:** Objects can appear at different positions
- **horizontal_flip:** Left-facing vs right-facing airplane is valid
- **zoom_range:** Objects at different distances
- **NO vertical_flip:** Preserves semantic meaning

**General Augmentation Principles:**
- Augmentation should reflect real-world variations
- Don't create label-changing transformations
- Consider domain semantics (what makes sense for the objects)
- Apply ONLY to training data, NEVER to validation/test

**Marking Scheme:**
- Identifying rotation error (1 mark)
- Identifying vertical flip error (1 mark)
- Appropriate augmentation strategy (2 marks)
- Justification and principles (1 mark)

---

### **Q7.** Transfer Learning Strategy

You have a dataset of 500 medical X-ray images to classify lung diseases (4 classes). You consider two approaches: (A) Train a CNN from scratch, (B) Use transfer learning with VGG16 pre-trained on ImageNet. Explain which approach is more suitable and describe the transfer learning strategy you would use.

**Module:** 4 | **Week:** 12 (or 11 if covered) | **Difficulty:** Moderate | **CO:** CO-4

**Expected Answer (5 marks):**

**Comparing Approaches:**

**Approach A: Train from Scratch**
- Requires large dataset (typically 10,000+ images per class)
- You have only 500 images (125 per class) - too small!
- Risk of severe overfitting
- Needs extensive training time
- May not learn meaningful features with limited data

**Approach B: Transfer Learning with VGG16**
- Leverages features learned from 1.2M ImageNet images
- Works well with small datasets (hundreds to thousands)
- Pre-trained features (edges, textures, shapes) are universal
- Faster training (only fine-tune top layers)
- Better generalization

**Which is More Suitable:**
- **Transfer Learning (Approach B)** is clearly better
- 500 images is too small for training from scratch
- Pre-trained features from natural images transfer well to medical images
- Low-level features (edges, textures) are domain-agnostic

**Transfer Learning Strategy:**

**Step 1: Load Pre-trained VGG16**
```python
base_model = tf.keras.applications.VGG16(
    weights='imagenet',
    include_top=False,  # Remove ImageNet classifier
    input_shape=(224, 224, 3)
)
```

**Step 2: Freeze Base Layers Initially**
```python
base_model.trainable = False  # Freeze all VGG16 layers
```
- Preserves pre-trained features
- Only train new classification head

**Step 3: Add Custom Classification Head**
```python
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(4, activation='softmax')  # 4 classes
])
```

**Step 4: Initial Training (Frozen Base)**
- Train only the new dense layers
- Use moderate learning rate (0.001)
- Train for 10-20 epochs until validation accuracy plateaus

**Step 5: Fine-tuning (Optional)**
- Unfreeze top few layers of VGG16
- Use very low learning rate (0.0001)
- Train for additional 5-10 epochs
- Adapts features to X-ray domain

**Why This Works:**
- Early VGG16 layers: Generic features (edges, textures) - keep frozen
- Later VGG16 layers: Specific to natural images - fine-tune
- Custom head: Learns X-ray disease patterns
- Small dataset handled well with pre-trained features

**Expected Outcome:**
- 70-85% accuracy with 500 images
- Much better than from-scratch (<60% likely)
- Reduced overfitting with frozen features

**Marking Scheme:**
- Approach comparison (1 mark)
- Choosing transfer learning with justification (1 mark)
- Transfer learning strategy steps (2 marks)
- Fine-tuning explanation (1 mark)

---

---

## Summary Statistics

### Distribution:
- **Module 3:** 3 questions
  - Feature extraction comparison (Q1)
  - Segmentation techniques (Q2)
  - Edge detection selection (Q3)

- **Module 4:** 4 questions
  - BatchNorm placement (Q4)
  - Dropout placement strategy (Q5)
  - Data augmentation (Q6)
  - Transfer learning strategy (Q7)

### Difficulty:
- Easy: 2 questions (Q3, Q5)
- Moderate: 5 questions (Q1, Q2, Q4, Q6, Q7)
- Difficult: 0 questions

### Key Features:
- ✅ Conceptual focus (light calculations)
- ✅ NO code-based debugging questions
- ✅ CNN-specific applications (Q4, Q5, Q6)
- ✅ Includes Transfer Learning (Q7)
- ✅ Scenario-based explanations (similar to FT1 format)
- ✅ All questions expect flowing paragraph answers with reasoning
- ✅ Complete marking schemes provided

### Course Outcomes Coverage:
- CO-3 (Image Processing): 3 questions
- CO-4 (CNNs & Transfer Learning): 4 questions

---

## Marking Guidelines

**5-mark Question Rubric:**
- **Excellent (5 marks):** Complete explanation with correct reasoning, examples, and practical understanding
- **Good (4 marks):** Mostly correct with minor gaps in explanation or reasoning
- **Satisfactory (3 marks):** Basic understanding shown but missing key points or depth
- **Poor (1-2 marks):** Incomplete or partially incorrect explanation
- **Fail (0 marks):** Completely incorrect or no relevant answer

**Key Assessment Criteria:**
- Understanding of core concepts
- Ability to compare/contrast approaches
- Practical decision-making skills
- Justification with reasoning
- Clear, structured explanations

---

**Last Updated:** October 30, 2025
**Status:** Complete - Ready for FT2 test paper generation
