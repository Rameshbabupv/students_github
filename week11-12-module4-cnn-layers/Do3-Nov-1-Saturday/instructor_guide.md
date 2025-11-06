# Instructor Guide: CNN Practical Sessions

**Course:** 21CSE558T - Deep Neural Network Architectures
**Module 4:** CNNs - Hands-on Implementation
**Sessions:** Saturday, November 1 (2 hours) + Monday, November 3 (1 hour)
**Target:** M.Tech students with basic Python/TensorFlow knowledge

---

## 📋 Table of Contents

1. [Session Overview](#session-overview)
2. [Session 1: Saturday (2 hours)](#session-1-saturday-2-hours)
3. [Session 2: Monday (1 hour)](#session-2-monday-1-hour)
4. [Teaching Tips](#teaching-tips)
5. [Common Student Questions](#common-student-questions)
6. [Troubleshooting](#troubleshooting)
7. [Assessment Guidelines](#assessment-guidelines)

---

## Session Overview

### Learning Objectives

By the end of both sessions, students will:

1. ✅ Understand convolution operations (1D and 2D) from first principles
2. ✅ Build and train CNNs using TensorFlow/Keras
3. ✅ Apply regularization techniques (Dropout, BatchNorm)
4. ✅ Implement data augmentation strategies
5. ✅ Achieve 90%+ accuracy on Fashion-MNIST
6. ✅ Design production-ready CNN architectures

### Materials Provided

**7 Jupyter Notebooks:**
1. `01_convolution_fundamentals.ipynb` - 1D convolution (~30 min)
2. `02_2d_convolution_visual.ipynb` - 2D kernels (~30 min)
3. `03_first_cnn_fashion_mnist.ipynb` - Complete CNN (~40 min)
4. `04_cnn_architecture_experiments.ipynb` - Homework assignment (~2-3 hours)
5. `05_deep_cnn_regularization.ipynb` - Regularization (~30 min)
6. `06_data_augmentation.ipynb` - Data augmentation (~30 min)
7. `07_final_challenge.ipynb` - Final project (~3-4 hours)

**Supporting Documents:**
- `instructor_guide.md` (this file)
- `student_submission_guide.md`
- `requirements.txt`

---

## Session 1: Saturday (2 hours)

### Pre-Session Setup (15 minutes before class)

**Checklist:**
- [ ] Test Google Colab access (students should use Colab)
- [ ] Share notebook links via Google Classroom/Drive
- [ ] Verify GPU availability in Colab
- [ ] Have backup: Download notebooks locally in case of internet issues
- [ ] Test all notebooks run without errors

**Environment Setup for Students:**
```python
# Students should run this first cell in Colab
!pip install tensorflow==2.15.0 matplotlib seaborn scikit-learn
```

---

### Detailed Timeline: Saturday Session

#### **Segment 1: Introduction & 1D Convolution (35 minutes)**

**⏰ Time:** 0:00 - 0:35

**Notebook:** `01_convolution_fundamentals.ipynb`

**Teaching Approach:**
1. **Motivation (5 min):** Start with the analogy: "Convolution = sliding window with multiply & sum"
2. **Live Coding (15 min):** Walk through the `conv1d_simple()` function implementation
   - Explain each line of code
   - Show debugger stepping through the loop
   - Draw on whiteboard: signal, kernel, output
3. **Student Experimentation (10 min):** Let students run Part 6 (Interactive Experiment)
   - Encourage them to modify kernel values
   - Ask: "What happens with negative values in kernel?"
4. **Key Concepts (5 min):** Review stride, padding, output size formula

**Key Teaching Points:**
- ⭐ **Formula:** `Output = (Input + 2*Padding - Kernel) / Stride + 1`
- ⭐ **Kernels detect patterns** (edges, smoothing, sharpening)
- ⭐ **1D → 2D is the same math** (just 2 dimensions)

**Common Questions:**
- *"Why start with 1D?"* → Easier to visualize, same principles apply
- *"What is stride exactly?"* → Step size when sliding the kernel
- *"Why do we need padding?"* → To preserve spatial dimensions

**Expected Outcome:** Students understand convolution as element-wise multiply + sum

---

#### **Segment 2: 2D Convolution Visualization (30 minutes)**

**⏰ Time:** 0:35 - 1:05

**Notebook:** `02_2d_convolution_visual.ipynb`

**Teaching Approach:**
1. **Connection to 1D (3 min):** "Same concept, now on images"
2. **Classic Kernels Demo (15 min):**
   - Run Part 3: Show blur, sharpen, edge detection live
   - Pause after each kernel: "Notice how the image changes"
   - Ask students to predict: "What will Sobel X detect?"
3. **Multiple Feature Maps (7 min):** Part 4 - This is how CNNs work!
   - Emphasize: "CNN learns these kernels automatically"
4. **Student Experimentation (5 min):** Part 7 (Interactive Experiment)
   - Students create custom 3×3 kernels

**Key Teaching Points:**
- ⭐ **Different kernels detect different patterns**
- ⭐ **CNNs learn kernel values during training** (not hand-designed)
- ⭐ **Multiple kernels → Multiple feature maps**

**Visual Focus:** Spend time on the visualizations - they're the most important part!

**Expected Outcome:** Students see the connection between math and visual results

---

#### **Break (10 minutes)** ☕

**⏰ Time:** 1:05 - 1:15

---

#### **Segment 3: First Complete CNN (40 minutes)**

**⏰ Time:** 1:15 - 1:55

**Notebook:** `03_first_cnn_fashion_mnist.ipynb`

**Teaching Approach:**
1. **Dataset Intro (5 min):** Fashion-MNIST overview
2. **Architecture Walkthrough (10 min):**
   - Draw the architecture on whiteboard:
   ```
   Input 28×28×1
      ↓
   Conv(32) → Pool → Conv(64) → Pool → Flatten → Dense(128) → Dense(10)
   ```
   - Explain each layer's purpose
   - Calculate output sizes together
3. **Training Live (10 min):**
   - Run the training (will take 3-5 minutes)
   - While training: Explain batch size, epochs, validation split
   - Show real-time accuracy improving
4. **Results Analysis (10 min):**
   - Review training curves: "See the gap? That's overfitting"
   - Confusion matrix: "Which classes are confused?"
5. **CNN vs MLP Comparison (5 min):** Show why CNNs are better

**Key Teaching Points:**
- ⭐ **Conv → Pool pattern** is standard building block
- ⭐ **Spatial structure preserved** in conv layers
- ⭐ **Parameter sharing** makes CNNs efficient
- ⭐ **~90% accuracy** is achievable with basic CNN

**Important Note:** Training will take 3-5 minutes - use this time to explain concepts!

**Expected Outcome:** Students have trained their first CNN and understand the architecture

---

#### **Segment 4: Assignment & Wrap-up (5 minutes)**

**⏰ Time:** 1:55 - 2:00

**Actions:**
1. Introduce Notebook 4 (Homework): "5 experiments to understand architecture choices"
2. Due date: Before Monday class
3. Quick preview: "You'll experiment with filters, kernel sizes, depth, pooling"
4. Encourage: "Try to beat 92% accuracy!"

**Homework Assignment:**
- **Notebook:** `04_cnn_architecture_experiments.ipynb`
- **Points:** 20 points
- **Time:** 2-3 hours
- **Due:** Before Monday, November 3

---

## Session 2: Monday (1 hour)

### Pre-Session Check (5 minutes before class)

**Checklist:**
- [ ] Review student homework submissions (Notebook 4)
- [ ] Identify common issues/questions from homework
- [ ] Have Notebooks 5, 6, 7 ready to share

---

### Detailed Timeline: Monday Session

#### **Segment 1: Homework Discussion (10 minutes)**

**⏰ Time:** 0:00 - 0:10

**Activities:**
1. Quick poll: "Who achieved >90% accuracy?"
2. Ask 2-3 students to share their best architectures (1 min each)
3. Discuss common findings:
   - "Did more filters always help?"
   - "What was the optimal kernel size?"
   - "Did deeper networks perform better?"

**Teaching Points:**
- Celebrate successes
- Learn from experiments
- There's no single "perfect" architecture

---

#### **Segment 2: Advanced Regularization (25 minutes)**

**⏰ Time:** 0:10 - 0:35

**Notebook:** `05_deep_cnn_regularization.ipynb`

**Teaching Approach:**
1. **Overfitting Problem (5 min):**
   - Show baseline model overfitting (Part 2)
   - Ask: "Notice the gap between train and validation?"
2. **Dropout Explained (5 min):**
   - Draw neurons being "dropped" on whiteboard
   - Analogy: "Like training a team where random players sit out each game"
3. **Batch Normalization (5 min):**
   - Explain: Normalizes layer inputs
   - Benefits: Faster training, regularization effect
4. **Live Comparison (10 min):**
   - Run all 4 models (will take time - use for explanation)
   - Compare results: Baseline vs Dropout vs BatchNorm vs Optimized

**Key Teaching Points:**
- ⭐ **Dropout rate:** 0.25 for conv layers, 0.5 for dense layers
- ⭐ **BatchNorm:** Place after Conv2D, before activation
- ⭐ **GlobalAveragePooling:** Reduces parameters massively
- ⭐ **Combine techniques** for best results

**Time Management:** Training takes ~5-7 minutes - perfect time to explain concepts!

**Expected Outcome:** Students understand regularization reduces overfitting

---

#### **Segment 3: Data Augmentation (20 minutes)**

**⏰ Time:** 0:35 - 0:55

**Notebook:** `06_data_augmentation.ipynb`

**Teaching Approach:**
1. **Motivation (3 min):** "Limited data? Generate more!"
2. **Visual Demonstrations (10 min):**
   - Show Part 2-5: rotation, shifting, zoom, flip, combined
   - Students react: "Wow, same image but different!"
3. **Training Comparison (5 min):**
   - Show with vs without augmentation results
   - Emphasize: Better generalization
4. **Guidelines (2 min):** When to use which augmentation

**Key Teaching Points:**
- ⭐ **Only augment training data** (not validation/test)
- ⭐ **Appropriate augmentations** depend on dataset
- ⭐ **Fashion-MNIST:** Rotation, shift, zoom, flip all work
- ⭐ **MNIST digits:** NO horizontal flip (6 becomes 9!)

**Interactive:** Ask students: "Should we flip medical X-rays?" (Answer: Usually no!)

**Expected Outcome:** Students know how to implement and apply data augmentation

---

#### **Segment 4: Final Challenge Introduction (5 minutes)**

**⏰ Time:** 0:55 - 1:00

**Notebook:** `07_final_challenge.ipynb`

**Presentation:**
1. **Assignment Overview (2 min):**
   - "Build the BEST possible CNN for Fashion-MNIST"
   - Combine everything learned
   - Target: 93%+ for top marks
2. **Grading Rubric (2 min):**
   - 25 points total
   - Bonus points available (+5)
   - Show grading table
3. **Due Date & Submission (1 min):**
   - Due: [Instructor specifies]
   - Submit via Google Colab/Classroom
   - Include all reflection answers

**Final Challenge:**
- **Notebook:** `07_final_challenge.ipynb`
- **Points:** 25 points (+ 5 bonus)
- **Time:** 3-4 hours
- **Due:** [Instructor specifies]

---

## Teaching Tips

### General Pedagogy

1. **🎯 Focus on Intuition First, Math Second**
   - Start with analogies and visualizations
   - Then introduce formulas
   - Students remember concepts better than equations

2. **👥 Encourage Experimentation**
   - "What happens if you change this value?"
   - "Try it and see!"
   - Learning by doing is most effective

3. **📊 Use Visualizations Heavily**
   - Every notebook has rich visualizations
   - Pause on each plot: "What do you notice?"
   - Compare side-by-side results

4. **⏰ Time Management**
   - Training takes time - use it for explanations
   - Have backup: Pre-run notebooks if internet/GPU issues
   - It's okay to skip some experiments if time is tight

5. **❓ Anticipate Questions**
   - See "Common Student Questions" section below
   - Have answers ready
   - Encourage questions throughout

### Code Execution Strategy

**Option A: Live Coding (Recommended)**
- Run cells live in class
- Show outputs in real-time
- Students follow along

**Option B: Pre-run + Explain**
- Have notebooks pre-executed with outputs
- Walk through outputs
- Students can re-run at home

**Option C: Hybrid**
- Pre-run slow cells (training)
- Live code fast cells (visualizations)

### Student Engagement

1. **Polls/Questions:**
   - "How many think more filters always help?"
   - "What accuracy do you predict?"
   - Use hand raises or chat

2. **Pair Programming:**
   - Students work in pairs
   - Discuss architecture choices together
   - Share screen for homework

3. **Challenge Mindset:**
   - Frame as competition: "Can you beat 92%?"
   - Celebrate high scores
   - Create leaderboard (optional)

---

## Common Student Questions

### Technical Questions

**Q1: "Why is my model training so slow?"**

**A:** Several reasons:
- No GPU: Check if Colab GPU is enabled (Runtime → Change runtime type)
- Large batch size: Try batch_size=128 or 64
- Too many parameters: Use GlobalAveragePooling instead of Flatten
- Network issue: Download dataset once, then train offline

---

**Q2: "My validation accuracy is much lower than training accuracy. Why?"**

**A:** This is **overfitting**. Solutions:
- Add Dropout layers (0.25 after conv, 0.5 after dense)
- Use data augmentation
- Add BatchNormalization
- Reduce model complexity (fewer filters/layers)
- Use early stopping

---

**Q3: "What's the difference between MaxPooling and AveragePooling?"**

**A:**
- **MaxPooling:** Takes maximum value in window (preserves strong features)
- **AveragePooling:** Takes average (smooths features)
- **When to use:** MaxPooling is standard and works better in most cases

---

**Q4: "Should I use 'same' padding or 'valid' padding?"**

**A:**
- **'valid':** No padding (output shrinks)
- **'same':** Padding added (output same size as input)
- **Recommendation:** Use 'same' to preserve spatial dimensions, especially in deep networks

---

**Q5: "How do I know how many filters to use?"**

**A:** Common pattern:
- Start small: 32 or 64 filters in first layer
- Double each block: 32 → 64 → 128 → 256
- Rule of thumb: More complex dataset = more filters needed
- Experiment and compare validation accuracy!

---

**Q6: "Why does data augmentation sometimes hurt accuracy?"**

**A:**
- Too aggressive: rotation=180 might create unrealistic images
- Wrong augmentation: Flipping digits 6↔9 is bad
- Solution: Start conservative, visualize augmented images, adjust parameters

---

**Q7: "What's the output size formula again?"**

**A:** Write this on the board:
```
Output = (Input + 2*Padding - Kernel) / Stride + 1

Example: (28 + 2*0 - 3) / 1 + 1 = 26
```

---

**Q8: "BatchNormalization before or after activation?"**

**A:** Both work, but common practice:
```python
Conv2D → BatchNormalization → Activation('relu')
```
This is what modern architectures (ResNet, etc.) use.

---

**Q9: "Why use GlobalAveragePooling instead of Flatten?"**

**A:**
- **Flatten:** Converts 5×5×256 → 6400, then Dense(128) needs 6400×128 = 819K parameters!
- **GlobalAveragePooling:** Converts 5×5×256 → 256, then Dense(128) needs 256×128 = 33K parameters
- **Benefit:** 25× fewer parameters, less overfitting, same accuracy

---

**Q10: "Can I use pre-trained models like ResNet for Fashion-MNIST?"**

**A:**
- Yes, but overkill for 28×28 images
- Pre-trained models designed for 224×224 images
- For this assignment: Build from scratch to learn
- For real projects: Transfer learning is great!

---

### Conceptual Questions

**Q11: "Why do CNNs work better than MLPs for images?"**

**A:** Three main reasons:
1. **Spatial structure preserved:** Conv layers maintain 2D relationships
2. **Parameter sharing:** Same filter used everywhere (fewer parameters)
3. **Translation invariance:** Object detected regardless of position

---

**Q12: "What's actually happening during convolution?"**

**A:**
1. Place kernel on top of image region
2. Multiply element-wise
3. Sum all products → single output value
4. Slide kernel to next position
5. Repeat!

Show this visually on whiteboard with 3×3 example.

---

**Q13: "How does the network 'learn' the kernel values?"**

**A:**
- Kernels start with random values
- Backpropagation adjusts kernel weights to minimize loss
- First layers learn simple patterns (edges)
- Deeper layers learn complex patterns (shapes, objects)
- Just like dense layer weights!

---

**Q14: "Why do we need multiple filters in one layer?"**

**A:**
- Each filter detects different pattern
- Filter 1: Vertical edges
- Filter 2: Horizontal edges
- Filter 3: Diagonal edges
- Filter 4: Curves
- More filters = more pattern detectors!

---

**Q15: "What's the difference between epoch and batch?"**

**A:**
- **Batch:** Subset of data processed together (e.g., 128 images)
- **Epoch:** One complete pass through ALL training data
- Example: 60,000 images, batch_size=128 → 469 batches per epoch

---

## Troubleshooting

### Common Technical Issues

#### Issue 1: "ImportError: No module named tensorflow"

**Solution:**
```python
!pip install tensorflow==2.15.0
# Restart runtime
import tensorflow as tf
```

---

#### Issue 2: "ResourceExhaustedError: OOM when allocating tensor"

**Cause:** Out of memory (GPU/CPU)

**Solutions:**
1. Reduce batch size: `batch_size=64` or `batch_size=32`
2. Reduce model size: Fewer filters or layers
3. Use gradient accumulation
4. Restart Colab runtime to clear memory

---

#### Issue 3: "Training is stuck at 10% accuracy"

**Causes & Solutions:**
- **Wrong labels:** Check one-hot encoding vs integers
- **Wrong normalization:** Ensure `x / 255.0`
- **Dead ReLU:** Try different initialization or learning rate
- **Wrong loss function:** Use 'categorical_crossentropy' for one-hot labels

---

#### Issue 4: "Model not improving after epoch 1"

**Possible causes:**
- Learning rate too high: Try `lr=0.0001`
- Learning rate too low: Try `lr=0.01`
- Batch size too small: Use 128
- Bad initialization: Add BatchNormalization

---

#### Issue 5: "Notebook crashes when training"

**Solutions:**
1. Check GPU memory: Use smaller model
2. Restart runtime
3. Close other Colab tabs
4. Use CPU if GPU issues persist
5. Reduce batch size

---

#### Issue 6: "ImageDataGenerator not working"

**Check:**
```python
# Correct usage
datagen = ImageDataGenerator(rotation_range=15)
generator = datagen.flow(x_train, y_train, batch_size=128)
model.fit(generator, epochs=10, steps_per_epoch=len(x_train)//128)
```

Common mistake: Forgetting `steps_per_epoch`

---

#### Issue 7: "Validation accuracy higher than training accuracy"

**Explanation:** This is NORMAL when using Dropout!
- Dropout is active during training (removes neurons)
- Dropout is OFF during validation (uses all neurons)
- If gap is small (< 2%), it's fine

---

#### Issue 8: "Model predictions are all the same class"

**Causes:**
- Imbalanced data (unlikely with Fashion-MNIST)
- Dying ReLU: All activations are 0
- Wrong loss function
- Model collapsed to predicting majority class

**Solutions:**
- Check data distribution
- Use BatchNormalization
- Try LeakyReLU instead of ReLU
- Reduce learning rate

---

### Debugging Strategy

**Step-by-step debugging process:**

1. **Check data shapes:**
```python
print(x_train.shape)  # Should be (N, 28, 28, 1)
print(y_train.shape)  # Should be (N, 10) for one-hot
```

2. **Check data range:**
```python
print(x_train.min(), x_train.max())  # Should be 0.0, 1.0
```

3. **Test model on one batch:**
```python
model.fit(x_train[:128], y_train[:128], epochs=1)
```

4. **Check model output shape:**
```python
pred = model.predict(x_train[:1])
print(pred.shape)  # Should be (1, 10)
```

5. **Visualize predictions:**
```python
print(np.argmax(pred, axis=1))  # Predicted class
print(y_train[0])  # True label
```

---

## Assessment Guidelines

### Notebook 4: Architecture Experiments (20 points)

**Grading Rubric:**

| Component | Points | Criteria |
|-----------|--------|----------|
| Experiment 1 (Filters) | 3 | Completed with reflections |
| Experiment 2 (Kernel Size) | 3 | Completed with reflections |
| Experiment 3 (Depth) | 3 | Completed with reflections |
| Experiment 4 (Pooling) | 3 | Completed with reflections |
| Experiment 5 (Custom) | 4 | Creative architecture, >88% accuracy |
| Final Reflection | 2 | Thoughtful answers |
| Code Quality | 2 | Runs without errors |
| **Bonus: >92% accuracy** | +2 | Extra credit |

**What to look for:**
- ✅ All experiments run successfully
- ✅ Reflection questions answered thoughtfully
- ✅ Custom architecture shows experimentation
- ✅ Results table and visualizations generated
- ✅ Code is clean and documented

**Common Issues:**
- Incomplete reflection answers (deduct 0.5 pts each)
- Custom model doesn't improve baseline (acceptable, but note)
- Code doesn't run (major issue - return for revision)

---

### Notebook 7: Final Challenge (25 points)

**Grading Rubric:**

| Component | Points | Criteria |
|-----------|--------|----------|
| Architecture Design | 5 | Well-designed, justified CNN |
| Data Augmentation | 4 | Appropriate, explained |
| Regularization | 4 | 2+ techniques used |
| Test Accuracy | 6 | 87%:2pts, 89%:4pts, 91%:6pts |
| Documentation | 4 | Clear explanations |
| Code Quality | 2 | Clean, runs error-free |
| **Bonus Points** | +5 | Extra features |

**Bonus Opportunities:**
- +2 points: Test accuracy ≥93%
- +1 point: Learning rate scheduling
- +1 point: Detailed error analysis
- +1 point: Ensemble model

**What to look for:**
- ✅ Demonstrates understanding of all concepts
- ✅ Architecture is well-thought-out (not copy-paste)
- ✅ Data augmentation appropriate for Fashion-MNIST
- ✅ All reflection questions answered thoroughly
- ✅ Training curves show good practices
- ✅ Error analysis shows critical thinking

**Red Flags:**
- 🚩 Copy-pasted architecture without modification
- 🚩 No explanation of choices
- 🚩 Reflection answers are generic/minimal
- 🚩 Code doesn't run
- 🚩 Results contradict explanations

**Grading Time Estimate:** 15-20 minutes per submission

---

### Quick Grading Checklist

For each submission:

```
☐ Code runs without errors (5 pts)
☐ Test accuracy meets minimum threshold (6 pts)
☐ Architecture is documented and justified (5 pts)
☐ Data augmentation implemented and explained (4 pts)
☐ Regularization techniques used (4 pts)
☐ Reflection questions completed (4 pts)
☐ Code is clean and readable (2 pts)
☐ Bonus features attempted? (+5 pts possible)

Total: _____ / 25 (+ bonus)
```

---

## Additional Resources for Instructor

### Recommended Reading

1. **CS231n Stanford** - Convolutional Neural Networks
   - http://cs231n.stanford.edu/
   - Excellent lecture notes and assignments

2. **Deep Learning Book** - Goodfellow et al.
   - Chapter 9: Convolutional Networks
   - Available free online

3. **TensorFlow Tutorials**
   - https://www.tensorflow.org/tutorials/images/cnn
   - Official CNN tutorial

### Video Resources

1. **3Blue1Brown** - Neural Networks Series
   - Beautiful visualizations of convolution
   - https://www.youtube.com/c/3blue1brown

2. **Andrew Ng** - Deep Learning Specialization
   - Course 4: CNNs
   - Coursera

### Tools for Teaching

1. **CNN Explainer** - Interactive visualization
   - https://poloclub.github.io/cnn-explainer/
   - Show students how CNNs work visually

2. **TensorFlow Playground**
   - https://playground.tensorflow.org/
   - Interactive neural network training

3. **Netron** - Model visualization
   - https://netron.app/
   - Visualize model architecture

---

## Quick Reference: Key Formulas

**Output Size Calculation:**
```
Output = (Input + 2*Padding - Kernel) / Stride + 1
```

**Parameter Count:**
```
Conv Layer: (Kh × Kw × Cin + 1) × Cout
Dense Layer: (Input × Output) + Output
```

**Receptive Field:**
```
RF = 1 + Σ (Kernel - 1) × Π Previous_Strides
```

---

## Feedback Collection

After sessions, collect student feedback:

### Quick Survey (Google Form)

1. Rate understanding (1-5): Convolution operations
2. Rate understanding (1-5): CNN architecture design
3. Rate understanding (1-5): Regularization techniques
4. Rate understanding (1-5): Data augmentation
5. What was most helpful?
6. What was most confusing?
7. Pace: Too slow / Just right / Too fast
8. Suggestions for improvement?

Use feedback to adjust future sessions!

---

## Success Metrics

**Session is successful if:**
- ✅ 80%+ students achieve >88% test accuracy
- ✅ 50%+ students achieve >90% test accuracy
- ✅ 20%+ students achieve >92% test accuracy
- ✅ Students can explain regularization techniques
- ✅ Students can design CNN architectures independently
- ✅ Student satisfaction rating >4/5

---

## Contact & Support

**For technical issues:**
- TensorFlow GitHub: https://github.com/tensorflow/tensorflow/issues
- Stack Overflow: Tag `tensorflow` and `keras`

**For pedagogical questions:**
- CS Education research papers
- Teaching Deep Learning workshop materials

---

## Version History

- **v1.0** (November 1, 2025): Initial creation for Week 11-12 practical sessions

---

**Good luck with your teaching! Your students are lucky to have such comprehensive materials!** 🎓

