# Student Submission Guide: CNN Practical Assignments

**Course:** 21CSE558T - Deep Neural Network Architectures
**Module 4:** CNNs - Hands-on Implementation
**Instructor:** [Your Name]

---

## 📋 Assignments Overview

You have **TWO assignments** to submit:

| Assignment | Notebook | Points | Due Date | Difficulty |
|------------|----------|--------|----------|------------|
| **Homework** | `04_cnn_architecture_experiments.ipynb` | 20 pts | Before Monday class | Intermediate |
| **Final Challenge** | `07_final_challenge.ipynb` | 25 pts | [TBD by instructor] | Advanced |

**Total: 45 points** (18% of course grade)

---

## Assignment 1: Architecture Experiments (20 points)

### 📌 Overview

**Notebook:** `04_cnn_architecture_experiments.ipynb`
**Due:** Before Monday, November 3, 2025 class
**Time Required:** 2-3 hours
**Goal:** Systematically experiment with CNN architecture choices

### What You'll Do

Complete **5 experiments:**
1. **Experiment 1:** Effect of number of filters (16, 32, 64, 128)
2. **Experiment 2:** Effect of kernel size (3×3, 5×5, 7×7)
3. **Experiment 3:** Effect of network depth (2, 3, 4 conv blocks)
4. **Experiment 4:** Pooling strategies (Max, Average, Strided Conv)
5. **Experiment 5:** Your custom architecture (be creative!)

### Submission Requirements

**Mandatory:**
- ✅ All 5 experiments completed and running
- ✅ All reflection questions answered
- ✅ Results table generated
- ✅ Visualizations displayed
- ✅ Final reflection completed
- ✅ Custom architecture achieves >88% accuracy

**Optional Bonus:**
- 🌟 +2 points: Custom model achieves >92% accuracy

### How to Submit

#### Option A: Google Colab (Recommended)

1. **Open the notebook in Google Colab:**
   - Go to https://colab.research.google.com/
   - Upload `04_cnn_architecture_experiments.ipynb`
   - Or use File → Open notebook → Upload

2. **Complete all sections:**
   - Run all code cells
   - Answer all reflection questions
   - Ensure all visualizations are visible

3. **Save your work:**
   - File → Save a copy in Drive
   - Rename: `YourName_CNN_Experiments.ipynb`

4. **Share the notebook:**
   - Click "Share" button (top right)
   - Change access to "Anyone with the link can view"
   - Copy the link

5. **Submit:**
   - Paste the Colab link in [Google Classroom / LMS]
   - OR download as .ipynb and upload file

#### Option B: Jupyter Notebook (Local)

1. **Complete the notebook locally**
2. **Ensure all outputs are saved:**
   - Cell → Run All
   - File → Save and Checkpoint
3. **Submit the .ipynb file to [Google Classroom / LMS]**

### Before Submitting - Checklist

- [ ] All code cells execute without errors
- [ ] All 5 experiments show results
- [ ] Every reflection question answered
- [ ] Custom model (Experiment 5) achieves >88% test accuracy
- [ ] Final reflection section completed
- [ ] Notebook is named correctly: `YourName_CNN_Experiments.ipynb`
- [ ] All visualizations are visible in the notebook

### Common Mistakes to Avoid

❌ **Don't:**
- Submit without running all cells
- Leave reflection questions blank
- Submit with error messages
- Copy-paste others' custom architectures
- Submit the original template without modifications

✅ **Do:**
- Test your notebook runs top-to-bottom without errors
- Write thoughtful reflection answers (not just "yes" or "no")
- Document your custom architecture choices
- Show your work and thinking process

### Grading Rubric

| Component | Points | What We're Looking For |
|-----------|--------|------------------------|
| Experiment 1 | 3 | Runs + reflections answered |
| Experiment 2 | 3 | Runs + reflections answered |
| Experiment 3 | 3 | Runs + reflections answered |
| Experiment 4 | 3 | Runs + reflections answered |
| Experiment 5 | 4 | Creative architecture + good accuracy |
| Final Reflection | 2 | Thoughtful insights |
| Code Quality | 2 | Clean, runs without errors |
| **Bonus** | +2 | >92% accuracy on custom model |

**Minimum to Pass:** 14/20 points (70%)
**Good Performance:** 17/20 points (85%)
**Excellent:** 20/20 points (100%)

---

## Assignment 2: Final Challenge (25 points)

### 📌 Overview

**Notebook:** `07_final_challenge.ipynb`
**Due:** [Instructor will specify]
**Time Required:** 3-4 hours
**Goal:** Build the BEST possible CNN for Fashion-MNIST

### What You'll Do

Build an optimized CNN that combines:
- ✅ Well-designed architecture
- ✅ Regularization techniques (Dropout, BatchNorm, etc.)
- ✅ Data augmentation
- ✅ Training optimization (callbacks, learning rate)
- ✅ Comprehensive evaluation and error analysis

### Performance Targets

| Grade Tier | Test Accuracy | Description |
|------------|---------------|-------------|
| ⭐⭐⭐⭐⭐ Excellent | ≥93% | Outstanding! |
| ⭐⭐⭐⭐ Very Good | 91-92.9% | Strong work |
| ⭐⭐⭐ Good | 89-90.9% | Solid |
| ⭐⭐ Acceptable | 87-88.9% | Basic proficiency |
| ⭐ Needs Work | <87% | Keep trying |

**Your goal:** Achieve the highest accuracy possible!

### Submission Requirements

**Mandatory Sections:**
- ✅ Part 1: Data exploration
- ✅ Part 2: Data preprocessing
- ✅ Part 3: Data augmentation (documented)
- ✅ Part 4: CNN architecture (documented)
- ✅ Part 5: Training strategy
- ✅ Part 6: Training execution
- ✅ Part 7: Training visualization
- ✅ Part 8: Test evaluation
- ✅ Part 10: Final reflection (all questions)

**Optional for Bonus:**
- 🌟 Part 9: Detailed error analysis (+1 point)
- 🌟 Learning rate scheduling (+1 point)
- 🌟 Ensemble model (+1 point)
- 🌟 Test accuracy ≥93% (+2 points)

### How to Submit

#### Step 1: Complete the Notebook

1. **Open in Google Colab** (recommended)
2. **Work through all sections:**
   - Write your code in marked sections
   - Answer all reflection questions
   - Document your design choices
3. **Ensure your model trains successfully**
4. **Achieve at least 87% test accuracy**

#### Step 2: Final Review

Run this checklist **before submitting:**

```python
# Submission Checklist Code
print("Submission Checklist:")
print("=" * 50)

# 1. Check test accuracy
if test_acc >= 0.87:
    print("✅ Test accuracy ≥87%:", f"{test_acc:.2%}")
else:
    print("❌ Test accuracy <87%:", f"{test_acc:.2%}")
    print("   ⚠️ Please improve your model before submitting!")

# 2. Check notebook runs
print("\n✅ All cells run without errors")
print("   (If you see this, your notebook ran successfully!)")

# 3. Check documentation
print("\n📝 Documentation Checklist:")
print("   - [ ] All reflection questions answered?")
print("   - [ ] Architecture choices explained?")
print("   - [ ] Augmentation strategy documented?")
print("   - [ ] Final reflection completed?")

print("\n" + "=" * 50)
print("If all items are checked, you're ready to submit!")
```

#### Step 3: Save and Share

**Google Colab:**
1. File → Save a copy in Drive
2. Rename: `YourName_CNN_Final_Challenge.ipynb`
3. Share → Anyone with link can view
4. Copy link

**Submit:**
- Paste link to [Google Classroom / LMS]
- OR download .ipynb and upload

#### Step 4: Optional - Save Your Model

If you achieved excellent results:
```python
# Save your best model
model.save('YourName_best_model.h5')
```

Download and keep for your portfolio!

### Before Submitting - Complete Checklist

- [ ] **Code Quality**
  - [ ] All cells run without errors (Run All → Check)
  - [ ] No placeholder comments like "TODO: Your code here"
  - [ ] Code is readable and well-commented

- [ ] **Performance**
  - [ ] Test accuracy ≥87% (minimum requirement)
  - [ ] Training curves show learning (not flat lines)
  - [ ] No extreme overfitting (gap <10%)

- [ ] **Documentation**
  - [ ] Augmentation strategy explained
  - [ ] Architecture design justified
  - [ ] All reflection questions answered thoroughly
  - [ ] Design decisions documented

- [ ] **Completeness**
  - [ ] All mandatory parts completed (Parts 1-8, 10)
  - [ ] Confusion matrix generated
  - [ ] Sample predictions visualized
  - [ ] Final summary generated

- [ ] **Submission Format**
  - [ ] Notebook renamed correctly
  - [ ] All outputs visible (don't clear outputs!)
  - [ ] Shared link works (test in incognito window)

### Common Mistakes to Avoid

❌ **Architecture Issues:**
- Too simple (single conv layer)
- Too complex (>5M parameters)
- No regularization
- No data augmentation

❌ **Documentation Issues:**
- Generic answers ("I used dropout because it's good")
- Empty reflection questions
- No justification for choices
- Copy-pasted explanations

❌ **Technical Issues:**
- Model doesn't train (accuracy stuck at 10%)
- Extreme overfitting (95% train, 70% test)
- Code doesn't run
- Missing outputs

✅ **Best Practices:**
- Start early, iterate multiple times
- Document your thinking process
- Experiment and compare results
- Show what you tried and why
- Ask questions if stuck!

### Grading Rubric

| Component | Points | Criteria |
|-----------|--------|----------|
| **Architecture Design** | 5 | Well-designed, justified CNN structure |
| **Data Augmentation** | 4 | Appropriate augmentation with explanation |
| **Regularization** | 4 | Multiple techniques used effectively |
| **Test Accuracy** | 6 | 87%:2pts, 89%:4pts, 91%:6pts |
| **Documentation** | 4 | Clear, thoughtful explanations |
| **Code Quality** | 2 | Runs cleanly, well-organized |
| **Base Total** | **25** | |
| **BONUS** | +5 | Extra features, high accuracy |

**Bonus Opportunities:**
- +2 points: Test accuracy ≥93%
- +1 point: Learning rate scheduling
- +1 point: Detailed error analysis
- +1 point: Ensemble model

**Maximum Possible: 30 points** (25 base + 5 bonus)

---

## General Submission Guidelines

### File Naming Convention

Always use this format:
```
YourName_Assignment_Date.ipynb

Examples:
- RaviSharma_CNN_Experiments.ipynb
- PriyaPatel_CNN_Final_Challenge.ipynb
```

### Submission Methods

**Method 1: Google Colab Link (Preferred)**
- Fastest for grading
- Instructor can view immediately
- Can run your code to verify
- Make sure sharing is enabled!

**Method 2: Jupyter Notebook File**
- Upload .ipynb file to LMS
- Ensure all outputs are saved
- File size should be <10MB

**Method 3: GitHub (Advanced)**
- Create public repo
- Push notebook with outputs
- Share repo link
- Bonus points for professional presentation!

### Late Submission Policy

- **On time:** Full points
- **1 day late:** -10% penalty
- **2 days late:** -20% penalty
- **>2 days late:** Case-by-case basis

**Extensions:** Contact instructor BEFORE due date if you need extra time.

---

## Technical Support

### Getting Help

**Before asking for help, try:**
1. Read error message carefully
2. Check the Troubleshooting section in instructor guide
3. Google the error message
4. Check Stack Overflow

**When asking for help, provide:**
1. What you're trying to do
2. What error you're getting (full error message)
3. What you've already tried
4. Code snippet (not screenshot!)

### Common Issues & Quick Fixes

**Issue 1: "Notebook won't run"**
```python
# Solution: Restart runtime
Runtime → Restart runtime
# Then run all cells again
```

**Issue 2: "Out of memory error"**
```python
# Solution: Reduce batch size
batch_size = 64  # Instead of 128
```

**Issue 3: "Can't install TensorFlow"**
```python
# Solution: Use specific version
!pip install tensorflow==2.15.0
# Then restart runtime
```

**Issue 4: "Model not improving"**
```python
# Check these:
# 1. Data normalized? x_train / 255.0
# 2. Labels one-hot encoded?
# 3. Using correct loss function?
# 4. Learning rate reasonable? (0.001 is good start)
```

**Issue 5: "Colab disconnects during training"**
```python
# Solution: Use Early Stopping
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=10)
model.fit(..., callbacks=[early_stop])
```

### Office Hours

**When:** [Instructor will specify]
**Where:** [Location / Zoom link]
**How:** First come, first served or appointment

---

## Academic Integrity

### What's Allowed ✅

- Discussing concepts with classmates
- Helping each other debug errors
- Sharing resources (tutorials, documentation)
- Asking instructor for clarification
- Using online resources (with citation)

### What's NOT Allowed ❌

- Copying code from others
- Sharing your complete solution
- Using someone else's trained model
- Plagiarizing reflection answers
- Submitting the same work as others

### Citation Guidelines

If you use external code/ideas:
```python
# Source: https://stackoverflow.com/questions/...
# Adapted from TensorFlow documentation
# Inspired by CS231n lecture notes
```

### Consequences

- **First offense:** Warning + resubmission
- **Second offense:** 0 on assignment
- **Serious cases:** Reported to academic office

**When in doubt:** Ask the instructor!

---

## Tips for Success

### Time Management

**Assignment 1 (Architecture Experiments):**
- Day 1 (1 hour): Complete experiments 1-4
- Day 2 (1 hour): Work on experiment 5 (custom architecture)
- Day 3 (0.5 hour): Final reflections and submission

**Assignment 2 (Final Challenge):**
- Week 1 (1 hour): Initial architecture and baseline
- Week 1 (1 hour): Add regularization and augmentation
- Week 2 (1 hour): Optimization and tuning
- Week 2 (1 hour): Error analysis and documentation

### Performance Tips

**To achieve 90%+ accuracy:**
1. Use data augmentation (rotation, shift, zoom)
2. Add BatchNormalization after conv layers
3. Use Dropout (0.25 after conv, 0.5 after dense)
4. Try 3-4 conv blocks with increasing filters (32→64→128→256)
5. Use GlobalAveragePooling instead of Flatten
6. Train for enough epochs (20-30 with early stopping)

**To achieve 93%+ accuracy:**
1. All of the above, plus:
2. Learning rate scheduling
3. More aggressive augmentation
4. Deeper network (4-5 conv blocks)
5. Multiple runs with different seeds
6. Ensemble predictions

### Writing Good Reflections

**Bad reflection answer:**
> "I used dropout because it's good for regularization."

**Good reflection answer:**
> "I used dropout with rate 0.25 after conv layers and 0.5 after dense layers because the baseline model showed a 15% gap between training and validation accuracy, indicating overfitting. After adding dropout, the gap reduced to 3%, and validation accuracy improved from 88% to 91%."

**Key differences:**
- ✅ Specific numbers and rates
- ✅ Evidence from experiments
- ✅ Explains reasoning
- ✅ Shows understanding

---

## Frequently Asked Questions (FAQ)

### General Questions

**Q: Can I use pre-trained models like ResNet?**
A: No, for these assignments you must build from scratch to demonstrate learning. Pre-trained models are for future projects!

**Q: How much code should I write?**
A: Quality over quantity. A well-designed 50-line architecture is better than a messy 200-line one.

**Q: Can I work in groups?**
A: You can discuss concepts, but submit individual work. Your code and reflections must be your own.

**Q: What if I can't achieve the target accuracy?**
A: Submit your best attempt with documentation. Partial credit is available. Show what you tried!

**Q: Can I submit early?**
A: Yes! Early submission is encouraged. You can revise if needed before the deadline.

### Technical Questions

**Q: Should I use CPU or GPU?**
A: GPU is much faster. In Colab: Runtime → Change runtime type → GPU.

**Q: How long should training take?**
A: With GPU: 5-10 minutes. With CPU: 30-60 minutes. Use early stopping to save time.

**Q: What if my Colab keeps disconnecting?**
A: Colab free tier has session limits. Upgrade to Colab Pro, or train locally, or use early stopping.

**Q: Can I use different datasets?**
A: No, these assignments are specifically for Fashion-MNIST to ensure fair comparison.

**Q: How do I know if my accuracy is good enough?**
A: Check the grading rubric. For Assignment 1: >88%. For Assignment 2: >87% (better is encouraged!).

### Grading Questions

**Q: When will grades be posted?**
A: Within 1 week of submission deadline.

**Q: Can I resubmit if I made a mistake?**
A: Before deadline: Yes, resubmit anytime. After deadline: Contact instructor.

**Q: How are reflection answers graded?**
A: Based on thoughtfulness and demonstration of understanding, not length.

**Q: What if my code doesn't run on instructor's machine?**
A: You'll be asked to fix and resubmit with late penalty. Test thoroughly before submitting!

---

## Resources

### Official Documentation

- **TensorFlow:** https://www.tensorflow.org/tutorials/images/cnn
- **Keras API:** https://keras.io/api/
- **NumPy:** https://numpy.org/doc/
- **Matplotlib:** https://matplotlib.org/

### Learning Resources

- **CS231n Stanford:** http://cs231n.stanford.edu/
- **Deep Learning Book:** https://www.deeplearningbook.org/
- **3Blue1Brown Neural Networks:** https://www.youtube.com/watch?v=aircAruvnKk

### Tools

- **Google Colab:** https://colab.research.google.com/
- **CNN Explainer:** https://poloclub.github.io/cnn-explainer/
- **TensorFlow Playground:** https://playground.tensorflow.org/

---

## Contact Information

**Instructor:** [Your Name]
**Email:** [Your Email]
**Office:** [Location]
**Office Hours:** [Times]

**Course Website:** [URL]
**Google Classroom:** [Link]

---

## Final Checklist Before Each Submission

```
Assignment Submission Checklist

□ Notebook runs without errors (tested with "Run All")
□ All reflection questions answered
□ All visualizations visible
□ Achieves minimum accuracy requirement
□ File named correctly: YourName_Assignment.ipynb
□ Shared link works (tested in incognito mode)
□ No TODO comments left in code
□ Code is clean and commented
□ Ready to submit!

Submit to: [Google Classroom / LMS Link]
```

---

## Good Luck! 🎓

You've got this! These assignments are challenging but doable. Key tips:

1. **Start early** - Don't wait until the last day
2. **Experiment** - Try different approaches
3. **Document** - Write down what you learn
4. **Ask questions** - Use office hours
5. **Have fun** - Deep learning is exciting!

Remember: The goal is learning, not just the grade. The skills you gain here (CNN design, regularization, debugging, experimentation) are valuable for your career in AI/ML.

**You're building production-ready deep learning skills!**

---

*Last updated: November 1, 2025*
*Version 1.0*
