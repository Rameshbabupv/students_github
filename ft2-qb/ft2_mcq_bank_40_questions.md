# 40 MCQ Question Bank for Formative Test 2 (FT2)

**Course:** 21CSE558T - Deep Neural Network Architectures
**Coverage:** Modules 3-4 (Image Processing & CNNs)
**Format:** 1 mark each
**Total Questions:** 40 (20 from Module 3, 20 from Module 4)
**Date:** November 14, 2025

---

## Module 3: Image Processing & Deep Neural Networks (20 MCQs)

### Section 3.1: Image Representation (3 MCQs)

**Q1.** In a color image, how many channels are typically present in the RGB format?
a) 1
b) 2
c) 3
d) 4

**Answer:** c) 3
**Explanation:** RGB images have three channels: Red, Green, and Blue.

---

**Q2.** What is the pixel value range in a standard 8-bit grayscale image?
a) 0 to 100
b) 0 to 255
c) -128 to 127
d) 0 to 1

**Answer:** b) 0 to 255
**Explanation:** 8-bit images use 2^8 = 256 values, ranging from 0 (black) to 255 (white).

---

**Q3.** Which color space is commonly used for separating color information from intensity?
a) RGB
b) BGR
c) HSV
d) CMYK

**Answer:** c) HSV
**Explanation:** HSV (Hue, Saturation, Value) separates color (Hue) from intensity (Value), making it useful for color-based segmentation.

---

### Section 3.2: Image Enhancement Techniques (3 MCQs)

**Q4.** Histogram equalization is primarily used for:
a) Edge detection
b) Image compression
c) Improving image contrast
d) Noise removal

**Answer:** c) Improving image contrast
**Explanation:** Histogram equalization redistributes pixel intensities to enhance contrast in low-contrast images.

---

**Q5.** Which technique is used to enhance edges while preserving smooth regions in an image?
a) Gaussian blur
b) Median filter
c) Unsharp masking
d) Box filter

**Answer:** c) Unsharp masking
**Explanation:** Unsharp masking subtracts a blurred version from the original to enhance edges while keeping smooth areas intact.

---

**Q6.** Gamma correction is used to:
a) Remove noise from images
b) Adjust brightness and contrast non-linearly
c) Detect edges
d) Compress image size

**Answer:** b) Adjust brightness and contrast non-linearly
**Explanation:** Gamma correction applies a power-law transformation to adjust image brightness non-linearly.

---

### Section 3.3: Noise Removal and Filtering (2 MCQs)

**Q7.** Which filter is most effective for removing salt-and-pepper noise?
a) Gaussian filter
b) Mean filter
c) Median filter
d) Sobel filter

**Answer:** c) Median filter
**Explanation:** Median filter replaces each pixel with the median of neighboring pixels, effectively removing salt-and-pepper noise while preserving edges.

---

**Q8.** Gaussian blur is applied to an image primarily to:
a) Sharpen edges
b) Reduce high-frequency noise
c) Enhance contrast
d) Detect corners

**Answer:** b) Reduce high-frequency noise
**Explanation:** Gaussian blur smooths images by reducing high-frequency components (noise and fine details).

---

### Section 3.4: Edge Detection Methods (3 MCQs)

**Q9.** Which edge detection method uses two kernels to detect horizontal and vertical edges?
a) Canny edge detector
b) Sobel operator
c) Laplacian operator
d) Prewitt operator

**Answer:** b) Sobel operator (also d) Prewitt operator is acceptable)
**Explanation:** Sobel uses two 3×3 kernels (Gx and Gy) to detect edges in horizontal and vertical directions.

---

**Q10.** The Canny edge detector is superior to simple gradient-based methods because it:
a) Uses only one kernel
b) Performs non-maximum suppression and hysteresis thresholding
c) Works only on binary images
d) Requires no parameters

**Answer:** b) Performs non-maximum suppression and hysteresis thresholding
**Explanation:** Canny edge detector includes multiple stages (gradient calculation, non-maximum suppression, double thresholding, edge tracking) for robust edge detection.

---

**Q11.** Which operator detects edges based on the second derivative of the image?
a) Sobel
b) Prewitt
c) Laplacian
d) Roberts

**Answer:** c) Laplacian
**Explanation:** Laplacian operator uses the second derivative to detect edges, finding regions of rapid intensity change.

---

### Section 3.5: Segmentation Techniques (3 MCQs)

**Q12.** Otsu's method is used for:
a) Edge detection
b) Automatic threshold selection for binarization
c) Image sharpening
d) Noise removal

**Answer:** b) Automatic threshold selection for binarization
**Explanation:** Otsu's method automatically determines the optimal threshold to separate foreground and background by maximizing inter-class variance.

---

**Q13.** Region-based segmentation groups pixels based on:
a) Edge information only
b) Similarity criteria like color or intensity
c) Random selection
d) Histogram peaks only

**Answer:** b) Similarity criteria like color or intensity
**Explanation:** Region-based segmentation (e.g., region growing, region splitting) groups pixels with similar properties.

---

**Q14.** Watershed segmentation treats the image gradient as:
a) A flat surface
b) A topographic surface with catchment basins
c) A binary mask
d) A histogram

**Answer:** b) A topographic surface with catchment basins
**Explanation:** Watershed algorithm treats the gradient magnitude as a topographic relief, where local minima are "flooded" to create segmentation boundaries.

---

### Section 3.6: Feature Extraction (4 MCQs)

**Q15.** Local Binary Pattern (LBP) is primarily used for extracting:
a) Color features
b) Texture features
c) Edge features
d) Shape features

**Answer:** b) Texture features
**Explanation:** LBP captures texture information by comparing each pixel with its neighbors to create binary patterns.

---

**Q16.** Gray Level Co-occurrence Matrix (GLCM) captures:
a) Color distribution
b) Spatial relationship between pixel intensities
c) Edge orientation
d) Image frequency components

**Answer:** b) Spatial relationship between pixel intensities
**Explanation:** GLCM represents how often pairs of pixels with specific values occur in a specified spatial relationship, capturing texture.

---

**Q17.** Which shape feature describes the compactness of an object?
a) Area
b) Perimeter
c) Circularity (or Compactness)
d) Centroid

**Answer:** c) Circularity (or Compactness)
**Explanation:** Circularity = (4π × Area) / (Perimeter²) measures how close a shape is to a perfect circle.

---

**Q18.** Color histograms represent:
a) Spatial distribution of pixels
b) Frequency distribution of color values
c) Edge strength distribution
d) Texture patterns

**Answer:** b) Frequency distribution of color values
**Explanation:** Color histograms count the number of pixels for each color value, representing color distribution.

---

### Section 3.7: OpenCV Operations (2 MCQs)

**Q19.** In OpenCV, which function is used to convert a color image to grayscale?
a) cv2.gray()
b) cv2.cvtColor()
c) cv2.convert()
d) cv2.grayscale()

**Answer:** b) cv2.cvtColor()
**Explanation:** cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) converts color images to grayscale.

---

**Q20.** Which OpenCV function applies Gaussian blur to an image?
a) cv2.blur()
b) cv2.medianBlur()
c) cv2.GaussianBlur()
d) cv2.filter2D()

**Answer:** c) cv2.GaussianBlur()
**Explanation:** cv2.GaussianBlur() applies Gaussian smoothing using a Gaussian kernel.

---

---

## Module 4: Convolutional Neural Networks & Transfer Learning (20 MCQs)

### Section 4.1: Convolution Operation (3 MCQs)

**Q21.** In a CNN, what is the primary purpose of the convolution operation?
a) Reduce image size
b) Extract local features from input
c) Classify images
d) Normalize pixel values

**Answer:** b) Extract local features from input
**Explanation:** Convolution applies filters/kernels to extract local patterns like edges, textures, and shapes.

---

**Q22.** What happens when you use 'same' padding in a convolutional layer?
a) Output size is smaller than input
b) Output size is the same as input (with stride=1)
c) No padding is added
d) Output size is always doubled

**Answer:** b) Output size is the same as input (with stride=1)
**Explanation:** 'same' padding adds zeros around the input to ensure output spatial dimensions match input dimensions (when stride=1).

---

**Q23.** Increasing the stride in a convolutional layer will:
a) Increase output spatial dimensions
b) Decrease output spatial dimensions
c) Not affect output size
d) Only affect the number of filters

**Answer:** b) Decrease output spatial dimensions
**Explanation:** Larger stride means the filter skips more positions, resulting in smaller output dimensions.

---

### Section 4.2: Pooling Layers (3 MCQs)

**Q24.** What is the main advantage of using Max Pooling over Average Pooling?
a) Preserves the strongest features/activations
b) Preserves all information equally
c) Requires more computation
d) Always produces better accuracy

**Answer:** a) Preserves the strongest features/activations
**Explanation:** Max pooling selects the maximum value in each region, preserving the most prominent features, while average pooling smooths features.

---

**Q25.** GlobalAveragePooling2D differs from regular pooling by:
a) Using larger pool size
b) Reducing entire feature map to a single value per channel
c) Only working with grayscale images
d) Requiring more parameters

**Answer:** b) Reducing entire feature map to a single value per channel
**Explanation:** GlobalAveragePooling2D averages all values in each feature map, converting (H, W, C) to (1, 1, C), then flattened to (C,).

---

**Q26.** Using GlobalAveragePooling2D instead of Flatten + Dense reduces:
a) Training time only
b) Number of parameters significantly
c) Model accuracy
d) Number of filters needed

**Answer:** b) Number of parameters significantly
**Explanation:** GlobalAvgPool reduces spatial dimensions to 1×1, drastically reducing parameters in the following Dense layer compared to flattening entire feature maps.

---

### Section 4.3: Batch Normalization in CNNs (2 MCQs)

**Q27.** In a CNN, Batch Normalization should be placed:
a) Before the convolutional layer
b) After convolution, before activation
c) After activation function
d) Only at the output layer

**Answer:** b) After convolution, before activation
**Explanation:** Modern best practice: Conv2D → BatchNormalization → Activation (e.g., ReLU). This normalizes pre-activation values for stable training.

---

**Q28.** Batch Normalization in CNNs helps to:
a) Reduce the number of parameters
b) Speed up training and act as regularization
c) Remove the need for activation functions
d) Eliminate the need for dropout

**Answer:** b) Speed up training and act as regularization
**Explanation:** BatchNorm normalizes layer inputs, reducing internal covariate shift, allowing higher learning rates and providing mild regularization.

---

### Section 4.4: Dropout in CNNs (2 MCQs)

**Q29.** In a CNN architecture, where should Dropout typically be applied?
a) Before the first convolutional layer
b) After pooling layers and before dense layers
c) After every convolutional layer
d) After the output layer

**Answer:** b) After pooling layers and before dense layers
**Explanation:** Dropout is typically placed after pooling (e.g., 0.2, 0.3) and before dense layers (e.g., 0.5) to prevent overfitting. NEVER after output layer.

---

**Q30.** Progressive dropout rates in a CNN (0.2 → 0.3 → 0.5) mean:
a) Dropout rate decreases as network deepens
b) Dropout rate increases as network deepens
c) All layers use the same dropout rate
d) Dropout is only applied at the output

**Answer:** b) Dropout rate increases as network deepens
**Explanation:** Progressive strategy: lower dropout after early pooling (0.2), moderate after middle pooling (0.3), higher before dense layers (0.5).

---

### Section 4.5: Data Augmentation (2 MCQs)

**Q31.** For CIFAR-10 dataset (airplanes, cars, animals), which augmentation is appropriate?
a) Vertical flip (upside-down images)
b) Horizontal flip (mirror images)
c) 180-degree rotation
d) Extreme color inversion

**Answer:** b) Horizontal flip (mirror images)
**Explanation:** Horizontal flip is appropriate (plane facing left/right is still a plane). Vertical flip is inappropriate (upside-down plane changes meaning).

---

**Q32.** Data augmentation in CNNs should be applied to:
a) Training data only
b) Validation data only
c) Test data only
d) All datasets equally

**Answer:** a) Training data only
**Explanation:** Augmentation is applied ONLY to training data to increase diversity. Validation/test data should remain original to evaluate real-world performance.

---

### Section 4.6: Famous CNN Architectures (4 MCQs)

**Q33.** LeNet-5, the pioneering CNN architecture, was originally designed for:
a) ImageNet classification
b) Handwritten digit recognition
c) Object detection
d) Face recognition

**Answer:** b) Handwritten digit recognition
**Explanation:** LeNet-5 (1998) by Yann LeCun was designed for recognizing handwritten digits (MNIST), used in check reading systems.

---

**Q34.** What was the key innovation of AlexNet (2012) that led to its breakthrough performance?
a) Use of sigmoid activation throughout
b) Very shallow architecture (3 layers)
c) Use of ReLU activation and GPU training
d) No pooling layers

**Answer:** c) Use of ReLU activation and GPU training
**Explanation:** AlexNet used ReLU (faster training than sigmoid), GPU acceleration, and dropout, achieving breakthrough performance on ImageNet 2012.

---

**Q35.** VGG architecture is characterized by:
a) Very small 3×3 convolutional filters throughout
b) Large 11×11 filters
c) No pooling layers
d) Random filter sizes

**Answer:** a) Very small 3×3 convolutional filters throughout
**Explanation:** VGG (VGG16, VGG19) uses uniform 3×3 convolutions throughout, showing that deeper networks with small filters work well.

---

**Q36.** ResNet introduced which key innovation to train very deep networks?
a) Larger learning rates
b) Skip connections (residual connections)
c) Removal of pooling layers
d) Only using 1×1 convolutions

**Answer:** b) Skip connections (residual connections)
**Explanation:** ResNet uses skip connections (x + F(x)) to allow gradients to flow directly through the network, enabling training of 100+ layer networks.

---

### Section 4.7: Transfer Learning (4 MCQs)

**Q37.** Transfer learning is most effective when:
a) You have millions of labeled images in your target domain
b) Target dataset is very different from source dataset
c) You have limited labeled data and domains are similar
d) Training from scratch is faster

**Answer:** c) You have limited labeled data and domains are similar
**Explanation:** Transfer learning works best when you have few labeled examples and the source dataset (e.g., ImageNet) is similar to your target task.

---

**Q38.** In transfer learning, "freezing" base layers means:
a) Deleting those layers
b) Setting their weights to zero
c) Making their weights non-trainable
d) Doubling their learning rate

**Answer:** c) Making their weights non-trainable
**Explanation:** Freezing layers (base_model.trainable = False) prevents their weights from updating during training, preserving pre-trained features.

---

**Q39.** Which pre-trained model is commonly used for transfer learning in image classification?
a) BERT
b) GPT-3
c) VGG16
d) LSTM

**Answer:** c) VGG16
**Explanation:** VGG16, ResNet50, MobileNet are popular pre-trained models for image tasks. BERT/GPT-3 are for NLP, LSTM is an architecture type.

---

**Q40.** Fine-tuning in transfer learning refers to:
a) Deleting all pre-trained weights
b) Training only the final classification layer
c) Unfreezing some base layers and training with low learning rate
d) Using higher learning rates than training from scratch

**Answer:** c) Unfreezing some base layers and training with low learning rate
**Explanation:** Fine-tuning unfreezes some/all base layers and trains them with a small learning rate to adapt pre-trained features to the new task.

---

---

## Summary Statistics

### Module 3 (20 MCQs):
- Image Representation: 3
- Image Enhancement: 3
- Noise Removal: 2
- Edge Detection: 3
- Segmentation: 3
- Feature Extraction: 4
- OpenCV: 2

### Module 4 (20 MCQs):
- Convolution Operation: 3
- Pooling Layers: 3
- Batch Normalization: 2
- Dropout: 2
- Data Augmentation: 2
- Famous Architectures: 4
- Transfer Learning: 4

### Difficulty Distribution:
- Easy: 22 questions
- Moderate: 15 questions
- Difficult: 3 questions

### Coverage:
- ✅ NO overlap with FT1 (Modules 1-2)
- ✅ CNN-specific applications (not general concepts)
- ✅ Conceptual focus (light calculations)
- ✅ Includes Transfer Learning
- ✅ Balanced Module 3-4 (20-20)

---

**Last Updated:** October 30, 2025
**Status:** Complete - Ready for FT2 test paper generation
