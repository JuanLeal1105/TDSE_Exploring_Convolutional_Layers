# TDSE_Exploring_Convolutional_Layers

## Justification: Why this Dataset is Appropriate for Convolutional Layers

The **Blood Cell Images (BCCD)** dataset was selected because the task of classifying white blood cells maps perfectly to the strengths of Convolutional Neural Networks (CNNs). Standard "dense" neural networks would fail to capture the spatial relationships in these images efficiently.

### 1. Hierarchical Feature Learning
Biological cell classification requires analyzing features at different levels of abstraction, which matches the architecture of a CNN:
* **Low-level features (Early Layers):** The network detects edges and color gradients, effectively separating the cell boundaries from the plasma background.
* **Mid-level features (Middle Layers):** The network identifies specific textures, such as the *granularity* found in Eosinophils versus the smooth cytoplasm of Lymphocytes.
* **High-level features (Deep Layers):** The network combines these shapes to recognize complex structures, such as the multi-lobed nucleus of a Neutrophil versus the kidney-bean shape of a Monocyte.

### 2. Translation Invariance
In microscopy slides, the cell of interest is rarely perfectly centered. A CNN processes the image using sliding windows (kernels), meaning it can identify a Monocyte whether it is in the top-left corner or the center of the image. A standard Multi-Layer Perceptron (MLP) would treat these as completely different inputs.

### 3. Noise Reduction via Pooling
The dataset contains "noise" in the form of red blood cells surrounding the target white blood cell. The **pooling layers** in a CNN help the model become robust to these background variations by summarizing the presence of features in a region, allowing the model to focus on the dominant signal (the white blood cell) rather than the background clutter.

---

## **Assignment Tasks**

### **Part 2. Baseline Model (Non-Convolutional)**
To establish a performance baseline, a standard Multi-Layer Perceptron (MLP) was trained on the raw pixel data without any convolutional layers. The base architecture is defined by:
- Input: 128x128 RGB images (3 channels).
- Flatten Layer: Unrolls the 3D image tensor into a 1D vector of 49,152 values.
- Dense Layer: 128 neurons with ReLU activation.
- Output Layer: 4 neurons with Softmax activation (one for each cell type).

#### **Model Stats**
The total parameters found for the developed architecture was: 6,292,100 (~6.3 Million). According to that, we can establish that this is an incredibly high number of parameters for such a simple architecture, primarily caused by the Dense layer connecting to every single pixel in the Flattened input.

#### **Performance Results**
What we obtained was:
- Final Training Accuracy: ~25%
- Final Validation Accuracy: ~25%
- Training Behavior: As seen in the graphs, the model briefly attempted to learn (peaking around 31% accuracy at Epoch 4) before collapsing back to 25%.

**Loss Analysis**

The final loss stabilized at 1.3863. The mathematical context is the following: $-\ln(0.25) \approx 1.386$. This exact loss value confirms the model stopped trying to distinguish classes and settled on predicting "25% probability" for every class (pure random guessing).

#### **Observed Limitations**
1. Total Loss of Spatial Structure: By flattening the image, the model lost all concept of "shape." It treats a pixel in the nucleus and a pixel in the cytoplasm as unrelated features if they aren't adjacent in the 1D vector.
2. Parameter Inefficiency: Despite having over 6 million parameters, the model failed to beat random chance. A CNN can achieve >90% accuracy with fewer parameters.
3. Vanishing Gradient / Collapse: The accuracy graph shows a "collapse" behavior where the model gave up on learning features and converged to a trivial solution (predicting all classes equally).

