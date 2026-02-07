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

### **Part 1. Dataset Exploration**
The Blood Cell Images (BCCD) dataset was selected to classify white blood cell types. The initial exploration phase focused on understanding the data structure and quality before training.
#### **Dataset Size and Class Distribution**
The dataset contains a total of 9,957 images split into training (80%) and validation (20%) sets. The classes are remarkably balanced, with approximately 2,500 images per category, preventing the need for class weighting techniques.
- Total images: 9957
- Four classes were found:
  - EOSINOPHIL
  - LYMPHOCYTE
  - MONOCYTE
  - NEUTROPHIL
    
#### **Image Dimensions and Channel** 
- Original Resolution: 320 x 240 pixels (Width x Height).
- Color Channels: 3 (RGB).
- Data Type: 8-bit Integer (0-255 pixel values).

#### **Preprocessing Strategy**
Based on the EDA, the following preprocessing pipeline is required before feeding data into the CNN:
1. Resizing (Rectangular to Square):
   - Issue: Original images are rectangular (320x240), but standard CNN kernels operate efficiently on square inputs.
   - Action: Resize all images to 128x128.
3. Normalization (Pixel Scaling):
   - Issue: Raw pixel values range from [0, 255]. Large integer inputs can cause training instability (exploding gradients).
   - Action: Rescale pixel values to the range [0, 1] by dividing by 255.0.

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

### **Part 3. Convolutional Architecture Design**
To solve the classification task efficiently, I designed a 4-stage Convolutional Neural Network (CNN). The architecture follows a pyramidal design: as the network goes deeper, the spatial dimensions (H x W) decrease while the feature depth (Filters) increases.
#### **Design Decisions and Justifications**
1. Number of Convolutional Layers (4 Blocks): A configuration of 4 Blocks of `Conv2D` + `MaxPooling2D`.
   - Block 1: 32 Filters
   - Block 2: 64 Filters
   - Block 3: 128 Filters
   - Block 4: 128 Filters

   **Justification:**
   - Hierarchical Learning: The first layers capture simple low-level features (edges, cell boundaries), while deeper layers capture complex high-level structures (nucleus lobes, granule textures).
   - Parameter Efficiency: A 3-block design resulted in a "flattened" output of 16 x 16, causing a parameter explosion (~4 million parameters) in the Dense layer. Adding a 4th block reduces the spatial output to 8 x 8, reducing the parameter count by nearly 75% without losing accuracy.

2. Kernel Size (3 x 3): A fixed 3 x 3 kernels for all convolutional layers

   **Justification:**
   - Feature Locality: A 3 x 3 window is small enough to capture fine details (like the granular texture of an Eosinophil) while keeping the parameter count low.
   - Stacking Efficiency: Stacking multiple 3 x 3 layers allows the network to learn complex patterns effectively, similar to using a larger 5 x 5 or 7 x 7 kernel, but with fewer parameters and more non-linearity.

3. Stride and Padding Choices: A configuration of `stride = 1` and `padding = 'same'`

   **Justification:**
   - Preservation of Spatial Info: Using `stride=1` and `padding='same'` ensures that the convolutional layers do not reduce the image size. This decouples feature extraction from dimensionality reduction.
   - Control: We leave the reduction of image size entirely to the Pooling layers, making the architecture modular and the output shapes predictable ($128 \to 64 \to 32 \to 16 \to 8$).

4. Activation Functions: I chose a Rectified Linear Unit (ReLU) for all hidden layers, defined like this:

   $f(x) = \max(0, x)$

   **Justification:**
   - Sparsity: ReLU outputs true zero for negative values, creating sparse representations that are easier to learn.
   - Gradient Flow: Unlike Sigmoid or Tanh, ReLU does not saturate for positive values, preventing the vanishing gradient problem and allowing the model to converge much faster.

5. Pooling Strategy (Max Pooling 2 x 2): A MaxPooling2D with a 2 x 2 window applied after every convolution block.

   **Justification:**
   - Dimensionality Reduction: Each pooling step reduces the height and width by half, discarding 75% of the pixel data. This forces the model to retain only the most important features (the "max" activation).
   - Translation Invariance: By taking the maximum value in a local patch, the model recognizes a feature (e.g., a nucleus edge) regardless of its precise position within that patch. This is crucial since cells are not perfectly centered in every image.
