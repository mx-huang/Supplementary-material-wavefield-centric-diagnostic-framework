# Supplementary Materials for [Automated high-resolution characterization of concrete crack pattern using wavefield-centric diagnostic framework]

This repository contains the supplementary materials for the paper titled "[Automated high-resolution characterization of concrete crack pattern using wavefield-centric diagnostic framework]". It includes details of the finite element simulations, UWI-ConvLSTM model details and SLDV scan settings. Codes for automated modeling and data processing,  UWI-ConvLSTM model.

## 1. Finite Element Simulation

This section provides the specific details of the finite element (FE) simulations conducted in this research, including material properties, analysis step settings, meshing details, and the Python scripts used for parametric, batch-process modeling. The numerical simulations were performed using the commercial FE software Abaqus 2024, with its built-in Python interpreter for parametric modeling.

### 1.1 Model Geometry

* **Concrete Specimen:** 400 mm (Length) × 400 mm (Width) × 50 mm (Height)
* **PZT Transducer:** Cylindrical patch with a diameter of 20 mm and a height of 2 mm.

### 1.2 Material Properties

To optimize computational efficiency, the concrete was modeled as a homogeneous material. The PZT transducer was modeled as PZT-5H. The specific material properties are detailed in the tables below.

**Table 1. The material properties of concrete**

| Property | Value | Unit |
| :--- | :--- | :--- |
| ρ (Density) | 2500 | kg·m⁻³ |
| E (Young's Modulus) | 30.00 | GPa |
| υ (Poisson's Ratio) | 0.2 | |

**Table 2. The material properties of PZT**

| Property | Value | Property | Value | Property | Value |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Mechanical** | | **Piezoelectric** | (m·V⁻¹) | **Dielectric** | (f·m⁻¹) |
| ρ (Density) | 7500 kg·m⁻³ | d₃₁₁ | -2.74e⁻¹⁰ | D₁₁ | 1.505e⁻⁸ |
| E₁ | 60.61 GPa | d₃₂₂ | -2.74e⁻¹⁰ | D₂₂ | 1.505e⁻⁸ |
| E₂ | 60.61 GPa | d₃₃₃ | 5.93e⁻¹⁰ | D₃₃ | 1.301e⁻⁸ |
| E₃ | 40.31 GPa | d₁₁₂ | 7.41e⁻¹⁰ | | |
| υ₁₂ | 0.289 | d₂₂₃ | 7.41e⁻¹⁰ | | |
| υ₁₃ | 0.512 | | | | |
| υ₂₃ | 0.512 | | | | |
| G₁₂ | 23.5 GPa | | | | |
| G₁₃ | 23.0 GPa | | | | |
| G₂₃ | 23.0 GPa | | | | |

*Note: In Abaqus, the piezoelectric constant `d` and dielectric constant `D` are used to define the material properties.*

### 1.3 Analysis Step Configuration

* **Analysis Type:** The simulation was conducted using a dynamic, implicit analysis step.
* **Contact Interaction:** A "tie" constraint was used to define the contact between the concrete and the PZT transducer, ensuring a perfect bond.
* **Excitation Signal:** The excitation signal was a five-cycle sine wave with a center frequency of 120 kHz, modulated by a Hanning window.

### 1.4 Python Scripts

The Python scripts used in this study are organized into three main parts: batch modeling, data extraction, and wavefield visualization.

* **Batch Modeling:** These scripts automate the generation of models with varying crack locations. By modifying the coordinates of the crack in the script, simulations for different damage scenarios can be efficiently created.
* **Data Extraction:** During the modeling phase, a node set was created on the surface of the concrete model. The scripts then extract the velocity time-history data from the nodes within this predefined set.
* **Wavefield Visualization:** The extracted velocity time-history data is processed and plotted to generate wavefield snapshots. Interpolation techniques are employed to achieve the desired spatial resolution for the visualization.

For detailed implementation, please refer to the commented code in the ` Finite Element Simulation` directory of this repository.

## 2. UWI-ConvLSTM

This section details the proposed UWI-ConvLSTM model, including its architecture, training setup, hyperparameters, loss function, quantitative evaluation metrics, and ablation studies.

### 2.1 Model Architecture

The model architecture is detailed below. It is a sequential model consisting of three `ConvLSTM2D` layers, three `BatchNormalization` layers, and a final `Conv2D` layer. The number of filters and the kernel size for each layer are specified in the code.

```python
model = Sequential([
    ConvLSTM2D(filters=12, kernel_size=3, padding='same', return_sequences=True, input_shape=input_shape),
    BatchNormalization(),
    ConvLSTM2D(filters=6, kernel_size=3, padding='same', return_sequences=True),
    BatchNormalization(),
    ConvLSTM2D(filters=12, kernel_size=3, padding='same', return_sequences=False),
    BatchNormalization(),
    Conv2D(filters=1, kernel_size=3, activation='sigmoid', padding='same', data_format='channels_last')
])

```
### 2.2 Training Setup

To accelerate computation, all variables and operations, including forward and backward propagation, loss calculation, weight updates, and gradient computation, are configured on a Graphics Processing Unit (GPU).

* **GPU:** NVIDIA A100-PCIE-40GB
* **Software:** TensorFlow 2.9.0, Python 3.8, CUDA 11.2

### 2.3 Hyperparameters

* **`batch_size`**: 4
* **`epoch`**: 100
* **`patience`**: 20 for early stopping
* **`window_size`**: 25
* **`optimizer`**: Adam
* **`learning_rate`**: 0.001
* **`gradient_clipping`**: 1.0

Throughout the iteration process, the model with the minimum loss on the validation set is saved and used as the optimal model for final testing and subsequent experiments.

### 2.4 Loss Function

The model utilizes the **Tversky loss function**. Due to the fine nature of the cracks to be detected, this loss function is chosen to address the class imbalance by adjusting the trade-off between False Negatives (FN) and False Positives (FP) through its parameters. The specific loss function is defined as follows:

$$L_{\text{Tversky}} = 1 - \frac{\text{TP}}{\text{TP} + \alpha \cdot \text{FN} + \beta \cdot \text{FP} + \epsilon}$$

Where:
* $y$ is the ground truth.
* $\hat{y}$ is the model prediction.
* $\text{TP} = \sum_{i} y_i \hat{y}_i$ (True Positives)
* $\text{FN} = \sum_{i} y_i (1 - \hat{y}_i)$ (False Negatives)
* $\text{FP} = \sum_{i} (1 - y_i) \hat{y}_i$ (False Positives)
* $\alpha$ and $\beta$ are hyperparameters that control the weights of FN and FP.
* $\epsilon$ (epsilon) is a very small constant (e.g., $10^{-6}$) to prevent division by zero and enhance numerical stability.

In our implementation, the specific parameters are set to **$\alpha = 0.7$**, **$\beta = 0.3$**, and **$\epsilon = 10^{-6}$**.

### 2.5 Quantitative Metrics

* **Chamfer Distance (CD)**: Measures the similarity between the actual ($A$) and predicted ($B$) crack contours.
    $$D_{CD}(A, B) = \frac{1}{|A|}\sum_{a \in A} \min_{b \in B} \|a-b\| + \frac{1}{|B|}\sum_{b \in B} \min_{a \in A} \|b-a\|$$

* **Trainable Parameters (Params.)**: The total number of adjustable weights and biases in all layers of the neural network that are updated during training.

* **Average Time per Epoch (Avg-time)**: A metric used to measure the time it takes to train the model for one complete pass through the entire training dataset.

### 2.6 Ablation Study

This section details the ablation experiments conducted to evaluate the contributions of different components of the model.
