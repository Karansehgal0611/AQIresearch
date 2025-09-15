# In-Depth Report on Deep Learning Models for AQI Prediction

## 1. Introduction

This report provides a detailed examination of the various deep learning architectures implemented and evaluated for the task of Air Quality Index (AQI) prediction. All models were designed for a regression task to predict the numerical AQI value based on a sequence of historical pollution and temporal data.

### Common Technical Details

Across all models, a common set of tools and methodologies were employed:

- **Input Data Shape:** The models were trained on sequential data with a shape of `(samples, timesteps, features)`. This structure is essential for time-series forecasting with recurrent and convolutional networks.
- **Loss Function:** **Mean Squared Error (MSE)** was used as the loss function, which is standard for regression tasks as it heavily penalizes larger errors.
- **Optimizer:** The **Adam** optimizer was used for its adaptive learning rate capabilities, which helps in efficient convergence.
- **Callbacks for Training:**
  - **Early Stopping:** To prevent overfitting, training was set to stop automatically if the validation loss did not improve for a specified number of epochs (patience).
  - **ReduceLROnPlateau:** The learning rate was automatically reduced if the validation loss plateaued, allowing the model to fine-tune its weights in a more stable manner.

---

## 2. Baseline Models

These models represent foundational deep learning architectures for sequence data.

### 2.1. Simple Recurrent Neural Network (RNN)

- **Concept:** The Simple RNN is the most basic recurrent network. It processes sequences by maintaining a hidden state that captures information from previous timesteps. However, it often struggles with long-term dependencies due to the vanishing gradient problem.
- **Tuned Architecture:**
  - The model consisted of stacked `SimpleRNN` layers (e.g., 2 layers with 256 and 128 units).
  - A `Dropout` layer was applied after the RNN layers to reduce overfitting.
  - A final `Dense` layer with a single neuron produced the continuous AQI prediction.
- **Performance:** The best RNN model achieved a validation loss of **2106.98**, demonstrating baseline capability but was outperformed by more complex architectures.

### 2.2. Long Short-Term Memory (LSTM)

- **Concept:** LSTM networks are an advanced type of RNN designed to overcome the short-term memory issue. They use a gating mechanism (input, forget, and output gates) to regulate the flow of information, allowing them to learn and remember patterns over long sequences.
- **Tuned Architecture:**
  - The architecture involved multiple stacked `LSTM` layers (e.g., 3 layers with 256, 128, and 64 units).
  - `recurrent_dropout` was used within the LSTM layers, a specific form of dropout that helps prevent overfitting in recurrent networks.
  - A `Dropout` layer followed the LSTM block before the final `Dense` output layer.
- **Performance:** The best LSTM model achieved a validation loss of **2046.70**, showing a notable improvement over the Simple RNN.

### 2.3. Bidirectional LSTM (BiLSTM)

- **Concept:** A BiLSTM enhances the standard LSTM by processing the input sequence in two directions: forward (from start to end) and backward (from end to start). This allows the network to have context from both past and future timesteps at any given point, often leading to better performance.
- **Tuned Architecture:**
  - The core of the model used `Bidirectional(LSTM(...))` layers.
  - The outputs from the forward and backward passes were combined using a `merge_mode` (both `sum` and `concat` were tested, with `sum` performing slightly better in the best model).
  - The best-performing model had two BiLSTM layers and used a `sum` merge mode.
- **Performance:** The BiLSTM achieved a validation loss of **1996.05**, outperforming the unidirectional LSTM by effectively leveraging future context.

### 2.4. 1D Convolutional Neural Network (CNN)

- **Concept:** While typically used for image data, 1D CNNs are highly effective for sequence data, including time series. They use convolutional filters to slide across the sequence and extract local patterns or features (e.g., a pollution spike over a few days). `MaxPooling1D` is then used to downsample the feature maps, retaining the most important information.
- **Tuned Architecture:**
  - The model started with multiple `Conv1D` layers (e.g., 3 layers with 32 filters and a kernel size of 5).
  - Each `Conv1D` layer was followed by a `MaxPooling1D` layer to reduce dimensionality.
  - A `Flatten` layer was used to convert the 2D feature maps into a 1D vector.
  - `Dense` layers, along with `Dropout`, were used to make the final prediction.
- **Performance:** The best CNN model achieved a validation loss of **2274.01**. While not the best overall, it demonstrates that pattern extraction is a valid approach for this problem.

---

## 3. Hybrid Models

Hybrid models were developed to combine the strengths of different architectures, aiming for superior performance.

### 3.1. CNN-LSTM & CNN-BiLSTM

- **Concept:** This architecture uses the CNN's ability for robust feature extraction and the LSTM/BiLSTM's strength in interpreting temporal sequences. The `Conv1D` layers first act as a feature engineering module, identifying relevant local patterns in the time-series data. The output of this convolutional block is then fed into the LSTM or BiLSTM layers to model the temporal relationships between these extracted features.
- **Tuned Architecture (CNN-BiLSTM):**
  - An initial `Conv1D` layer with 64 filters.
  - A `MaxPooling1D` layer.
  - The output was passed to a `Bidirectional(LSTM(...))` layer with 64 units.
  - A final `Dense` block with `Dropout` for the prediction.
- **Performance:**
  - **CNN-LSTM:** 2283.55 (validation loss)
  - **CNN-BiLSTM:** 2200.39 (validation loss)

### 3.2. RNN-LSTM

- **Concept:** This model creates a deep, hierarchical recurrent architecture. The initial `SimpleRNN` layers capture basic temporal patterns, and their output sequences are then passed to `LSTM` layers, which can model more complex, long-term dependencies based on the features already processed by the RNNs.
- **Tuned Architecture:**
  - The model stacked three `SimpleRNN` layers with `relu` activation.
  - The output sequence was then processed by an `LSTM` layer with 128 units.
  - A `Dense` block with `Dropout` produced the final output.
- **Performance:** The RNN-LSTM model was the **best-performing model overall**, achieving a validation loss of **1993.73**. This suggests that a deep, hierarchical processing of temporal features is highly effective for this dataset.

### 3.3. RNN-CNN

- **Concept:** This is a more experimental architecture where the sequence is first processed by `SimpleRNN` layers. The output of the RNNs (a sequence of hidden states) is then treated as a feature map and passed to `Conv1D` layers to extract higher-level patterns from the temporal features themselves.
- **Tuned Architecture:**
  - Three `SimpleRNN` layers with `tanh` activation.
  - A `Conv1D` layer with 32 filters.
  - A `MaxPooling1D` and `Flatten` layer.
  - A final `Dense` block for prediction.
- **Performance:** This model achieved a validation loss of **2618.24**, making it the least performant of the hybrid models. This indicates that applying convolutions *after* recurrent processing was less effective than the CNN-first approach.

## 4. Summary of Model Performance

| Model         | Best Validation Loss (MSE) | Key Architectural Feature                                  |
|---------------|----------------------------|------------------------------------------------------------|
| **RNN-LSTM**  | **1993.73**                | Deep recurrent hierarchy (RNN followed by LSTM)            |
| BiLSTM        | 1996.05                    | Processes sequence in both forward and backward directions |
| LSTM          | 2046.70                    | Standard recurrent model with memory gates                 |
| RNN           | 2106.98                    | Basic recurrent model                                      |
| CNN-BiLSTM    | 2200.39                    | CNN for feature extraction, BiLSTM for sequence modeling   |
| CNN           | 2274.01                    | Extracts local patterns using 1D convolutions              |
| CNN-LSTM      | 2283.55                    | CNN for feature extraction, LSTM for sequence modeling     |
| RNN-CNN       | 2618.24                    | RNN for sequence modeling, CNN for feature extraction      |

**Conclusion:** The results clearly indicate that hybrid models, particularly the **RNN-LSTM**, provide the best performance. This architecture's success is likely due to its ability to create a hierarchical representation of temporal features, where initial RNN layers capture simpler patterns and subsequent LSTM layers model more complex, long-range dependencies.
