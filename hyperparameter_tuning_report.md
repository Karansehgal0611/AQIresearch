# Report on Hyperparameter Tuning for AQI Prediction Models

## 1. Introduction to Hyperparameter Tuning

### What is Hyperparameter Tuning?

Hyperparameter tuning is the process of systematically searching for the optimal set of parameters that govern the learning process of a model. Unlike model parameters (like weights), which are learned during training, hyperparameters are set before the training begins. The goal of this process is to find a combination of hyperparameters that maximizes the model's performance on a validation dataset.

### Methodology in this Project

In this project, an automated, trial-based approach was used to tune the models. For each model architecture (e.g., LSTM, CNN-LSTM), a series of trials were run. In each trial, a different combination of hyperparameters was selected from a predefined search space. The performance of each combination was evaluated based on its **validation loss (Mean Squared Error)**, and the set of parameters that yielded the lowest validation loss was identified as the optimal configuration.

---

## 2. Tuned Hyperparameters (Search Space)

The following hyperparameters were systematically tuned across the various models to find the best-performing configurations:

- **`learning_rate`:** Controls how much the model's weights are updated during training. A range from `0.0001` to `0.001` was explored.
- **`dropout_rate`:** The fraction of neurons to drop during training to prevent overfitting. Values typically ranged from `0.2` to `0.5`.
- **`batch_size`:** The number of samples processed before the model is updated. Common values tested were `16`, `32`, and `64`.
- **`activation_function`:** The function used to introduce non-linearity. `relu` and `tanh` were the primary choices for RNN and LSTM layers.
- **`units/filters`:** The number of neurons in a layer (for RNN/LSTM/Dense) or the number of filters (for CNN). This determines the layer's capacity.
- **`recurrent_dropout`:** A specific type of dropout applied to the recurrent connections in LSTMs, crucial for reducing overfitting in time-series tasks.
- **`merge_mode` (for BiLSTM):** The method for combining the outputs of the forward and backward passes. `concat` and `sum` were evaluated.

---

## 3. Tuning of Unimodal Models

### 3.1. RNN (Simple Recurrent Neural Network)
- **Findings:** The RNN model was sensitive to the learning rate and the number of units. Higher learning rates often led to unstable training. The best performance was found with a moderate learning rate and a deeper architecture.
- **Best Configuration:**
  - **Validation Loss:** 2106.98
  - **Learning Rate:** `0.0005`
  - **RNN Units:** `[256, 128]`
  - **Dropout:** `0.2`
  - **Batch Size:** `32`

### 3.2. LSTM (Long Short-Term Memory)
- **Findings:** LSTMs benefited from deeper architectures and the use of `recurrent_dropout`. The `tanh` activation function proved effective. Smaller batch sizes (`16`) allowed for more stable learning compared to larger ones.
- **Best Configuration:**
  - **Validation Loss:** 2046.70
  - **Learning Rate:** `0.001`
  - **LSTM Units:** `[256, 128, 64]`
  - **Recurrent Dropout:** `0.2`
  - **Batch Size:** `16`

### 3.3. BiLSTM (Bidirectional LSTM)
- **Findings:** The BiLSTM models consistently outperformed their unidirectional counterparts. The choice of `merge_mode` was significant, with `sum` providing a slight edge in the best-performing trial by creating a more compact representation. A smaller batch size (`16`) was again favored.
- **Best Configuration:**
  - **Validation Loss:** 1996.05
  - **Learning Rate:** `0.0005`
  - **LSTM Units:** `[256, 128]`
  - **Merge Mode:** `sum`
  - **Batch Size:** `16`

### 3.4. CNN (1D Convolutional Neural Network)
- **Findings:** The CNN's performance was heavily dependent on the number of filters, kernel size, and dense layer units. The best results came from a deeper network with more filters and a larger dense layer for final predictions.
- **Best Configuration:**
  - **Validation Loss:** 2274.01
  - **Learning Rate:** `0.001`
  - **Convolutional Layers:** 3 (with 32 filters each)
  - **Dense Units:** `256`
  - **Batch Size:** `32`

---

## 4. Tuning of Hybrid Models

The tuning of hybrid models involved finding the right balance between the parameters of their constituent parts (e.g., CNN filters and LSTM units).

### 4.1. CNN-LSTM
- **Findings:** The key was to balance the feature extraction capacity of the CNN with the sequence modeling capacity of the LSTM. The best model used a relatively high number of both CNN filters and LSTM units.
- **Best Configuration (from Bayesian Optimization):**
  - **Validation Loss:** 2129.17
  - **Learning Rate:** `~0.0014`
  - **CNN Filters:** `75`
  - **LSTM Units:** `140`
  - **Dense Units:** `221`
  - **Dropout:** `0.21`

### 4.2. CNN-BiLSTM
- **Findings:** Similar to the CNN-LSTM, this model's performance depended on the synergy between its convolutional and recurrent parts. A lower learning rate (`0.0005`) proved more effective.
- **Best Configuration:**
  - **Validation Loss:** 2200.39
  - **Learning Rate:** `0.0005`
  - **CNN Filters:** `64`
  - **LSTM Units:** `64`
  - **Batch Size:** `64`

### 4.3. RNN-LSTM
- **Findings:** This model, which emerged as the overall best performer, benefited from a deep stack of RNN layers followed by a powerful LSTM layer. The `relu` activation was found to be highly effective in the RNN layers.
- **Best Configuration:**
  - **Validation Loss:** 1993.73
  - **Learning Rate:** `0.001`
  - **RNN Layers:** 3 (with 256 units each)
  - **LSTM Units:** `128`
  - **Dropout:** `0.3`
  - **Batch Size:** `32`

### 4.4. RNN-CNN
- **Findings:** This unconventional architecture was the most challenging to tune. The results suggest that applying convolutions after the recurrent processing is less intuitive and harder to optimize. The best trial still lagged behind other hybrid models.
- **Best Configuration:**
  - **Validation Loss:** 2618.24
  - **Learning Rate:** `0.001`
  - **RNN Layers:** 3 (with 128 units each)
  - **CNN Filters:** `32`
  - **Batch Size:** `64`

## 5. Conclusion

Hyperparameter tuning was a critical phase of this project. The process revealed several key insights:

- **No One-Size-Fits-All:** The optimal hyperparameters were unique to each architecture, highlighting the importance of individual tuning.
- **Performance Gains:** Tuning led to significant improvements in validation loss compared to baseline or default parameters.
- **Model Complexity:** The best-performing models (RNN-LSTM, BiLSTM) were those with a higher capacity to model complex temporal dependencies, and tuning was essential to control this complexity and prevent overfitting.

The results from the numerous trials, logged in `all_hyperparameter_results.csv`, were instrumental in identifying the architectural choices and learning parameters that led to the most accurate AQI prediction models.
