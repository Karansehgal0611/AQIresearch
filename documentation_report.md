# Air Quality Index (AQI) Prediction and Analysis Documentation

## 1. Project Overview

This project focuses on analyzing historical air quality data for various cities in India and developing a suite of deep learning models to predict the Air Quality Index (AQI). The primary goal is to understand the relationships between different pollutants, identify temporal and geographical patterns, and build accurate predictive models for forecasting air quality.

**Dataset:** The project utilizes the `city_day.csv` dataset, which contains daily air quality measurements for 26 cities from 2015 to 2020.

## 2. Data Preprocessing and Feature Engineering

The initial phase of the project involved extensive data cleaning and preparation to create a robust dataset for modeling.

- **Data Cleaning:**
  - The `Date` column was converted to a datetime object.
  - Duplicate rows were checked and removed.
  - The `Xylene` column was dropped due to having over 60% missing values.

- **Missing Value Imputation:**
  - Missing values for pollutant features (`PM2.5`, `PM10`, `NO`, etc.) were imputed using the median value for each respective city. This approach preserves the central tendency of the data for each specific location.
  - Rows where both `AQI` and `AQI_Bucket` were missing were dropped.

- **Outlier Handling:**
  - Outliers were detected for each pollutant using the Interquartile Range (IQR) method.
  - To mitigate the effect of extreme values, outliers were capped at the 99th percentile for each pollutant.

- **Feature Engineering:**
  - **Temporal Features:** New features were extracted from the `Date` column to capture time-based patterns:
    - `Year`, `Month`, `Day`, `DayOfWeek`
    - `Season` (1: Winter, 2: Spring, 3: Summer, 4: Fall)
  - **Pollutant Ratios:** Ratios between related pollutants were created to capture chemical interactions:
    - `PM_Ratio` (PM2.5 / PM10)
    - `NOx_Ratio` (NO2 / NO)
    - `SO2_NO2_Ratio` (SO2 / NO2)

- **Categorical Encoding:**
  - **Target Variable:** The `AQI_Bucket` (e.g., 'Good', 'Moderate') was label-encoded into numerical values (0-5).
  - **City:** The `City` column was one-hot encoded to be used as a feature in the models, allowing the models to learn city-specific patterns.

## 3. Exploratory Data Analysis (EDA)

EDA was performed to uncover insights and patterns within the data.

- **Temporal Trends:**
  - **Yearly:** A general trend of improving air quality was observed, with average AQI decreasing from 2015 to 2020.
  - **Seasonal:** Air quality is significantly worse during the **Winter** season, likely due to meteorological conditions trapping pollutants.

- **City-wise Analysis:**
  - The most polluted cities in terms of average PM2.5 are **Patna, Delhi, and Gurugram**.
  - Cities like Aizawl and Shillong show significantly better air quality.

- **Correlation Analysis:**
  - `AQI` shows the strongest positive correlation with `PM2.5`, `CO`, and `NO2`. This indicates that these pollutants are primary drivers of high AQI values.

## 4. Dimensionality Reduction with PCA

Principal Component Analysis (PCA) was used to reduce the dimensionality of the pollutant features and mitigate multicollinearity.

- **Explained Variance:**
  - The first 5 principal components (PCs) explain approximately 78% of the variance in the data.
  - **8 components** were chosen to retain over **92%** of the variance, providing a good balance between dimensionality reduction and information preservation.

- **Component Interpretation:**
  - **PC1:** Represents overall pollution intensity, with high positive loadings from `NO2`, `NOx`, and `NO`.
  - **PC2:** Appears to represent a contrast between combustion-related pollutants (`CO`, `SO2`) and particulate matter (`PM2.5`, `PM10`).

## 5. Predictive Modeling

A variety of deep learning models were implemented to predict the `AQI` value. The data was structured into sequences (timesteps) to be used in time-series models.

The following model architectures were developed and evaluated:
- **SimpleRNN:** A basic Recurrent Neural Network.
- **LSTM:** Long Short-Term Memory network, capable of learning long-term dependencies.
- **BiLSTM:** Bidirectional LSTM, which processes the sequence in both forward and backward directions.
- **CNN:** A 1D Convolutional Neural Network, effective at extracting features from sequences.
- **Hybrid Models:**
  - **CNN-LSTM:** Combines a CNN for feature extraction with an LSTM for sequence prediction.
  - **CNN-BiLSTM:** Similar to CNN-LSTM but uses a Bidirectional LSTM layer.
  - **RNN-LSTM:** A hybrid model combining SimpleRNN and LSTM layers.
  - **RNN-CNN:** A hybrid model combining SimpleRNN and CNN layers.

## 6. Model Performance and Evaluation

The models were trained and evaluated based on their performance on the test set. The primary metrics used were Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared (R2).

| Model         | Best Validation Loss (MSE) |
|---------------|----------------------------|
| **RNN-LSTM**  | **1993.73**                |
| BiLSTM        | 1996.05                    |
| LSTM          | 2046.70                    |
| RNN           | 2106.98                    |
| CNN-BiLSTM    | 2200.39                    |
| CNN-LSTM      | 2283.55                    |
| CNN           | 2274.01                    |
| RNN-CNN       | 2618.24                    |

**Best Performing Model:** The **RNN-LSTM** hybrid model achieved the lowest validation loss, indicating it was the most accurate model for predicting AQI in this study.

## 7. Hyperparameter Tuning

Extensive hyperparameter tuning was performed for each model to find the optimal combination of parameters. This involved experimenting with:
- Learning Rate
- Dropout Rate
- Number of Layers and Units (e.g., LSTM units, CNN filters)
- Batch Size
- Activation Functions

The `hybrid_models_summary.csv` and other `_tuning_results.csv` files show that the best parameters varied for each model architecture, but the tuning process was crucial in achieving the performance reported above. For instance, the best RNN-LSTM model used a learning rate of `0.001`, a batch size of `32`, and a dropout rate of `0.3`.

## 8. Model Interpretation with LIME

LIME (Local Interpretable Model-agnostic Explanations) was used to understand the predictions of the best-performing models (CNN-LSTM, CNN-BiLSTM, RNN-LSTM). The LIME explanation plots (e.g., `lime_explanation_CNN-BiLSTM.png`) show which input features were most influential for a specific prediction. This adds a layer of transparency to the "black box" nature of deep learning models, helping to build trust and understand what drives the model's forecasts.

## 9. Conclusion and Recommendations

- **Conclusion:** This project successfully demonstrated the entire machine learning pipeline from data cleaning to model deployment. The analysis revealed significant patterns in air quality across India. The hybrid **RNN-LSTM** model proved to be the most effective at predicting AQI.
- **Recommendations:**
  - **Focus on Key Pollutants:** Mitigation strategies should focus on `PM2.5`, `CO`, and `NO2`, as they are the strongest drivers of AQI.
  - **Seasonal Action:** Implement stricter pollution controls during the winter months when AQI is at its worst.
  - **Model Deployment:** The trained RNN-LSTM model is a strong candidate for deployment in a real-world air quality forecasting system.
  - **Further Research:** Future work could involve incorporating meteorological data, which is known to heavily influence air quality.

## 10. How to Use This Project

1.  **Environment Setup:** Ensure you have Python and the required libraries (`pandas`, `numpy`, `scikit-learn`, `tensorflow`, `matplotlib`, `seaborn`) installed.
2.  **Run the Notebook:** Execute the cells in the `aqi.ipynb` notebook sequentially.
3.  **Review Results:** The notebook will generate all the analysis, visualizations, and save the model results to the corresponding CSV files. The final trained models and processed data are also saved.
