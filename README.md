# AQI Analysis and Prediction using Deep Learning

This repository contains the complete code and analysis for a project focused on predicting the Air Quality Index (AQI) in India using various deep learning models. The project involves extensive data preprocessing, exploratory data analysis (EDA), feature engineering, and the implementation and evaluation of several time-series models.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Detailed Reports](#detailed-reports)
- [Dataset](#dataset)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Modeling Approach](#modeling-approach)
  - [Unimodal Models](#unimodal-models)
  - [Hybrid Models](#hybrid-models)
- [Key Findings](#key-findings)

---

## Project Overview

The primary goal of this project is to develop a reliable system for forecasting air quality. By analyzing historical data, we identify key pollutants and patterns affecting AQI. The core of the project is a suite of deep learning models, including RNNs, LSTMs, and hybrid architectures, trained to predict AQI values based on past measurements.

## Key Features

- **In-Depth Data Analysis:** Comprehensive EDA to understand temporal, geographical, and pollutant-specific trends.
- **Advanced Feature Engineering:** Creation of temporal features and pollutant ratios to improve model performance.
- **Dimensionality Reduction:** Use of PCA to handle multicollinearity and reduce feature space.
- **Multiple Deep Learning Models:** Implementation and evaluation of RNN, LSTM, BiLSTM, CNN, and several hybrid models (e.g., CNN-LSTM, RNN-LSTM).
- **Hyperparameter Tuning:** Systematic tuning to find the optimal configuration for each model.
- **Model Interpretation:** Use of LIME to explain model predictions and build trust.

## Detailed Reports

This repository includes several detailed reports that document the project from start to finish:

- **`Comprehensive_AQI_Project_Report.md`**: The main report covering all aspects of the project.
- **`modeling_report.md`**: A deep dive into the architecture and technical details of each model.
- **`hyperparameter_tuning_report.md`**: A summary of the tuning process and results.
- **`findings_and_recommendations_report.md`**: A strategic summary of key findings and actionable recommendations.

## Dataset

To keep the repository lightweight, the raw data files are not included here. The primary dataset used is the **"Air Quality Data in India (2015-2020)"** dataset, which can be downloaded from Kaggle:

- **Link:** [https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)

After downloading, please place the `city_day.csv` and other relevant files into a directory named `archive/` in the project's root folder.

## Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

- Python 3.8+
- Pip (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Karansehgal0611/AQIresearch.git
    cd AQIresearch
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # For Linux/macOS
    python3 -m venv env
    source env/bin/activate

    # For Windows
    python -m venv env
    .\env\Scripts\activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

All the analysis, modeling, and report generation are contained within the Jupyter Notebook:

- **`aqi.ipynb`**

To run the project, simply start Jupyter and open this notebook:

```bash
jupyter notebook aqi.ipynb
```

Execute the cells sequentially to see the full workflow, from data loading to model evaluation.

## Modeling Approach

The project explores and compares several deep learning architectures for time-series forecasting.

### Unimodal Models

- **RNN:** A baseline recurrent neural network.
- **LSTM:** A more advanced RNN capable of learning long-term dependencies.
- **BiLSTM:** A bidirectional LSTM that processes sequences in both directions.
- **1D CNN:** A convolutional network used to extract local patterns from the data.

### Hybrid Models

- **CNN-LSTM & CNN-BiLSTM:** Use CNNs for feature extraction and LSTMs for sequence modeling.
- **RNN-LSTM:** The best-performing model, which uses a deep stack of RNN and LSTM layers to create a hierarchical representation of temporal features.

## Key Findings

- **Primary AQI Drivers:** `PM2.5`, `CO`, and `NO2` are the most influential pollutants.
- **Hotspots:** Pollution is most severe in cities like **Patna and Delhi**, especially during the **Winter** season.
- **Best Predictive Model:** The hybrid **RNN-LSTM** model demonstrated the highest accuracy in predicting AQI values.
- **Data Quality:** The analysis highlighted the critical need for complete and reliable data collection for future modeling improvements.
