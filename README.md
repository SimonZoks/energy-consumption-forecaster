# ⚡ Energy Consumption Forecaster

> **Domain:** Smart Home / Energy Management  
> **Task:** Multi-Step Time Series Forecasting (ML + Deep Learning)  
> **Institution:** Avenga Academy — Capstone Project  

---

## 📌 Project Overview

Accurately predicting household electrical energy consumption is a critical component of modern **Smart Grids** and dynamic energy management systems. Unpredicted consumption spikes create operational inefficiencies, higher costs, and risks of grid instability.

This project implements an end-to-end Machine Learning & Deep Learning forecasting pipeline using the **Individual Household Electric Power Consumption Dataset**. It compares classic tree-based algorithms with an **Encoder-Decoder (Seq2Seq) LSTM** architecture designed to predict multi-step energy profiles ($t+1$ to $t+24$ hours) into the future.

---

## 🛠️ Key Technical Features

* **Data Preprocessing & Resampling:** Cleaning, linear interpolation of missing values, and resampling minute-level raw consumption metrics to hourly aggregates ($1\text{h}$ mean).
* **Advanced Feature Engineering:**
  * **Lag Features:** $t-1, t-2, t-3, t-24$ (yesterday), $t-168$ (last week).
  * **Rolling Window Statistics:** 24-hour moving mean and standard deviation (`rolling_mean_24h`, `rolling_std_24h`).
  * **Cyclic Temporal Encoding:** Sine and Cosine trigonometric transformations for hourly ($0\text{--}23$) and weekly ($0\text{--}6$) cycles to ensure smooth temporal transitions between continuous periods (e.g., hour $23$ to hour $0$).
* **Cross-Validation:** 5-fold **Walk-Forward (TimeSeriesSplit)** validation to guarantee strict chronological integrity without data leakage.
* **Multi-Step Deep Learning Forecast:** Sequence-to-Sequence (Seq2Seq) LSTM network predicting a 24-hour continuous horizon simultaneously.
* **Theoretical Research:** Conceptual comparative evaluation of the **N-BEATS** architecture for interpretable time-series decomposition.

---

## 🏗️ System & Model Architecture

### 1. Classical Machine Learning Pipeline
Evaluated using 5-Fold Walk-Forward Cross-Validation:
* **Random Forest Regressor** (`n_estimators=100`)
* **Gradient Boosting Regressor** (`n_estimators=100`)
* **XGBoost Regressor** (`learning_rate=0.05`, `n_estimators=100`)

### 2. Seq2Seq LSTM Architecture
Designed for continuous 24-hour sequence output:
* **Input Layer:** Historical 24-hour energy sequence shape `(24, 1)`.
* **Encoder LSTM:** 64 units with ReLU activation and 0.2 Dropout.
* **Repeat Vector:** Bridges encoder to decoder across 24 output time steps.
* **Decoder LSTM:** 64 units with ReLU activation and 0.2 Dropout.
* **TimeDistributed Dense:** Dense layer outputting single kW values per hour `(24, 1)`.

---

## 📊 Experimental Results

### Classical ML Comparison (5-Fold Walk-Forward CV)

| Model | CV MAE ($\text{kW}$) | CV RMSE ($\text{kW}$) | CV $R^2$ |
| :--- | :---: | :---: | :---: |
| **XGBoost** | **0.4562** | **0.6397** | **0.4969** |
| **Random Forest** | 0.4638 | 0.6441 | 0.4665 |
| **Gradient Boosting** | 0.5470 | 0.7606 | 0.2879 |

### Multi-Step Seq2Seq LSTM Performance (24-Hour Horizon)

| Metric | Test Value ($\text{kW}$) |
| :--- | :---: |
| **LSTM Test MAE** | **0.6872 kW** |
| **LSTM Test RMSE** | **0.8406 kW** |

> **Result Insights:** While single-step classical tree models yield lower instantaneous error metrics, the Seq2Seq LSTM retains dynamic temporal context across the entire 24-hour horizon without error propagation. It accurately models base power loads ($\sim 0.5\text{--}1.0\text{ kW}$), though isolated high-power appliance spikes introduce variance.

---

## 🔬 Research Focus: N-BEATS Architecture

As part of the project's theoretical evaluation, **N-BEATS (Neural Basis Expansion Analysis for Interpretable Time Series)** was researched as an advanced alternative to RNN/LSTM models:

* **Key Innovation:** Uses fully connected (MLP) residual blocks with backward and forward residual connections (backcast/forecast), avoiding recurrent loops or self-attention mechanisms.
* **Interpretable Decomposition:**
  * **Trend Blocks:** Constrained by low-degree polynomials to capture monotonic long-term shifts.
  * **Seasonality Blocks:** Constrained by periodic Fourier series to isolate recurring daily and weekly cycles.
* **Smart Grid Application:** Allows grid operators to interpret *why* a specific peak forecast occurred, decomposing predicted loads into structural baseline trends and diurnal consumption patterns.

---

## 🚀 Production & Deployment Vision

A production-ready microservice architecture strategy for this pipeline includes:

1. **REST API Service:** Built with FastAPI to handle incoming client or grid data requests.
2. **Feature Pipeline:** Real-time generation of 24-hour lag and rolling window features from operational streams using Pandas.
3. **Inference Engine:** Loads pre-trained model weights saved in modern `.keras` format to generate fast predictions.
4. **Containerization:** Packaged inside a Docker container for consistent deployment across cloud infrastructure.

---

## 💻 Quick Start & Running the Notebook

### Prerequisites
* Python 3.10+
* TensorFlow 2.11+
* `xgboost`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`

### Installation & Execution

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SimonZoks/energy-consumption-forecaster.git


---

1. Install dependencies:
pip install -r requirements.txt


2. Prepare the Data Directory:
Place `household_power_consumption.txt` inside the `data/` subfolder:

   data/
     └── household_power_consumption.txt
   main_notebook.ipynb
   README.md


3. Run Jupyter Notebook:
jupyter notebook main_notebook.ipynb
