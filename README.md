# Energy Consumption Forecaster ⚡

This project delivers an end-to-end Machine Learning and Deep Learning time-series forecasting system designed to predict smart home energy consumption 24 hours into the future. By evaluating advanced ensemble methods and a deep Sequence-to-Sequence LSTM architecture on chronological data, the system establishes a robust pipeline for optimized grid load management and automated smart home scheduling.

##  Project Resources & Demo
* **[Video Presentation on YouTube](https://youtu.be/wvivLxtCHig)**
* **[Project Dataset on Google Drive](https://drive.google.com/drive/folders/15Bled-ImrJ3UeZizZZQqNb8JCmB_UovS?usp=sharing)**

---

##  Installation & Setup Guide

Follow these steps to set up the environment and run the project locally:

1. **Clone the repository:**
   git clone https://github.com/SimonZoks/energy-consumption-forecaster.git
   cd energy-consumption-forecaster

2. **Install the required dependencies:**
   Make sure you are using Python 3.12, then run:
   pip install pandas numpy scikit-learn xgboost tensorflow matplotlib gdown

3. **Launch the Jupyter Notebook:**
   jupyter notebook notebook.ipynb
   
   Note: Open the notebook in VS Code or Jupyter, select your Python 3.12 kernel, and execute the cells sequentially from top to bottom.

##  Model Comparison & Results

The models were evaluated using a strict Walk-Forward Cross-Validation strategy to prevent data leakage. The evaluation metrics on the test set are summarized below:

* **Random Forest**
  - Test MAE: 0.6974 kW
  - Test MSE: 1.0967 kW²
  - Key Characteristics: Strong baseline using engineered cyclic time features.

* **XGBoost**
  - Test MAE: 0.7011 kW
  - Test MSE: 1.1091 kW²
  - Key Characteristics: Fast, robust gradient boosting performance on linear/non-linear trends.

* **Multi-Step Seq2Seq LSTM**
  - Test MAE: 0.7077 kW
  - Test MSE: 1.1118 kW²
  - Key Characteristics: Full 24-hour multi-step horizon prediction with healthy convergence.

##  Key Conclusion

While classical ensemble models demonstrate exceptional precision for short-term steps, the Sequence-to-Sequence LSTM proves to be the most viable solution for complex 24-hour ahead forecasting, though its tendency to smoothen stochastic spikes highlights the critical necessity of integrating future exogenous weather variables to fully optimize predictive performance.
Што да направиш сега:
Залепи го ова вака како што е.
