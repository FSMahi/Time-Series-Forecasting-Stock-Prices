
# Time Series Forecasting: Stock Prices

This repository contains a **comprehensive analysis and modeling of stock prices** using time series forecasting techniques, including **ARIMA** and **Stacked LSTM** models. The notebook demonstrates **data preprocessing, model training, evaluation, and comparison**, providing insights into which model generalizes better for stock price prediction.

---

## **Table of Contents**

* [Dataset](#dataset)
* [Objectives](#objectives)
* [Approach](#approach)
* [Models Used](#models-used)
* [Performance Metrics](#performance-metrics)
* [Results & Analysis](#results--analysis)
* [Visualizations](#visualizations)
* [Dependencies](#dependencies)
* [Usage](#usage)
* [Author](#author)

---

## **Dataset**

The dataset is sourced from **Yahoo Finance** using the `yfinance` library. It contains historical stock prices with daily records, including:

* Date
* Open, High, Low, Close prices
* Volume

The notebook uses the **Close price** as the target variable for forecasting.

---

## **Objectives**

* Forecast future stock prices using **time series models**.
* Compare **ARIMA** (linear) and **Stacked LSTM** (nonlinear, deep learning) performance.
* Analyze which model generalizes better based on evaluation metrics.
* Visualize actual vs predicted stock prices.

---

## **Approach**

1. **Data Preprocessing:**

   * Download historical stock price data.
   * Handle missing values and visualize trends.
   * Scale the data for LSTM input using `MinMaxScaler`.

2. **Feature Engineering:**

   * Create sequences of past observations (`lookback window`) for LSTM.

3. **Model Training:**

   * **ARIMA:** Fit on historical data to capture linear trends.
   * **Stacked LSTM:** Two LSTM layers followed by Dense layers to capture nonlinear and long-term dependencies.

4. **Evaluation:**

   * Metrics: **RMSE, MAPE, SMAPE**
   * Plot actual vs predicted prices.
   * Compare model generalization.

---

## **Models Used**

1. **ARIMA (AutoRegressive Integrated Moving Average):**

   * Suitable for linear, stationary time series.
   * Captures trends and seasonality.

2. **Stacked LSTM (Long Short-Term Memory):**

   * Deep learning model for sequential data.
   * Can model nonlinear patterns and long-term dependencies.

---

## **Performance Metrics**

| Model | RMSE  | MAPE   | SMAPE  |
| ----- | ----- | ------ | ------ |
| ARIMA | 1.98  | 1.28%  | 1.31%  |
| LSTM  | 27.19 | 11.62% | 12.46% |

**Observation:** ARIMA performs better on this dataset due to the strong linear trend, while LSTM could improve with more data and tuning.

---

## **Results & Analysis**

* **ARIMA:** Better generalization with low error metrics.
* **LSTM:** Higher errors, possibly due to dataset simplicity or insufficient hyperparameter tuning.

---

## **Visualizations**

### 1. Stock Price Trend

![Stock Price Trend](path_to_your_image/stock_trend.png)

### 2. ARIMA Model Predictions vs Actual

![ARIMA Predictions](path_to_your_image/arima_predictions.png)

### 3. LSTM Model Predictions vs Actual

![LSTM Predictions](path_to_your_image/lstm_predictions.png)

### 4. Comparison of ARIMA and LSTM

![Model Comparison](path_to_your_image/model_comparison.png)

> **Note:** Replace `path_to_your_image/...` with actual paths to screenshots saved in your repository (e.g., `images/arima_plot.png`).

---

## **Dependencies**

* Python 3.x
* `pandas`, `numpy`, `matplotlib`, `seaborn`
* `yfinance`
* `statsmodels` (for ARIMA)
* `tensorflow` / `keras` (for LSTM)
* `scikit-learn` (for scaling & metrics)

Install dependencies via:

```bash
pip install pandas numpy matplotlib seaborn yfinance statsmodels tensorflow scikit-learn
```

---

## **Usage**

1. Clone the repository:

```bash
git clone https://github.com/FSMahi/Time-Series-Forecasting-Stock-Prices.git
```

2. Open the notebook:

```bash
Stock_Market_Analysis (1).ipynb
```

3. Run the notebook cells step by step to:

   * Download stock price data
   * Preprocess and visualize the data
   * Train ARIMA and LSTM models
   * Evaluate performance and compare results

---

## **Author**

**Fahmida Sultana** – MS.c Student, ICT, Jahangirnagar University

* Interested in Machine Learning, Deep Learning, and Time Series Analysis
* GitHub: [FSMahi](https://github.com/FSMahi)

---

