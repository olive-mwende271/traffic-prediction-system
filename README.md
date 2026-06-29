# 🚦 Traffic Prediction Analysis System

**An end-to-end hybrid time series forecasting project** for predicting hourly traffic volume on the Metro Interstate  highway.

This project combines **statistical modeling** (ARIMA & SARIMAX) with **deep learning** (Bidirectional LSTM) into a powerful **Hybrid SARIMAX-LSTM** model that significantly outperforms individual approaches by leveraging weather, temporal patterns, and holidays.

live demo  [https://traffic-prediction-system-7if7ok6uw7dcve3wxcpbdv.streamlit.app/]
---

## 📌 Project Overview

A complete traffic volume forecasting system featuring:
- Comprehensive Exploratory Data Analysis (EDA)
- Multiple time series models (ARIMA, SARIMAX, LSTM)
- Advanced **Hybrid SARIMAX-LSTM** architecture (Best performing)
- Interactive **Streamlit Web Application** (`app.py`)
- 12+ publication-quality visualizations
- Rigorous 5-Fold Time Series Cross-Validation

---

## ✨ Key Features

- Full data exploration and visualization
- Advanced feature engineering (cyclic time features, weather transformations, holiday/rush hour flags)
- Stationarity testing, seasonal decomposition, ACF/PACF analysis
- Four models compared: ARIMA, SARIMAX, Multivariate LSTM, and Hybrid SARIMAX-LSTM
- Perturbation-based feature importance analysis
- Interactive Streamlit dashboard for model demonstration

---

## 🏗️ Hybrid SARIMAX-LSTM Architecture (Best Model)

**Three complementary information streams:**

- **Branch A (Sequence)**: Bidirectional LSTM with 24-hour lookback on all multivariate features
- **Branch B (SARIMAX)**: Statistical seasonal forecast injected as a learned feature
- **Branch C (Context)**: Current weather, time-of-day, day-of-week, and holiday snapshot

---


## 🗂️ Project Structure

```bash
traffic-prediction-system/
├── app.py                        
├── requirements.txt
├── README.md
├── notebooks/
│   └── TRAFFIC_PREDICTION_ANALYSIS_SYSTEM.ipynb
└── figures/                      
