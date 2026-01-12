# Lahore Air Guard | AI Smog Intelligence Engine

**A real-time predictive dashboard for monitoring Air Quality (PM2.5) using Satellite Data and Statistical Machine Learning.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://lahoreaqiprediction-xmuha45svcjj4crqyyt2w4.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview
Lahore Air Guard is an advanced intelligence tool designed to tackle the air pollution crisis in South Asia. Unlike standard weather apps that only show *current* readings, this engine uses an **ARIMA (AutoRegressive Integrated Moving Average)** model to forecast pollution trends **24 hours into the future**.

While the model parameters `(8, 0, 1)` are fine-tuned for the heavy, stagnant smog of **Lahore**, the underlying engine is universal allowing users to track and predict air quality for **any city in the world** (New York, London, Tokyo, etc.) via the Custom Search feature.

## Key Features

* **Live Satellite Integration:** Fetches real-time PM2.5 and AQI data using the **Open-Meteo API** (Global Coverage).
* **Dual-AI Engine:**
    * **Validation Mode:** The model "backcasts" 1 hour into the past to predict the *current* moment. Comparing this against the real satellite reading proves the model's accuracy in real-time.
    * **Forecasting Mode:** Predicts hourly pollution levels for the next 24 hours.
* **Universal Search:** Capable of Geocoding any address globally to generate instant reports.
* **Live Heatmap:** Visualizes pollution intensity using an interactive Plotly density map.
* **Automated Reporting:** Generates a downloadable, signed **PDF Intelligence Report** for administrative or personal use.
* **Smart Advisory:** Automatically issues government-standard advisories (e.g., "Lockdown Likely", "Wear Mask") based on predicted severity.

## Tech Stack

* **Frontend:** Streamlit (Python) for the Glassmorphism UI.
* **Data Source:** Open-Meteo Air Quality API.
* **Machine Learning:** `statsmodels` (ARIMA Time-Series Forecasting).
* **Visualization:** Plotly Express & Graph Objects.
* **Geospatial:** Geopy (Nominatim) for coordinate conversion.
* **Reporting:** FPDF for dynamic PDF generation.

## Installation

1.  **Clone the repo:**
    ```bash
    git clone [https://github.com/Muhammad-Mujtaba-Git/Lahore-Air-Guard.git](https://github.com/Muhammad-Mujtaba-Git/Lahore-Air-Guard.git)
    cd Lahore-Air-Guard
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the app:**
    ```bash
    streamlit run app.py
    ```

## Screenshots

| Dashboard View | 24-Hour Forecast |
|:---:|:---:|
| *Real-time metrics with AI Validation check.* | *Interactive trend graph showing history vs. prediction.* |

## Contribution
This project is open-source. Feel free to fork the repository and tweak the ARIMA parameters (`order=(8,0,1)`) to optimize the model for your own city's weather patterns!

---
*Developed by **Muhammad Mujtaba** | [LinkedIn Profile](https://www.linkedin.com/in/muhammad-mujtaba-ml/)*
