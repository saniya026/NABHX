<div align="center">

<img src="./assets/logo.png" alt="NABHX Logo" width="220"/>

# 🌍 NABHX

### AI-Powered Environmental Intelligence Platform

**Bharatiya Antariksh Hackathon (BAH) 2026 – Problem Statement 3**

*Transforming Satellite Data into Actionable Environmental Intelligence.*

</div>

---

# 📖 About NABHX

Air pollution is one of India's most pressing environmental and public health challenges. Traditional monitoring systems rely on ground-based monitoring stations, which provide limited spatial coverage and are unable to represent pollution conditions in station-free regions.

**NABHX** is an AI-powered environmental intelligence platform that integrates **Satellite Remote Sensing, Machine Learning, Meteorological Analysis, and Geospatial Intelligence** to monitor, analyze, and forecast air quality across India.

The project was developed for **Bharatiya Antariksh Hackathon (BAH) 2026 – Problem Statement 3**.

---

# 🎯 Objectives

## 🌫 Objective 1 – Nationwide AQI Prediction

Develop a station-independent machine learning model capable of predicting Air Quality Index (AQI) using meteorological parameters and fire activity.

### Features

- Station-independent AQI Prediction
- 80:20 Train-Test Split
- Random Forest Regression
- XGBoost Regression
- LSTM Model Evaluation
- Daily AQI Forecasting
- AQI Heatmap Generation
- Feature Importance Analysis
- Model Performance Evaluation (R², RMSE & MAE)

---

## 🛰 Objective 2 – HCHO Hotspot Detection

Analyze atmospheric Formaldehyde (HCHO) concentration over India using Sentinel-5P satellite observations.

### Features

- Monthly HCHO Analysis
- Annual HCHO Analysis
- HCHO Hotspot Detection
- DBSCAN Clustering
- Fire–HCHO Correlation Analysis
- Wind Transport Analysis
- India Source Region Index (ISRI)
- Interactive Spatial Visualization

---

# 💡 Innovation

## 🔔 AI-Based Early Warning System

NABHX predicts next-day AQI and provides proactive alerts before hazardous pollution levels occur.

### Includes

- 24-Hour AQI Forecast
- Firebase Push Notifications
- High AQI Alerts
- Forecast-based Decision Support

---

# 🛰 Data Sources

| Dataset | Source |
|----------|--------|
| AQI Data | CPCB |
| HCHO Satellite Data | Sentinel-5P / TROPOMI |
| Meteorological Data | NASA POWER |
| Fire Count Data | NASA FIRMS (VIIRS) |
| Wind Data | ERA5 Reanalysis |

---

# 🤖 Machine Learning Models

The AQI prediction framework was evaluated using multiple machine learning approaches.

### Models

- Random Forest
- XGBoost ⭐ (Best Performing)
- LSTM Neural Network

### Data Split

- Training Data – 80%
- Testing Data – 20%

### Evaluation Metrics

- R² Score
- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)

---

# 🛠 Technology Stack

### Programming

- Python

### Machine Learning

- Scikit-learn
- XGBoost
- TensorFlow / Keras

### Geospatial Processing

- Rasterio
- GeoPandas
- Shapely
- Folium

### Data Analysis

- Pandas
- NumPy

### Visualization

- Matplotlib
- Plotly

### Notifications

- Firebase Cloud Messaging (FCM)

---

# 📂 Repository Structure

```text
NABHX
│
├── Objective1_AQI
│   ├── data/
│   ├── models/
│   ├── outputs/
│   ├── forecast_24h.py
│   ├── objective1_final.py
│   └── ...
│
├── Objective2_HCHO
│   ├── data/
│   ├── output/
│   ├── hcho_hotspot_dbscan.py
│   ├── india_source_region_index.py
│   ├── wind_transport.py
│   └── ...
│
├── assets/
│   └── logo.png
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# 📊 Outputs

The platform generates:

- AQI Prediction Maps
- AQI Heatmaps
- AQI Forecast Maps
- HCHO Hotspot Maps
- DBSCAN Cluster Maps
- Wind Transport Maps
- India Source Region Index (ISRI)
- Interactive Dashboards
- Early Warning Notifications

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/NABHX.git
```

Move into the project:

```bash
cd NABHX
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run

### Objective 1

```bash
python Objective1_AQI/objective1_final.py
```

### AQI Forecast

```bash
python Objective1_AQI/forecast_24h.py
```

### Objective 2

```bash
python Objective2_HCHO/hcho_hotspot_dbscan.py
```

---

# 🌍 Applications

- National Air Quality Monitoring
- Pollution Forecasting
- Environmental Research
- Biomass Burning Assessment
- Public Health Advisory
- Smart City Planning
- Atmospheric Science
- Decision Support for Environmental Agencies

---

# 🔮 Future Scope

- Near Real-Time Satellite Integration
- Multi-Pollutant Prediction
- Mobile Application
- Cloud Deployment
- AI-Based Health Advisory
- National Environmental Intelligence Dashboard

---

# 👥 Team

**Project Name:** NABHX

Developed for **Bharatiya Antariksh Hackathon (BAH) 2026**

---

<div align="center">

⭐ If you find this project useful, consider giving it a star.

</div>