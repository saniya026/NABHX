# 🌍 NABHX

## AI-Powered Environmental Intelligence Platform for ISRO Hackathon

NABHX is an AI-based environmental monitoring platform developed for the ISRO Hackathon. It integrates satellite observations, machine learning, geospatial analytics, and early warning systems to monitor India's air quality and formaldehyde (HCHO) pollution.

---

# 📌 Objectives

## 🎯 Objective 1 – Nationwide AQI Prediction

Develop an AI model to predict Air Quality Index (AQI) across India using weather and fire-related parameters.

### Features

- AQI Prediction using Machine Learning
- India AQI Heatmap
- Annual AQI Mapping
- Model Comparison
- Feature Importance Analysis
- Daily Forecast Visualization
- K-Fold Cross Validation
- XGBoost & Random Forest Models

---

## 🎯 Objective 2 – HCHO Hotspot Detection

Identify and analyze HCHO hotspots over India using Sentinel-5P satellite observations.

### Features

- HCHO Hotspot Detection
- Monthly & Annual HCHO Analysis
- DBSCAN Clustering
- Wind Transport Analysis
- India Source Region Index (ISRI)
- Interactive Dashboard
- Spatial Trend Analysis

---

# 💡 Innovation

## Early Warning System

NABHX includes an AI-powered Early Warning System that predicts next-day AQI and sends push notifications to users through Firebase Cloud Messaging (FCM).

Features include:

- 24-Hour AQI Forecast
- High AQI Alerts
- Firebase Push Notifications
- City-wise Forecasts
- Interactive Forecast Heatmaps

---

# 🛠️ Technologies Used

- Python
- Machine Learning
- XGBoost
- Random Forest
- Scikit-learn
- Pandas
- NumPy
- GeoPandas
- Rasterio
- Folium
- Matplotlib
- Plotly
- Firebase Cloud Messaging (FCM)

---

# 📂 Project Structure

```
NABHX/
│
├── Objective1_AQI/
│   ├── models/
│   ├── outputs/
│   ├── shapefiles/
│   ├── data/
│   └── Python scripts
│
├── Objective2_HCHO/
│   ├── data/
│   ├── output/
│   ├── images/
│   └── Python scripts
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# 📊 Datasets

The project uses publicly available datasets from:

- Sentinel-5P/TROPOMI Satellite Data
- NASA POWER Meteorological Data
- FIRMS Fire Data
- AQI Monitoring Data
- Administrative Shapefiles of India

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/NABHX.git
```

Move into the project directory:

```bash
cd NABHX
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Project

Run Objective 1:

```bash
python Objective1_AQI/objective1_final.py
```

Run Objective 2:

```bash
python Objective2_HCHO/hcho_hotspot_dbscan.py
```

---

# 📈 Outputs

The project generates:

- AQI Prediction Maps
- AQI Heatmaps
- HCHO Hotspot Maps
- DBSCAN Cluster Maps
- Wind Transport Maps
- Forecast Visualizations
- Feature Importance Graphs
- Interactive HTML Dashboards

---

# 👥 Team

Developed for the **ISRO Hackathon**.

Project Name: **NABHX**

---

# 📄 License

This project is developed for educational and research purposes under the ISRO Hackathon.