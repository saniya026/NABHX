# 🌍 NABHX – HCHO Hotspot Detection & Source Region Analysis

> **Bharatiya Antariksh Hackathon (BAH) 2026**  
> **Problem Statement 3 – Objective 2**

NABHX is an AI-powered geospatial analysis platform that detects Formaldehyde (HCHO) hotspots across India using satellite observations, fire activity, and wind reanalysis data. The project identifies pollution source regions, analyzes Fire–HCHO relationships, and studies atmospheric pollutant transport to support air quality monitoring and environmental decision-making.

---

# 🚀 Project Overview

NABHX integrates multiple geospatial datasets to identify HCHO hotspot regions across India. The project combines machine learning, satellite imagery, fire activity, and meteorological data to generate scientifically meaningful insights.

The complete pipeline includes:

- HCHO Hotspot Detection using DBSCAN
- Monthly & Annual HCHO Trend Analysis
- Fire–HCHO Correlation Analysis
- India Source Region Index (ISRI)
- ERA5 Wind Transport Analysis
- Interactive Dashboard

---

# 🎯 Objectives

- Detect HCHO hotspot regions across India.
- Identify major HCHO source regions.
- Analyze the relationship between fire activity and HCHO concentration.
- Assess atmospheric transport using ERA5 wind data.
- Visualize results through interactive dashboards.

---

# ✨ Key Features

- ✅ DBSCAN-based HCHO Hotspot Detection
- ✅ Annual Mean HCHO Heatmap
- ✅ Monthly Trend Analysis
- ✅ India Source Region Index (ISRI)
- ✅ Fire–HCHO Correlation Analysis
- ✅ ERA5 Wind Transport Visualization
- ✅ Interactive Plotly Dashboard
- ✅ High-Resolution Maps & Charts

---

# 🛰️ Datasets Used

The raw datasets are not included in this repository due to their large file size.

| Dataset | Source |
|----------|--------|
| HCHO Satellite Data | Sentinel-5P TROPOMI (Google Earth Engine) |
| Fire Data | VIIRS SNPP – NASA FIRMS |
| Wind Data | ERA5 Reanalysis (ECMWF Copernicus) |
| Ground Validation | CPCB Air Quality Data |

**Study Period:** 2023–2024

---

# 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Rasterio
- GeoPandas
- Matplotlib
- Plotly
- Folium

---

# 📂 Project Structure

```text
NABHX/
│
├── data/
├── output/
│   ├── images/
│   ├── hcho_dashboard.png
│   ├── hcho_hotspot_dashboard.html
│   └── *.csv
│
├── hcho_hotspot_dbscan.py
├── india_source_region_index.py
├── wind_transport.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 📊 Outputs

- HCHO Annual Mean Heatmap
- DBSCAN Hotspot Clusters
- Monthly HCHO Trend Analysis
- Top HCHO Cluster Comparison
- India Source Region Index (ISRI)
- Wind Transport Analysis
- Interactive HTML Dashboard

---

# ▶️ Installation

Clone the repository

```bash
git clone https://github.com/your-username/NABHX.git
```

Move to the project directory

```bash
cd NABHX
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Project

```bash
python hcho_hotspot_dbscan.py
```

```bash
python india_source_region_index.py
```

```bash
python wind_transport.py
```

---

# 💡 Innovation

NABHX introduces the **India Source Region Index (ISRI)**, a composite metric that ranks Indian states using:

- HCHO Concentration
- Fire Count
- Post-Monsoon Fire Activity
- Fire–HCHO Correlation

The project also integrates **ERA5 Wind Transport Analysis** to trace pollutant movement from biomass-burning regions to downstream HCHO hotspots.

---

# 🌍 Applications

- Air Quality Monitoring
- Pollution Source Identification
- Environmental Research
- Atmospheric Analysis
- Policy Support
- Smart Environmental Decision-Making

---

# 📌 Future Scope

- Near Real-Time Monitoring
- AQI Forecast Integration
- Early Warning & Alert System
- Web Dashboard Deployment
- Mobile Application Support

---

# 👥 Team

**Team Name:** NABHX

Developed for **Bharatiya Antariksh Hackathon (BAH) 2026**

---

# 📄 License

This project is developed for educational and research purposes as part of the **Bharatiya Antariksh Hackathon (BAH) 2026**.