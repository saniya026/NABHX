

import os
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from scipy.ndimage import gaussian_filter
from sklearn.cluster import DBSCAN
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

PATHS = {
    2023: "data/HCHO_2023_monthly.tif",   # 12-band GeoTIFF (Jan–Dec 2023)
    2024: "data/HCHO_2024_monthly.tif",   # 12-band GeoTIFF (Jan–Dec 2024)
}

OUTPUT_DIR        = "output"
PERCENTILE_THRESH = 80      # cluster only pixels above this percentile
EPS_DEG           = 0.3    # DBSCAN neighbourhood radius in degrees (~33 km)
MIN_SAMPLES       = 8       # minimum pixels to form a cluster
SMOOTHING_SIGMA   = 1.2    # gaussian blur to reduce speckle (0 = disable)

MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']

COLOR_MAP = {
    "Persistent anthropogenic":             "#e74c3c",
    "Anthropogenic (industrial/vehicular)": "#c0392b",
    "Agricultural burning":                 "#e67e22",
    "Biogenic (forest/vegetation)":         "#27ae60",
    "Episodic/mixed":                       "#8e44ad",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)




def load_tiff(path):
    """
    Load a 12-band GeoTIFF into a numpy stack (12, H, W).
    Band 1 = January, Band 12 = December.
    """
    with rasterio.open(path) as src:
        stack = src.read().astype(np.float64)   # (12, H, W)
        transform = src.transform
        crs = src.crs
        print(f"  Bands      : {src.count}")
        print(f"  Size       : {src.width} x {src.height} pixels")
        print(f"  CRS        : {src.crs}")
        print(f"  Bounds     : {src.bounds}")
    return stack, transform, crs



# STEP 2: MASK INVALID VALUES

def mask_stack(stack):
    """
    Sentinel-5P TROPOMI HCHO:
      - Negative values = retrieval noise/artifacts → set NaN
      - Values > 5e-3 mol/m² = physically unrealistic → set NaN
    """
    stack = stack.copy()
    stack[stack < 0]    = np.nan
    stack[stack > 5e-3] = np.nan
    return stack


# ─────────────────────────────────────────────────────────────
# STEP 3: COMPUTE FEATURES PER PIXEL
# ─────────────────────────────────────────────────────────────
def compute_features(stack):
    """
    Returns per-pixel feature arrays (H, W each):
      annual_mean  : mean HCHO over 12 months
      cv           : coefficient of variation (std/mean) — high = episodic source
      seasonality  : (summer_mean - winter_mean) / annual_mean
                     > 0 = biogenic (summer peak)
                     < 0 = anthropogenic/burning (winter/post-monsoon peak)
      pm_ratio     : post-monsoon (Oct-Nov) mean / annual mean
                     > 1.4 = likely crop burning signal
    """
    annual_mean = np.nanmean(stack, axis=0)
    annual_std  = np.nanstd(stack, axis=0)
    cv          = annual_std / (annual_mean + 1e-12)

    # Apr(3) May(4) Jun(5) Jul(6) Aug(7) = summer
    summer = np.nanmean(stack[[3, 4, 5, 6, 7]], axis=0)
    # Nov(10) Dec(11) Jan(0) Feb(1) = winter
    winter = np.nanmean(stack[[10, 11, 0, 1]], axis=0)
    seasonality = (summer - winter) / (annual_mean + 1e-12)

    # Oct(9) Nov(10) = post-monsoon crop burning window
    post_monsoon = np.nanmean(stack[[9, 10]], axis=0)
    pm_ratio     = post_monsoon / (annual_mean + 1e-12)

    return annual_mean, cv, seasonality, pm_ratio


# ─────────────────────────────────────────────────────────────
# STEP 4: RASTER → GEODATAFRAME
# ─────────────────────────────────────────────────────────────
def raster_to_gdf(annual_mean, cv, seasonality, pm_ratio, transform, sigma):
    """
    Apply optional smoothing, then convert valid raster pixels
    to a GeoDataFrame of points with lat/lon coordinates.
    """
    if sigma > 0:
        smoothed = gaussian_filter(np.nan_to_num(annual_mean, nan=0.0), sigma=sigma)
        smoothed[np.isnan(annual_mean)] = np.nan
    else:
        smoothed = annual_mean

    ri, ci = np.where(~np.isnan(smoothed))
    lons = transform.c + ci * transform.a + transform.a / 2
    lats = transform.f + ri * transform.e + transform.e / 2  # transform.e is negative

    gdf = gpd.GeoDataFrame({
        'hcho':        smoothed[ri, ci],
        'cv':          cv[ri, ci],
        'seasonality': seasonality[ri, ci],
        'pm_ratio':    pm_ratio[ri, ci],
        'lat':         lats,
        'lon':         lons,
        'row':         ri,
        'col':         ci,
    }, geometry=gpd.points_from_xy(lons, lats), crs='EPSG:4326')

    return gdf


# ─────────────────────────────────────────────────────────────
# STEP 5: FILTER TOP PERCENTILE
# ─────────────────────────────────────────────────────────────
def filter_hotpixels(gdf, percentile):
    threshold = np.nanpercentile(gdf['hcho'], percentile)
    hot = gdf[gdf['hcho'] >= threshold].copy()
    print(f"  Threshold (p{percentile}): {threshold:.4e} mol/m²")
    print(f"  Pixels above threshold : {len(hot):,}")
    return hot, threshold


# ─────────────────────────────────────────────────────────────
# STEP 6: DBSCAN
# ─────────────────────────────────────────────────────────────
def run_dbscan(hot, eps_deg, min_samples):
    """
    Cluster hot pixels using DBSCAN with haversine distance.
    Coords must be in radians for haversine metric.
    eps is converted from degrees to radians.
    Cluster label = -1 means noise (not part of any cluster).
    """
    coords_rad = np.radians(hot[['lat', 'lon']].values)
    db = DBSCAN(
        eps=np.radians(eps_deg),
        min_samples=min_samples,
        algorithm='ball_tree',
        metric='haversine',
        n_jobs=-1
    )
    labels = db.fit_predict(coords_rad)
    hot = hot.copy()
    hot['cluster'] = labels
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    print(f"  Clusters found : {n_clusters}")
    print(f"  Noise pixels   : {n_noise:,}")
    return hot, n_clusters


# ─────────────────────────────────────────────────────────────
# STEP 7: SOURCE ATTRIBUTION
# ─────────────────────────────────────────────────────────────
def attribute_source(season_idx, pm_r, cv_val):
    """
    Rule-based source attribution per cluster.

    Logic:
      pm_ratio > 1.4 + high CV  → post-monsoon spike = crop stubble burning
      seasonality > 0.3          → summer peak = forest/vegetation VOC (biogenic)
      seasonality < -0.15        → winter/year-round = industrial/vehicular
      high CV only               → irregular emissions = episodic/mixed
      else                       → persistent background anthropogenic
    """
    if pm_r > 1.4 and cv_val > 0.25:
        return "Agricultural burning"
    elif season_idx > 0.3:
        return "Biogenic (forest/vegetation)"
    elif season_idx < -0.15:
        return "Anthropogenic (industrial/vehicular)"
    elif cv_val > 0.35:
        return "Episodic/mixed"
    else:
        return "Persistent anthropogenic"


def build_cluster_summary(hot, year):
    """
    Aggregate per-cluster statistics and assign source type + intensity rank.
    """
    clustered = hot[hot['cluster'] >= 0]

    summary = clustered.groupby('cluster').agg(
        centroid_lat  = ('lat',         'mean'),
        centroid_lon  = ('lon',         'mean'),
        mean_hcho     = ('hcho',        'mean'),
        max_hcho      = ('hcho',        'max'),
        hcho_std      = ('hcho',        'std'),
        pixel_count   = ('hcho',        'count'),
        mean_cv       = ('cv',          'mean'),
        mean_season   = ('seasonality', 'mean'),
        mean_pm       = ('pm_ratio',    'mean'),
    ).reset_index()

    summary['source_type'] = summary.apply(
        lambda r: attribute_source(r['mean_season'], r['mean_pm'], r['mean_cv']),
        axis=1
    )

    p75 = summary['mean_hcho'].quantile(0.75)
    p90 = summary['mean_hcho'].quantile(0.90)
    summary['intensity'] = summary['mean_hcho'].apply(
        lambda v: 'Extreme' if v >= p90 else ('High' if v >= p75 else 'Moderate')
    )

    summary = summary.sort_values('mean_hcho', ascending=False).reset_index(drop=True)
    summary['rank'] = summary.index + 1
    summary['year'] = year

    return summary


# ─────────────────────────────────────────────────────────────
# STEP 8: MONTHLY TREND PER CLUSTER
# ─────────────────────────────────────────────────────────────
def compute_monthly_trend(stack, hot, summary, top_n=5):
    """
    For each of the top N clusters, compute mean HCHO per month.
    Returns a DataFrame with columns: cluster, month, month_name, mean_hcho
    """
    clustered = hot[hot['cluster'] >= 0]
    top_ids   = summary.head(top_n)['cluster'].tolist()
    records   = []

    for cid in top_ids:
        pixels = clustered[clustered['cluster'] == cid]
        for m in range(12):
            vals = stack[m][pixels['row'].values, pixels['col'].values]
            records.append({
                'cluster':    str(cid),
                'month':      m + 1,
                'month_name': MONTH_NAMES[m],
                'mean_hcho':  np.nanmean(vals),
            })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────
# STEP 9: BUILD DASHBOARD
# ─────────────────────────────────────────────────────────────
def build_dashboard(all_data):
    """
    6-panel interactive Plotly dashboard:
      Row 1: HCHO annual mean heatmap (2023 | 2024)
      Row 2: DBSCAN cluster map with ranked centroid bubbles (2023 | 2024)
      Row 3: Monthly trend line chart | Top-5 cluster bar chart
    """
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "HCHO Annual Mean — 2023",        "HCHO Annual Mean — 2024",
            "DBSCAN Hotspot Clusters — 2023", "DBSCAN Hotspot Clusters — 2024",
            "Monthly HCHO Trend (India-wide)", "Top 5 Clusters by Mean HCHO",
        ],
        specs=[
            [{"type": "scattermap"}, {"type": "scattermap"}],
            [{"type": "scattermap"}, {"type": "scattermap"}],
            [{"type": "scatter"},    {"type": "bar"}],
        ],
        row_heights=[0.35, 0.35, 0.30],
        vertical_spacing=0.07,
        horizontal_spacing=0.05,
    )

    MAP_KEYS = ['map', 'map2', 'map3', 'map4']
    center   = dict(lat=20.5, lon=82.0)

    for ci, year in enumerate([2023, 2024]):
        d       = all_data[year]
        gdf     = d['gdf']
        hot     = d['hot']
        summary = d['summary']

        # ── Row 1: full heatmap (sampled for render speed) ──────────────
        sample = gdf.sample(min(25000, len(gdf)), random_state=42)
        fig.add_trace(go.Scattermap(
            lat=sample['lat'], lon=sample['lon'],
            mode='markers',
            marker=dict(
                size=3, opacity=0.55,
                color=sample['hcho'],
                colorscale='YlOrRd',
                showscale=(ci == 0),
                colorbar=dict(title="mol/m²", x=-0.02, len=0.30, y=0.83) if ci == 0 else {},
            ),
            name=f"HCHO {year}",
            hovertemplate="Lat: %{lat:.2f}<br>Lon: %{lon:.2f}<br>HCHO: %{marker.color:.3e} mol/m²<extra></extra>",
            subplot=MAP_KEYS[ci],
        ), row=1, col=ci + 1)

        # ── Row 2: cluster pixels ────────────────────────────────────────
        hot_s = hot.sample(min(18000, len(hot)), random_state=42)
        fig.add_trace(go.Scattermap(
            lat=hot_s['lat'], lon=hot_s['lon'],
            mode='markers',
            marker=dict(
                size=3, opacity=0.6,
                color=hot_s['cluster'].clip(lower=0),
                colorscale='Turbo', showscale=False,
            ),
            text=hot_s['cluster'].apply(lambda c: f"Cluster {c}" if c >= 0 else "Noise"),
            hovertemplate="%{text}<br>Lat: %{lat:.2f}<br>Lon: %{lon:.2f}<extra></extra>",
            name=f"Pixels {year}",
            subplot=MAP_KEYS[ci + 2],
        ), row=2, col=ci + 1)

        # ── Row 2: centroid bubbles coloured by source type ──────────────
        fig.add_trace(go.Scattermap(
            lat=summary['centroid_lat'],
            lon=summary['centroid_lon'],
            mode='markers+text',
            marker=dict(
                size=summary['pixel_count'].apply(lambda x: min(max(x ** 0.38, 12), 38)),
                color=[COLOR_MAP.get(st, '#888') for st in summary['source_type']],
                opacity=0.92,
            ),
            text=summary['rank'].astype(str),
            textfont=dict(size=9, color='white'),
            customdata=np.column_stack([
                summary['rank'],
                summary['mean_hcho'].map('{:.3e}'.format),
                summary['pixel_count'],
                summary['source_type'],
                summary['intensity'],
            ]),
            hovertemplate=(
                "<b>Rank #%{customdata[0]}</b><br>"
                "Mean HCHO : %{customdata[1]} mol/m²<br>"
                "Pixels    : %{customdata[2]}<br>"
                "Source    : %{customdata[3]}<br>"
                "Intensity : %{customdata[4]}<extra></extra>"
            ),
            name=f"Centroids {year}",
            subplot=MAP_KEYS[ci + 2],
        ), row=2, col=ci + 1)

    # ── Row 3 left: monthly trend ────────────────────────────────────────
    for year, dash in [(2023, 'solid'), (2024, 'dash')]:
        fig.add_trace(go.Scatter(
            x=MONTH_NAMES,
            y=all_data[year]['monthly_means'],
            mode='lines+markers',
            name=str(year),
            line=dict(dash=dash, width=2.5),
            marker=dict(size=7),
        ), row=3, col=1)

    # ── Row 3 right: top-5 cluster bar chart ─────────────────────────────
    bar_colors = {2023: '#3498db', 2024: '#e74c3c'}
    for year in [2023, 2024]:
        s = all_data[year]['summary'].head(5)
        fig.add_trace(go.Bar(
            x=[f"#{r}" for r in s['rank']],
            y=s['mean_hcho'],
            name=str(year),
            marker_color=bar_colors[year],
            width=0.35,
            offset=-0.2 if year == 2023 else 0.2,
            error_y=dict(type='data', array=s['hcho_std'].fillna(0), visible=True),
            customdata=s[['source_type', 'pixel_count']].values,
            hovertemplate=(
                f"Year {year}<br>"
                "HCHO: %{y:.3e} mol/m²<br>"
                "%{customdata[0]}<br>"
                "Pixels: %{customdata[1]}<extra></extra>"
            ),
        ), row=3, col=2)

    # ── Layout ───────────────────────────────────────────────────────────
    map_cfg = dict(style='carto-positron', center=center, zoom=3.0)
    fig.update_layout(
        title=dict(
            text=(
                "<b>HCHO Hotspot Detection — India (2023 vs 2024)</b><br>"
                "<sup>Sentinel-5P TROPOMI · DBSCAN eps=0.3° · min_samples=8 · 12-band monthly GeoTIFF</sup>"
            ),
            x=0.5, font=dict(size=17),
        ),
        height=1400,
        map=map_cfg, map2=map_cfg, map3=map_cfg, map4=map_cfg,
        showlegend=True,
        legend=dict(orientation='h', x=0, y=-0.02, font=dict(size=10)),
        paper_bgcolor='#f9f9f9',
        plot_bgcolor='white',
        font=dict(family='Arial', size=11),
        barmode='group',
    )
    fig.update_xaxes(title_text="Month", row=3, col=1)
    fig.update_yaxes(title_text="Mean HCHO (mol/m²)", row=3, col=1)
    fig.update_xaxes(title_text="Cluster Rank", row=3, col=2)
    fig.update_yaxes(title_text="Mean HCHO (mol/m²)", row=3, col=2)

    return fig


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    all_data = {}

    for year, path in PATHS.items():
        print(f"\n{'=' * 55}")
        print(f"  Year: {year}  |  File: {path}")
        print(f"{'=' * 55}")

        # 1. Load
        stack, transform, crs = load_tiff(path)

        # 2. Mask
        stack = mask_stack(stack)

        # 3. Features
        annual_mean, cv, seasonality, pm_ratio = compute_features(stack)

        # 4. Raster → GDF
        gdf = raster_to_gdf(annual_mean, cv, seasonality, pm_ratio, transform, SMOOTHING_SIGMA)
        print(f"  Valid pixels   : {len(gdf):,}")

        # 5. Filter
        hot, threshold = filter_hotpixels(gdf, PERCENTILE_THRESH)

        # 6. DBSCAN
        hot, n_clusters = run_dbscan(hot, EPS_DEG, MIN_SAMPLES)

        # 7. Summary + source attribution
        summary = build_cluster_summary(hot, year)

        print(f"\n  Top clusters:")
        print(summary[['rank', 'centroid_lat', 'centroid_lon', 'mean_hcho',
                        'pixel_count', 'intensity', 'source_type']].to_string(index=False))

        # 8. Monthly trend
        trend_df      = compute_monthly_trend(stack, hot, summary, top_n=5)
        monthly_means = [float(np.nanmean(stack[m])) for m in range(12)]

        # Save CSVs
        summary.to_csv(os.path.join(OUTPUT_DIR, f"hcho_clusters_{year}.csv"), index=False)
        trend_df.to_csv(os.path.join(OUTPUT_DIR, f"hcho_trend_{year}.csv"),    index=False)
        hot.drop(columns='geometry').to_csv(
            os.path.join(OUTPUT_DIR, f"hcho_pixels_{year}.csv"), index=False)

        all_data[year] = {
            'gdf': gdf, 'hot': hot, 'summary': summary,
            'monthly_means': monthly_means, 'trend_df': trend_df,
        }

    # Combined cluster CSV
    combined = pd.concat([all_data[y]['summary'] for y in PATHS], ignore_index=True)
    combined.to_csv(os.path.join(OUTPUT_DIR, "hcho_clusters_combined.csv"), index=False)

    # 9. Dashboard
    print(f"\n  Building dashboard...")
    fig = build_dashboard(all_data)
    out = os.path.join(OUTPUT_DIR, "hcho_hotspot_dashboard.html")
    fig.write_html(out, include_plotlyjs='cdn')
    print(f"  Dashboard saved → {out}")

    print(f"\n{'=' * 55}")
    print(f"  DONE — outputs in: {os.path.abspath(OUTPUT_DIR)}/")
    print(f"{'=' * 55}")
    print(f"  hcho_hotspot_dashboard.html  ← open in browser")
    print(f"  hcho_clusters_2023.csv")
    print(f"  hcho_clusters_2024.csv")
    print(f"  hcho_clusters_combined.csv")
    print(f"  hcho_trend_2023.csv")
    print(f"  hcho_trend_2024.csv")


if __name__ == "__main__":
    main()