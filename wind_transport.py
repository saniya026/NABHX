import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import rasterio
import pandas as pd
import os
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

os.makedirs('output/images', exist_ok=True)

print("Loading ERA5 wind...")
ds = xr.open_dataset('data/era5_wind_india.nc')
u10 = ds['u10'].mean(dim='valid_time').values
v10 = ds['v10'].mean(dim='valid_time').values
lons_w = ds['longitude'].values
lats_w = ds['latitude'].values

print("Loading HCHO...")
results = {}
for year, path in [
    (2023, 'data/HCHO_2023_monthly.tif'),
    (2024, 'data/HCHO_2024_monthly.tif')
]:
    with rasterio.open(path) as src:
        stack = src.read().astype(float)
        t = src.transform
        bounds = src.bounds
    stack[stack < 0] = float('nan')
    stack[stack > 5e-3] = float('nan')
    results[year] = {
        'hcho': np.nanmean(stack[[9, 10]], axis=0) * 1e6,
        'bounds': bounds
    }

print("Loading fire data...")
df = pd.read_csv('data/india_fires_raw_2023_2024.csv')
df['acq_date'] = pd.to_datetime(df['acq_date'])
df['month'] = df['acq_date'].dt.month
df['year']  = df['acq_date'].dt.year
df_pm = df[
    (df['month'].isin([10, 11])) &
    (df['confidence'].isin(['h', 'n']))
]

fig = plt.figure(figsize=(22, 12))
fig.patch.set_facecolor('#f8f8f8')

fig.text(0.5, 0.98,
         'HCHO Transport Analysis — ERA5 Wind Vectors + Fire Sources',
         ha='center', va='top', fontsize=18, fontweight='bold', color='#1a1a2e')
fig.text(0.5, 0.94,
         'Post-Monsoon Season (Oct-Nov)  |  10m ERA5 Wind Field  |  VIIRS Fire Detections (FRP > 50 MW)',
         ha='center', va='top', fontsize=12, color='#555555')

for ax_idx, year in enumerate([2023, 2024]):
    ax = fig.add_subplot(1, 2, ax_idx + 1)
    b = results[year]['bounds']
    hcho = results[year]['hcho']
    hcho_s = gaussian_filter(np.nan_to_num(hcho, nan=0.), sigma=1.0)

    extent = [b.left, b.right, b.bottom, b.top]
    vmin = np.nanpercentile(hcho_s[hcho_s > 0], 10)
    vmax = np.nanpercentile(hcho_s[hcho_s > 0], 98)
    im = ax.imshow(hcho_s, extent=extent, origin='upper',
                   cmap='YlOrRd', alpha=0.88, vmin=vmin, vmax=vmax)

    step = 3
    LON, LAT = np.meshgrid(lons_w[::step], lats_w[::step])
    U = u10[::step, ::step]
    V = v10[::step, ::step]
    speed = np.sqrt(U**2 + V**2)
    ax.quiver(LON, LAT, U, V, speed,
              cmap='winter', scale=90, width=0.0025,
              alpha=0.80, zorder=4)

    fire_yr  = df_pm[df_pm['year'] == year]
    fire_int = fire_yr[fire_yr['frp'] > 50]
    n_fire   = len(fire_int)
    fire_s   = fire_int.sample(min(1000, n_fire), random_state=42)
    ax.scatter(fire_s['longitude'], fire_s['latitude'],
               c='#ff1a1a', s=fire_s['frp'] / 25,
               alpha=0.65, zorder=6,
               edgecolors='darkred', linewidths=0.3)

    regions = [
        ('IGP Belt',      80.0, 27.5),
        ('NE India',      92.5, 25.0),
        ('W. Ghats',      74.5, 13.0),
        ('Central India', 80.5, 21.0),
    ]
    for name, lx, ly in regions:
        ax.text(lx, ly, name,
                fontsize=9, fontweight='bold', color='#003366',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='white', edgecolor='#003366',
                          alpha=0.85, linewidth=1.2))

    cbar = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label('HCHO Concentration (umol/m2)', fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    title_line1 = str(year) + '  -  Oct-Nov Post-Monsoon'
    title_line2 = str(len(fire_yr)) + ' fire detections  |  ' + str(n_fire) + ' intense (FRP > 50 MW)'
    ax.set_title(title_line1 + '\n' + title_line2,
                 fontsize=12, fontweight='bold', pad=14, color='#1a1a2e')

    ax.set_xlabel('Longitude (E)', fontsize=11, labelpad=8)
    ax.set_ylabel('Latitude (N)', fontsize=11, labelpad=8)
    ax.set_xlim(b.left, b.right)
    ax.set_ylim(b.bottom, b.top)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.25, linestyle='--', color='gray')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='#ff1a1a', markersize=9,
               markeredgecolor='darkred', label='Fire (FRP > 50 MW)'),
        Patch(facecolor='#4682B4', alpha=0.6,
              label='Wind vector (10m ERA5)'),
    ]
    ax.legend(handles=legend_elements, fontsize=9,
              loc='lower left', framealpha=0.92,
              edgecolor='#aaaaaa')

plt.subplots_adjust(top=0.88, bottom=0.08, left=0.06,
                    right=0.96, wspace=0.20)
plt.savefig('output/images/wind_transport_map.png',
            dpi=150, bbox_inches='tight', facecolor='#f8f8f8')
plt.close()
print('Saved: output/images/wind_transport_map.png')
