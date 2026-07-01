import pandas as pd
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import pearsonr
import os

os.makedirs('output/images', exist_ok=True)

REGION_MAP = {
    'Amritsar':'Punjab','Ludhiana':'Punjab','Jalandhar':'Punjab','Patiala':'Punjab',
    'Gurugram':'Haryana','Delhi':'Delhi NCR','Noida':'Delhi NCR',
    'Lucknow':'Uttar Pradesh','Kanpur':'Uttar Pradesh','Varanasi':'Uttar Pradesh',
    'Gorakhpur':'Uttar Pradesh','Prayagraj':'Uttar Pradesh','Agra':'Uttar Pradesh',
    'Meerut':'Uttar Pradesh','Bareilly':'Uttar Pradesh','Firozabad':'Uttar Pradesh',
    'Patna':'Bihar','Bhopal':'Madhya Pradesh','Jhansi':'Madhya Pradesh',
    'Raipur':'Chhattisgarh','Brajrajnagar':'Chhattisgarh',
    'Jorapokhar':'Jharkhand','Pathardih':'Jharkhand','Durgapur':'Jharkhand',
    'Bhubaneswar':'Odisha','Talcher':'Odisha',
    'Kolkata':'West Bengal','Howrah':'West Bengal','Siliguri':'West Bengal',
    'Guwahati':'Assam','Shillong':'Meghalaya','Aizawl':'Mizoram',
    'Imphal':'Manipur','Gangtok':'Sikkim',
    'Hyderabad':'Telangana','Visakhapatnam':'Andhra Pradesh','Amaravati':'Andhra Pradesh',
    'Chennai':'Tamil Nadu','Coimbatore':'Tamil Nadu','Ooty':'Tamil Nadu',
    'Bengaluru':'Karnataka','Kochi':'Kerala','Thiruvananthapuram':'Kerala',
    'Mumbai':'Maharashtra','Pune':'Maharashtra','Nashik':'Maharashtra',
    'Nagpur':'Maharashtra','Aurangabad':'Maharashtra',
    'Ahmedabad':'Gujarat','Gandhinagar':'Gujarat',
    'Jaipur':'Rajasthan','Dehradun':'Uttarakhand','Rishikesh':'Uttarakhand',
}

CITY_COORDS = {
    'Agra':(27.18,78.01),'Ahmedabad':(23.03,72.57),'Aizawl':(23.73,92.72),
    'Amaravati':(16.51,80.51),'Amritsar':(31.63,74.87),'Aurangabad':(19.88,75.34),
    'Bareilly':(28.36,79.41),'Bengaluru':(12.97,77.59),'Bhopal':(23.26,77.41),
    'Bhubaneswar':(20.30,85.84),'Brajrajnagar':(21.82,83.92),'Chennai':(13.08,80.27),
    'Coimbatore':(11.02,76.96),'Dehradun':(30.32,78.03),'Delhi':(28.66,77.22),
    'Durgapur':(23.55,87.32),'Firozabad':(27.15,78.39),'Gandhinagar':(23.22,72.65),
    'Gangtok':(27.33,88.61),'Gorakhpur':(26.76,83.37),'Gurugram':(28.46,77.03),
    'Guwahati':(26.14,91.74),'Howrah':(22.58,88.31),'Hyderabad':(17.38,78.49),
    'Imphal':(24.82,93.94),'Jaipur':(26.91,75.79),'Jalandhar':(31.33,75.57),
    'Jhansi':(25.45,78.57),'Jorapokhar':(23.70,86.42),'Kanpur':(26.46,80.33),
    'Kochi':(9.93,76.26),'Kolkata':(22.57,88.36),'Lucknow':(26.85,80.95),
    'Ludhiana':(30.90,75.85),'Meerut':(28.98,77.71),'Mumbai':(19.08,72.88),
    'Nagpur':(21.15,79.09),'Nashik':(19.99,73.79),'Noida':(28.54,77.39),
    'Ooty':(11.41,76.70),'Pathardih':(23.77,86.37),'Patiala':(30.34,76.39),
    'Patna':(25.61,85.14),'Prayagraj':(25.45,81.84),'Pune':(18.52,73.86),
    'Raipur':(21.25,81.63),'Rishikesh':(30.09,78.27),'Shillong':(25.57,91.88),
    'Siliguri':(26.72,88.43),'Talcher':(20.95,85.23),'Thiruvananthapuram':(8.52,76.94),
    'Varanasi':(25.32,83.01),'Visakhapatnam':(17.69,83.22)
}

print("Loading data...")
df_city = pd.read_csv('data/fire_count_by_city.csv')
df_city['Date']  = pd.to_datetime(df_city['Date'])
df_city['month'] = df_city['Date'].dt.month
df_city['year']  = df_city['Date'].dt.year

stacks, transforms = {}, {}
for year, path in [
    (2023,'data/HCHO_2023_monthly.tif'),
    (2024,'data/HCHO_2024_monthly.tif')
]:
    with rasterio.open(path) as src:
        stacks[year] = src.read().astype(np.float64)
        transforms[year] = src.transform
    stacks[year][stacks[year] < 0] = np.nan
    stacks[year][stacks[year] > 5e-3] = np.nan

records = []
for city, region in REGION_MAP.items():
    if city not in CITY_COORDS: continue
    lat, lon = CITY_COORDS[city]
    for year in [2023, 2024]:
        t = transforms[year]
        stack = stacks[year]
        col = int((lon - t.c) / t.a)
        row = int((lat - t.f) / t.e)
        h, w = stack.shape[1], stack.shape[2]
        if not (0 <= row < h and 0 <= col < w): continue
        hcho_m = [stack[m,row,col]*1e6 for m in range(12)]
        city_fire = df_city[(df_city['City']==city) & (df_city['year']==year)]
        fire_m = [city_fire[city_fire['month']==m]['fire_count'].sum() for m in range(1,13)]
        valid = [(h,f) for h,f in zip(hcho_m,fire_m) if not np.isnan(h)]
        r = pearsonr([v[0] for v in valid],[v[1] for v in valid])[0] if len(valid)>3 else 0
        records.append({
            'City':city,'Region':region,'Year':year,
            'ann_hcho':np.nanmean(hcho_m),
            'total_fire':sum(fire_m),
            'pm_fire':fire_m[9]+fire_m[10],
            'fire_hcho_r':r
        })

df_rec = pd.DataFrame(records)
rg = df_rec.groupby('Region').agg(
    mean_hcho=('ann_hcho','mean'), total_fire=('total_fire','sum'),
    pm_fire=('pm_fire','sum'), fire_hcho_r=('fire_hcho_r','mean')
).reset_index()

def norm(s): return (s-s.min())/(s.max()-s.min()+1e-10)*100
rg['ISRI'] = (0.30*norm(rg['mean_hcho']) + 0.30*norm(rg['total_fire']) +
              0.25*norm(rg['pm_fire']) + 0.15*norm(rg['fire_hcho_r'].clip(lower=0))).round(1)
rg = rg.sort_values('ISRI', ascending=False).reset_index(drop=True)
rg['Rank'] = rg.index + 1

print("\n=== INDIA SOURCE REGION INDEX ===")
print(rg[['Rank','Region','ISRI','mean_hcho','total_fire','pm_fire']].head(15).to_string(index=False))
rg.to_csv('output/india_source_region_index.csv', index=False)

# Plot
top15 = rg.head(15).copy()
COLOR = {'Punjab':'#e74c3c','Haryana':'#e74c3c','Delhi NCR':'#e74c3c',
         'Uttar Pradesh':'#e74c3c','Jharkhand':'#e67e22','Chhattisgarh':'#e67e22',
         'Odisha':'#e67e22','Madhya Pradesh':'#e67e22'}
colors = [COLOR.get(r,'#3498db') for r in top15['Region']]

fig, axes = plt.subplots(1, 2, figsize=(18,8), gridspec_kw={'width_ratios':[2,1]})
fig.suptitle('India Source Region Index (ISRI)\nIdentification of Major HCHO Source Regions',
             fontsize=15, fontweight='bold')

ax = axes[0]
bars = ax.barh(top15['Region'][::-1], top15['ISRI'][::-1],
               color=list(reversed(colors)), edgecolor='white', height=0.7)
for bar, score in zip(bars, top15['ISRI'][::-1]):
    ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
            f'{score:.1f}', va='center', fontsize=10, fontweight='bold')
ax.set_xlabel('ISRI Score (0-100)', fontsize=12)
ax.set_title('Top 15 HCHO Source Regions', fontsize=13, fontweight='bold')
ax.set_xlim(0, 100)
ax.grid(True, alpha=0.3, linestyle='--', axis='x')
ax.set_facecolor('#fafafa')
patches = [mpatches.Patch(color='#e74c3c',label='Agricultural/Anthropogenic (IGP)'),
           mpatches.Patch(color='#e67e22',label='Industrial + Forest Fire'),
           mpatches.Patch(color='#3498db',label='Other')]
ax.legend(handles=patches, fontsize=9)

ax2 = axes[1]
ax2.axis('off')
top10 = rg.head(10)
tdata = [[f"#{int(r['Rank'])}",r['Region'],f"{r['ISRI']:.1f}",
          f"{r['mean_hcho']*1e6:.0f}" if r['mean_hcho']<1 else f"{r['mean_hcho']:.0f}",
          f"{int(r['total_fire']):,}"] for _,r in top10.iterrows()]
table = ax2.table(cellText=tdata,
                  colLabels=['Rank','Region','ISRI','HCHO','Fires'],
                  cellLoc='center', loc='center', bbox=[0,0,1,1])
table.auto_set_font_size(False)
table.set_fontsize(9)
for j in range(5):
    table[0,j].set_facecolor('#1F4E79')
    table[0,j].set_text_props(color='white', fontweight='bold')
for i in range(1,11):
    for j in range(5):
        table[i,j].set_facecolor('#F5F9FF' if i%2==0 else '#FFFFFF')
ax2.set_title('Score Breakdown — Top 10', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('output/images/india_source_region_index.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: output/images/india_source_region_index.png")
print("Saved: output/india_source_region_index.csv")