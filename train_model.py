import pandas as pd, numpy as np, re, pickle, json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings; warnings.filterwarnings('ignore')
np.random.seed(42)

df = pd.read_csv('data/rumah123_yogya_unfiltered.csv')
def parse_harga(s):
    if pd.isnull(s): return np.nan
    s = str(s).upper().replace('RP','').replace(' ','')
    s = s.replace('.','').replace(',','.')
    try:
        if 'MILIAR' in s or 'M' in s.replace('MILIAR',''):
            val = float(re.sub(r'[^0-9.]','', s.replace('MILIAR','').replace('M','')))
            return val*1_000_000_000
        elif 'JUTA' in s or 'JT' in s:
            val = float(re.sub(r'[^0-9.]','', s.replace('JUTA','').replace('JT','')))
            return val*1_000_000
        else:
            return float(re.sub(r'[^0-9.]','', s))
    except: return np.nan
def parse_luas(s):
    if pd.isnull(s): return np.nan
    m = re.search(r'[\d]+', str(s).replace(',','.'))
    return float(m.group()) if m else np.nan

df['harga'] = df['price'].apply(parse_harga)
df['luas_tanah'] = df['surface_area'].apply(parse_luas)
df['luas_bangunan'] = df['building_area'].apply(parse_luas)
df = df.rename(columns={'listing-location':'lokasi','bed':'kamar_tidur','bath':'kamar_mandi','carport':'garasi'})
df['kabupaten'] = df['lokasi'].apply(lambda x: str(x).split(',')[-1].strip() if pd.notnull(x) else 'Unknown')

base = df[['kabupaten','luas_tanah','luas_bangunan','kamar_tidur','kamar_mandi','garasi','harga']].copy()
base = base.dropna(subset=['harga','luas_tanah','luas_bangunan'])
for col in ['kamar_tidur','kamar_mandi','garasi']:
    base[col] = base[col].fillna(base[col].median())
base = base.dropna().reset_index(drop=True)

def iqr_bounds(s):
    q1,q3 = s.quantile(.25), s.quantile(.75); iqr=q3-q1
    return q1-1.5*iqr, q3+1.5*iqr

keep_idx = []
for kab, g in base.groupby('kabupaten'):
    lo_h,hi_h = iqr_bounds(g['harga']); lo_lt,hi_lt = iqr_bounds(g['luas_tanah']); lo_lb,hi_lb = iqr_bounds(g['luas_bangunan'])
    mask = (g['harga'].between(lo_h,hi_h) & g['luas_tanah'].between(max(lo_lt,1),hi_lt) & g['luas_bangunan'].between(max(lo_lb,1),hi_lb))
    keep_idx.extend(g[mask].index.tolist())
d = base.loc[keep_idx].reset_index(drop=True)

KABUPATEN_LIST = sorted(d['kabupaten'].unique().tolist())
ohe = pd.get_dummies(d['kabupaten'], prefix='kab')
for k in KABUPATEN_LIST:
    col = f'kab_{k}'
    if col not in ohe.columns: ohe[col]=0
ohe = ohe[[f'kab_{k}' for k in KABUPATEN_LIST]]

d2 = pd.concat([d[['luas_tanah','luas_bangunan','kamar_tidur','kamar_mandi','garasi','harga']], ohe], axis=1)
FEATURE_ORDER = ['luas_tanah','luas_bangunan','kamar_tidur','kamar_mandi','garasi'] + [f'kab_{k}' for k in KABUPATEN_LIST]
X = d2[FEATURE_ORDER]; y = d2['harga']
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(Xtr,ytr)
pred = model.predict(Xte)
rmse=np.sqrt(mean_squared_error(yte,pred)); mae=mean_absolute_error(yte,pred); r2=r2_score(yte,pred)
mape=np.mean(np.abs((yte-pred)/yte))*100
print(f'FINAL MODEL (Random Forest, skema 3): R2={r2:.4f} RMSE={rmse:,.0f} MAE={mae:,.0f} MAPE={mape:.2f}%')
print('Jumlah data final training:', d.shape[0])
print('Kabupaten list:', KABUPATEN_LIST)

with open('model/model.pkl','wb') as f:
    pickle.dump({
        'model': model,
        'kabupaten_list': KABUPATEN_LIST,
        'feature_order': FEATURE_ORDER,
        'metrics': {'r2':r2,'rmse':rmse,'mae':mae,'mape':mape},
        'model_name': 'Random Forest'
    }, f)
print('Tersimpan: model.pkl')
