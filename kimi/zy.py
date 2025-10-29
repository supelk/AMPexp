import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(
    '../dataset/traffic/traffic.csv',
    parse_dates=['date'],   # 自动转 datetime
    index_col='date'        # 设为索引
)
# df = pd.read_csv(
#     '../dataset/mydata/data_filled.csv',
#     parse_dates=['date'],   # 自动转 datetime
#     index_col='date'        # 设为索引
# )
print(df.head())
T, C = df.shape
print('长度:', T, '维度:', C)

import matplotlib.pyplot as plt
import seaborn as sns

# 只看最活跃的 3 条曲线
top3 = df.std().nlargest(3).index

plt.figure(figsize=(14,4))
df[top3].plot(subplots=True, sharex=True, title='Top-3 active channels vs. time')
plt.tight_layout(); plt.show()

plt.figure(figsize=(16,16))
sns.heatmap(df.T, cmap='coolwarm', cbar_kws={'label':'value'})
plt.title('57-D Time Series Heatmap')
plt.xlabel('Time step'); plt.ylabel('Channel'); plt.show()

std57 = df.std().sort_values(ascending=False)
plt.figure(figsize=(6,4))
sns.barplot(x=std57.values, y=std57.index, color='steelblue')
plt.title('Std per channel (top→bottom)')
plt.show()

fs = 100         # 采样率（自己改）
top_ch = std57.index[:10].tolist()   # 只画最活跃的 10 维
fft_vals = np.abs(np.fft.rfft(df[top_ch], axis=0))
freqs = np.fft.rfftfreq(T, d=1/fs)

plt.figure(figsize=(8,4))
sns.heatmap(fft_vals.T, cmap='viridis',
            xticklabels=np.round(freqs,1), yticklabels=top_ch)
plt.title('FFT Magnitude (top 10 channels)')
plt.xlabel('Frequency (Hz)'); plt.ylabel('Channel'); plt.show()

plt.figure(figsize=(7,6))
sns.heatmap(df.corr(), cmap='vlag', center=0, square=True, cbar_kws={'shrink':.8})
plt.title('57×57 Correlation')
plt.show()

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pc = pca.fit_transform(df)

plt.figure(figsize=(6,5))
plt.scatter(pc[:,0], pc[:,1], c=np.arange(T), cmap='plasma', s=8)
plt.colorbar(label='time step')
plt.title(f'PCA 2-D projection ({pca.explained_variance_ratio_[:2].sum():.1%} var)')
plt.show()

pca = PCA().fit(df)                       # 全部主成分
cumvar = np.cumsum(pca.explained_variance_ratio_)
print('前2维解释方差:', cumvar[1])
print('前5维解释方差:', cumvar[4])

