import numpy as np
import pandas as pd

df = pd.read_csv('dataset/mydata_v1/h57.csv')
target = df['OT']
import matplotlib.pyplot as plt
# 1.1 单目标序列平稳性/可预测性
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.tools import coint
adf = adfuller(target)                       # p>0.05 → 非平稳
print('ADF p=%.3f' % adf[1])

# 1.2 多变量冗余度 → 57维很可能有共线
import seaborn as sns
sns.heatmap(df.corr(), vmin=-1, vmax=1)
# 看与目标列绝对值<0.05的直接踢掉，先暴力降维

# 1.3 目标自相关+滞后互信息
from sklearn.feature_selection import mutual_info_regression
lags = range(1, 144)  # 24h回看
mic = [mutual_info_regression(df.values[:-lag], target[lag:])[0] for lag in lags]
plt.plot(lags, mic); plt.axhline(np.percentile(mic, 95), color='r', ls='--')
# 若全低于0.02 → 目标本身几乎白噪声，任何模型都会过拟合

# 1.4 谱能量泄漏检查
from scipy.signal import welch
f, Pxx = welch(target, fs=1/600)  # 10min=600s
plt.semilogy(f, Pxx); plt.xlim(0, 1e-3)
# 若50%能量集中在<2 cycle/day → 低频主导，需要差分或季节性分解