import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(
    '../dataset/traffic/traffic.csv',
    parse_dates=['date'],   # 自动转 datetime
    index_col='date'        # 设为索引
)
df = df[:][:57]
sns.set_theme(style="whitegrid", font_scale=0.8)
g = sns.FacetGrid(data=df.reset_index().melt(id_vars='date',
                                             var_name='channel',
                                             value_name='value'),
                  row='channel', hue='channel',
                  height=0.5, aspect=10,  # 高 0.5 英寸，宽 10 英寸
                  sharey=False,           # 各通道 y 轴独立
                  sharex=True,            # 共用时间轴
                  palette='tab20',        # 20 色循环
                  row_order=df.std().sort_values(ascending=False).index
                  )  # 按活跃降序

g.map(plt.plot, 'date', 'value', linewidth=0.6, alpha=0.9)
g.set_titles("{row_name}")          # 子图标题 = 通道名
g.set_axis_labels("Time", "")       # 只留最底部 x 轴
g.fig.subplots_adjust(hspace=0.5)  # 子图竖向间距几乎 0
plt.suptitle('57-D Time Series (clear view)', y=0.995)
plt.show()