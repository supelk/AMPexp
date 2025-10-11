import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
# from torch.backends.quantized import engine
engine = create_engine(
    'mysql+pymysql://root:supelk@127.0.0.1:3306/amptst_result?charset=utf8mb8'
)
df = pd.read_sql("""
    SELECT model, sl, mse_rsd*100 AS mse_rsd_pct
    FROM pair_fluctuation
""", con=engine)

sns.boxplot(x='model', y='mse_rsd_pct', data=df)
plt.title('MSE Relative Fluctuation (%) between Run-A & Run-B')
plt.show()