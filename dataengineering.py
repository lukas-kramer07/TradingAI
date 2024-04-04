import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_pickle('data/apple')
print(df.tail())

plot_cols = ['open', 'change', 'volume']
date_time = pd.to_datetime(df.pop('date'), format='%Y-%m-%d')
plot_features = df[plot_cols]
plot_features.index = date_time
_ = plot_features.plot(subplots=True)

plot_features = df[plot_cols][:20]
plot_features.index = date_time[:20]
_ = plot_features.plot(subplots=True)
plt.show()