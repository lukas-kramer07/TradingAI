import numpy as np
import matplotlib.pyplot as plt

# Existing data
np.random.seed(1)  # For reproducibility
x = np.linspace(0, 10, 100)
y = np.random.random(100)+10

# Forecast triangles
forecast_length = 100
x_forecast_start = x[-1]
x_forecast_end = x_forecast_start + 10
y_forecast_center = y[-1]

# Define forecast windows with different percentage deviations and opacities
windows = [
    {'deviation': 0.02, 'alpha': 0.2},
    {'deviation': 0.015, 'alpha': 0.4},
    {'deviation': 0.01, 'alpha': 0.6},
    {'deviation': 0.005, 'alpha': 0.8}
]

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(x, y, label='Existing Data')

for window in windows:
    deviation = window['deviation']
    y_forecast_top = y_forecast_center * (1 + deviation)
    y_forecast_bottom = y_forecast_center * (1 - deviation)
    plt.fill([x_forecast_start, x_forecast_end, x_forecast_end], 
             [y_forecast_center, y_forecast_top, y_forecast_bottom], 
             color='gray', alpha=window['alpha'], label=f'Forecast ±{deviation*100}%')

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Data with Percentage Deviation Forecast Triangles')
plt.legend()
plt.show()

