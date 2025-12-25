import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("traffic_data.csv")

# Plot traffic density over time
plt.figure(figsize=(8,5))
plt.plot(data['hour'], data['vehicle_count'], marker='o')
plt.xlabel("Hour of Day")
plt.ylabel("Vehicle Count")
plt.title("Traffic Density Over Time")
plt.show()

# Identify peak traffic hours
peak_hours = data[data['vehicle_count'] > 150]
print("Peak Traffic Hours:")
print(peak_hours)

# Optional: Predict traffic using regression
X = data[['hour']]
y = data['vehicle_count']

model = LinearRegression()
model.fit(X, y)

data['predicted_traffic'] = model.predict(X)

# Plot actual vs predicted
plt.figure(figsize=(8,5))
plt.plot(data['hour'], y, label="Actual Traffic")
plt.plot(data['hour'], data['predicted_traffic'], linestyle='--', label="Predicted Traffic")
plt.legend()
plt.title("Traffic Trend Prediction")
plt.show()
