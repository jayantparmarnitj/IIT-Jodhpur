import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. Fetch TVS Motor data from Yahoo Finance (India)
ticker = yf.Ticker("TVSMOTOR.NS")
data = ticker.history(start="2022-01-01", end="2025-08-01")

if data.empty:
    raise ValueError("No data downloaded. Please check ticker or internet.")

# 2. Create features
data["Lag1"] = data["Close"].shift(1)
data["MA5"] = data["Close"].rolling(5).mean()
data["VolumeLag1"] = data["Volume"].shift(1)
data = data.dropna()

# 3. Split features and target
X = data[["Lag1", "MA5", "VolumeLag1"]]
y = data["Close"]

# 4. Train/test split (time-series, no shuffling)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# 5. Train linear regression
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Predictions
y_pred = model.predict(X_test)

# 7. Evaluation metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R²:", r2)
print("Coefficients:", dict(zip(X.columns, model.coef_)))

# 8. Plot actual vs predicted
plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test, label="Actual")
plt.plot(y_test.index, y_pred, label="Predicted")
plt.legend()
plt.title("TVSMOTOR Stock Price Prediction — Linear Regression")
plt.xlabel("Date")
plt.ylabel("Closing Price (INR)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
