from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

def ml_btst_strategy(data, model_type="lr", **kwargs):
    """
    ML-based BTST strategy.
    model_type: 'lr' (Linear Regression), 'nb' (Naive Bayes), 'knn' (k-Nearest Neighbors)
    """

    df = data.copy()
    df["Return1"] = df["Close"].pct_change().fillna(0)
    df["Lag1"] = df["Return1"].shift(1).fillna(0)
    df["Lag2"] = df["Return1"].shift(2).fillna(0)

    X = df[["Lag1", "Lag2"]]
    y = np.where(df["Return1"].shift(-1).fillna(0) > 0, 1, 0)  # 1 = buy, 0 = sell

    # Choose model
    if model_type == "lr":
        model = LinearRegression()
        model.fit(X, df["Return1"].shift(-1).fillna(0))  # regression target
        preds = model.predict(X)
        signals = pd.DataFrame(index=df.index)
        signals["positions"] = np.where(preds > 0, 1.0, -1.0)

    elif model_type == "nb":
        model = GaussianNB()
        model.fit(X, y)
        preds = model.predict(X)
        signals = pd.DataFrame(index=df.index)
        signals["positions"] = np.where(preds == 1, 1.0, -1.0)

    elif model_type == "knn":
        n_neighbors = kwargs.get("n_neighbors", 5)
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(X, y)
        preds = model.predict(X)
        signals = pd.DataFrame(index=df.index)
        signals["positions"] = np.where(preds == 1, 1.0, -1.0)

    else:
        raise ValueError("Unknown model_type. Use 'lr', 'nb', or 'knn'.")

    return signals
