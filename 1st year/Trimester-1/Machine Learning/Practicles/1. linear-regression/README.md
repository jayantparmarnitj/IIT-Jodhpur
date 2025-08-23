# Linear/Polynomial Regression + Full Evaluation

Extended regression project with **California Housing dataset** and **Polynomial Regression**.

## Features
- Built-in support for:
  - Synthetic demo data
  - Your own CSV
  - California Housing dataset (`--dataset housing`)
- Models:
  - Linear Regression
  - Ridge (L2 regularization)
  - Lasso (L1 regularization)
  - Polynomial Regression
- Comparison mode (`--model all`) — trains all models and saves metrics + comparison chart
- Metrics: R², Adjusted R², MSE, RMSE, MAE, MAPE, MedianAE, Explained Variance
- Cross-validation support
- Residuals & Prediction plots
- JSON outputs (metrics, coefficients, predictions)

## Install
```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

## Example: Compare on Housing Dataset
```bash
python src/train.py --dataset housing --model all --outdir outputs
```

## Example: Polynomial Regression (degree=3)
```bash
python src/train.py --dataset housing --model poly --degree 3 --outdir outputs
```

Outputs:
- `comparison.json` — metrics for all models
- `comparison.png` — bar chart of Test R²
- `metrics.json`, `coefs.json`, `predictions.csv` for single model mode
