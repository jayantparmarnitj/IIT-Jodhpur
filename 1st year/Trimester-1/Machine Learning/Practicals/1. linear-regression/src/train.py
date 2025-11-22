from __future__ import annotations
import argparse, json, os
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import matplotlib.pyplot as plt

from metrics import RegressionReport

def make_demo_dataframe(n_samples: int = 500, n_features: int = 8, noise: float = 10.0, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    true_coefs = rng.uniform(-5, 5, size=n_features)
    y = X @ true_coefs + rng.normal(scale=noise, size=n_samples) + 25
    cols = [f"x{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df["y"] = y
    return df

def pick_model(name: str, alpha: float, degree: int):
    name = name.lower()
    if name == "linear":
        return LinearRegression()
    if name == "ridge":
        return Ridge(alpha=alpha)
    if name == "lasso":
        return Lasso(alpha=alpha, max_iter=10000)
    if name == "poly":
        return Pipeline([
            ("scale", StandardScaler()),
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("linreg", LinearRegression())
        ])
    raise ValueError(f"Unknown model '{name}'. Choose from: linear, ridge, lasso, poly.")

def load_dataset(args):
    if args.dataset == "housing":
        data = fetch_california_housing(as_frame=True)
        df = data.frame
        target = "MedHouseVal"
        return df, target
    elif args.demo:
        return make_demo_dataframe(seed=args.seed), "y"
    else:
        df = pd.read_csv(args.csv)
        return df, args.target

def run_one_model(model_name, X, y, feature_names, args, alpha=None, degree=2):
    model = pick_model(model_name, alpha, degree)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_report = RegressionReport.from_predictions(y_train, y_train_pred, X_train.shape[1])
    test_report = RegressionReport.from_predictions(y_test, y_test_pred, X_train.shape[1])

    return {
        "model": model_name,
        "alpha": alpha,
        "degree": degree if model_name == "poly" else None,
        "train": train_report.to_dict(),
        "test": test_report.to_dict(),
    }

def plot_bar(results, outpath):
    plt.figure()
    names = [r["model"] + (f"(deg={r['degree']})" if r["model"]=="poly" else "") for r in results]
    r2s = [r["test"]["r2"] for r in results]
    plt.bar(names, r2s)
    plt.ylabel("Test R²")
    plt.title("Model Comparison")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(outpath, dpi=140)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Train & compare regression models on housing/demo/CSV.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=str, help="Path to input CSV")
    src.add_argument("--demo", action="store_true", help="Use a synthetic demo dataset")
    src.add_argument("--dataset", choices=["housing"], help="Use a built-in dataset")

    parser.add_argument("--target", type=str, help="Target column name (ignored for housing/demo)")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--model", choices=["linear", "ridge", "lasso", "poly", "all"], default="linear")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--degree", type=int, default=2, help="Polynomial degree (for poly)")
    parser.add_argument("--outdir", type=str, default="outputs")

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df, target = load_dataset(args)
    y = df[target].to_numpy()
    X = df.drop(columns=[target]).select_dtypes(include=[np.number]).to_numpy()
    feature_names = [c for c in df.drop(columns=[target]).select_dtypes(include=[np.number]).columns]

    models_to_run = [args.model] if args.model != "all" else ["linear", "ridge", "lasso", "poly"]

    results = []
    for m in models_to_run:
        res = run_one_model(m, X, y, feature_names, args, alpha=args.alpha, degree=args.degree)
        results.append(res)

    # Save comparison if multiple models
    if len(results) > 1:
        with open(os.path.join(args.outdir, "comparison.json"), "w") as f:
            json.dump(results, f, indent=2)
        plot_bar(results, os.path.join(args.outdir, "comparison.png"))
        print(f"✅ Comparison saved in {args.outdir}/comparison.json and comparison.png")
    else:
        # single model -> save metrics.json
        with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
            json.dump(results[0], f, indent=2)
        print(f"✅ Single model metrics saved in {args.outdir}/metrics.json")

if __name__ == "__main__":
    main()
