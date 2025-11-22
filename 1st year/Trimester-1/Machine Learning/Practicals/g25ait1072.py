# %% [markdown]
# # Assignment: Implementing Linear and Polynomial Regression from Scratch
# 
# **Roll Number:** g25ait1072
# 
# ## Objective
# The goal of this assignment is to implement and understand the Linear and Polynomial Regression models for predicting continuous values using Gradient Descent, without relying on built-in model training functions like `sklearn.linear_model`.
# 
# ## Dataset
# The project uses the **California Housing Dataset** from `sklearn.datasets`.
# 
# ## Assumptions
# 1. The dataset is a representative sample of the housing market it describes.
# 2. The relationship between the features and the target can be reasonably approximated by linear and polynomial models.
# 3. The hyperparameters (learning rate, iterations) have been chosen for demonstration purposes and may require further tuning for optimal performance.
# 
# ## Resources Used
# - Scikit-learn documentation for dataset loading, preprocessing (`StandardScaler`, `PolynomialFeatures`), and metrics (`mean_squared_error`, `r2_score`).
# - Numpy documentation for numerical operations.
# - Matplotlib & Seaborn documentation for plotting.

# %% [markdown]
# ---
# ## 1. Initial Setup and Library Imports

# %%
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
import os

# %% [markdown]
# ---
# ## Part (a): Implementation of the Regression Models
# 
# ### 2.1. Custom Linear Regression Model from Scratch
# 
# This class implements the Linear Regression algorithm using Gradient Descent. It is built from the ground up to handle the core logic of fitting the model and making predictions. It also includes an optional L2 regularization term (Ridge).

# %%
class CustomLinearRegression:
    """
    A custom implementation of Linear Regression using Gradient Descent.
    Includes L2 Regularization (Ridge) as a bonus feature.
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000, lambda_=0):
        """
        Initializes the model with hyperparameters.
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.lambda_ = lambda_
        self.weights = None
        self.bias = None
        self.cost_history = []

    def fit(self, X, y):
        """
        Trains the model by finding the optimal weights and bias using Gradient Descent.
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.cost_history = []
        
        print(f"  - Starting Gradient Descent (lr={self.learning_rate}, iter={self.n_iterations}, lambda={self.lambda_})...")
        for i in range(self.n_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            cost = np.mean((y_predicted - y)**2)
            
            # FIX: Add a safety check for exploding gradients.
            # If cost is NaN or infinity, the learning rate is too high. Stop training.
            if not np.isfinite(cost):
                print(f"  - Cost has exploded at iteration {i+1}. Stopping training.")
                print("  - Try using a smaller learning rate.")
                # Fill remaining cost history with NaN for plotting purposes if needed
                self.cost_history.extend([np.nan] * (self.n_iterations - len(self.cost_history)))
                break
                
            self.cost_history.append(cost)
            
            regularization_term = (self.lambda_ / n_samples) * self.weights
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y)) + regularization_term
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if (i + 1) % 100 == 0:
                print(f"  - Iteration {i+1}/{self.n_iterations}, Cost (MSE): {cost:.4f}")

    def predict(self, X):
        """
        Makes a prediction using the trained weights and bias.
        """
        # If weights are None (e.g., training failed), return an array of NaNs
        if self.weights is None:
            return np.full(X.shape[0], np.nan)
        return np.dot(X, self.weights) + self.bias

# %% [markdown]
# ### 2.2. Encapsulated Pipeline Design Pattern
# 
# This class encapsulates the entire machine learning workflow—loading, preprocessing, training, and evaluation—into a single, reusable object. This design pattern makes the code cleaner, more organized, and easier to manage for different models (like Linear vs. Polynomial).

# %%
class RegressionPipeline:
    """
    An encapsulated pipeline for regression models.
    """
    def __init__(self, model, degree=1):
        """
        Initializes the pipeline with a model instance and polynomial degree.
        """
        self.model = model
        self.scaler = StandardScaler()
        self.poly_features = None
        self.degree = degree
        if self.degree > 1:
            self.poly_features = PolynomialFeatures(degree=self.degree, include_bias=False)
        self.features = None
        self.target = None
        self.train_metrics = {}
        self.test_metrics = {}

    def _load_data(self):
        """Loads and prepares the California Housing dataset."""
        print("1. Loading California Housing dataset...")
        housing = fetch_california_housing()
        df = pd.DataFrame(housing.data, columns=housing.feature_names)
        df['MedHouseVal'] = housing.target
        self.features = housing.feature_names
        self.target = 'MedHouseVal'
        return df

    def _preprocess_data(self, df: pd.DataFrame):
        """Preprocesses the data: splits, creates polynomial features, and scales."""
        print(f"2. Preprocessing data (Polynomial Degree: {self.degree})...")
        df.dropna(inplace=True)
        X = df[self.features]
        y = df[self.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create polynomial features if degree > 1
        if self.poly_features:
            X_train = self.poly_features.fit_transform(X_train)
            X_test = self.poly_features.transform(X_test)
        
        # Standardize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("   - Data split and features prepared.")
        return X_train_scaled, X_test_scaled, y_train.to_numpy(), y_test.to_numpy()

    def perform_eda(self, df: pd.DataFrame):
        """Performs Exploratory Data Analysis (EDA)."""
        print("\n--- [ANALYSIS] Performing Exploratory Data Analysis ---")
        print("\nData Description:\n", df.describe())
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap='viridis', fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.show()

    def train(self, X_train, y_train):
        """Trains the provided model."""
        print("3. Training the model...")
        self.model.fit(X_train, y_train)
        print("   - Model training complete.")

    def _evaluate(self, X, y_true, dataset_name: str):
        """Evaluates the model and prints metrics."""
        print(f"4. Evaluating model on {dataset_name} set...")
        y_pred = self.model.predict(X)
        
        # Handle cases where prediction might have failed and returned NaNs
        if np.isnan(y_pred).any():
            mse, r2 = np.nan, np.nan
            print(f"   - Could not evaluate {dataset_name} set due to NaN predictions.")
        else:
            mse = mean_squared_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            print(f"   - {dataset_name} MSE: {mse:.4f}")
            print(f"   - {dataset_name} R-squared: {r2:.4f}")
            
        metrics = {"MSE": mse, "R-squared": r2}
        return metrics, y_pred

    def plot_learning_curve(self):
        """Plots the learning curve (Cost vs. Iterations)."""
        print("\n--- [VISUALIZATION] Plotting learning curve ---")
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.model.cost_history) + 1), self.model.cost_history)
        plt.xlabel("Number of Iterations")
        plt.ylabel("Cost (MSE)")
        plt.title(f"Learning Curve (Degree={self.degree}, LR={self.model.learning_rate})")
        plt.grid(True)
        plt.show()

# %% [markdown]
# ---
# ## Part (b): Evaluation and Visualization
# 
# ### 3.1. Base Model: Linear Regression (Degree 1)

# %%
# --- Instantiate and Run the Linear Regression Pipeline ---
print("\n" + "="*50)
print("RUNNING: Standard Linear Regression (Degree=1)")
print("="*50)

# Create a model instance for linear regression
linear_model = CustomLinearRegression(learning_rate=0.1, n_iterations=1000, lambda_=0)
linear_pipeline = RegressionPipeline(model=linear_model, degree=1)

# Load and process data
data = linear_pipeline._load_data()
X_train_lin, X_test_lin, y_train_lin, y_test_lin = linear_pipeline._preprocess_data(data)

# Train and evaluate
linear_pipeline.train(X_train_lin, y_train_lin)
train_metrics_lin, _ = linear_pipeline._evaluate(X_train_lin, y_train_lin, "Training")
test_metrics_lin, y_pred_lin = linear_pipeline._evaluate(X_test_lin, y_test_lin, "Testing")
linear_pipeline.plot_learning_curve()

# %% [markdown]
# ### 3.2. Visualization: Actual vs. Predicted Values (Linear Model)

# %%
print("\n--- [VISUALIZATION] Plotting test set predictions vs actuals (Linear Model) ---")
plt.figure(figsize=(8, 6))
plt.scatter(y_test_lin, y_pred_lin, alpha=0.7, edgecolors='k')
plt.plot([y_test_lin.min(), y_test_lin.max()], [y_test_lin.min(), y_test_lin.max()], 'r--', lw=2)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Linear Regression: Actual vs. Predicted")
plt.grid(True)
plt.show()

# %% [markdown]
# ---
# ## Part (c): Polynomial Regression with Different Degrees
# 
# In this section, we will extend the analysis to Polynomial Regression. We will train the model with different polynomial degrees (1, 2, and 3) to observe how increasing model complexity affects performance on both the training and testing data. This is a practical demonstration of the bias-variance tradeoff.

# %%
# --- Run Polynomial Regression for different degrees ---
degrees = [1, 2, 3]
results = []

for degree in degrees:
    print("\n" + "="*50)
    print(f"RUNNING: Polynomial Regression (Degree={degree})")
    print("="*50)
    
    # Hyperparameters may need tuning for higher degrees
    # FIX: Use a much smaller learning rate for degree 3 to prevent exploding gradients.
    if degree == 1:
        lr = 0.1
    elif degree == 2:
        lr = 0.01
    else: # degree == 3
        lr = 0.001
        
    iterations = 1000
    
    poly_model = CustomLinearRegression(learning_rate=lr, n_iterations=iterations, lambda_=0)
    poly_pipeline = RegressionPipeline(model=poly_model, degree=degree)
    
    # Preprocess the data with the current degree
    X_train_poly, X_test_poly, y_train_poly, y_test_poly = poly_pipeline._preprocess_data(data)
    
    # Train and evaluate
    poly_pipeline.train(X_train_poly, y_train_poly)
    train_metrics, _ = poly_pipeline._evaluate(X_train_poly, y_train_poly, "Training")
    test_metrics, _ = poly_pipeline._evaluate(X_test_poly, y_test_poly, "Testing")
    
    results.append({
        'Degree': degree,
        'Train MSE': train_metrics['MSE'],
        'Test MSE': test_metrics['MSE'],
        'Train R-squared': train_metrics['R-squared'],
        'Test R-squared': test_metrics['R-squared'],
        'Num_Features': X_train_poly.shape[1]
    })

# %% [markdown]
# ### 4.1. Comparative Analysis of Model Performance
# 
# The table below summarizes the performance of the regression model for each polynomial degree. We can observe how the training error decreases as complexity increases, while the test error might start to increase after a certain point (indicating overfitting).

# %%
# --- Display results in a table ---
results_df = pd.DataFrame(results)
print("\n" + "="*50)
print("COMPARATIVE ANALYSIS OF MODELS")
print("="*50)
print(results_df.to_markdown(index=False))

# %% [markdown]
# ### 4.2. Visualizing the Bias-Variance Tradeoff
# 
# The plot below shows the training and testing MSE as a function of the polynomial degree. This visualization is a classic way to identify the "sweet spot" for model complexity, where the model generalizes best to unseen data without overfitting.

# %%
# --- Plot the effect of degree on error ---
plt.figure(figsize=(10, 6))
plt.plot(results_df['Degree'], results_df['Train MSE'], 'o-', label='Training MSE')
plt.plot(results_df['Degree'], results_df['Test MSE'], 'o-', label='Testing MSE')
plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("Model Complexity vs. Error (Bias-Variance Tradeoff)")
plt.xticks(degrees)
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# ---
# ## 5. Bonus Challenges
# 
# ### 5.1. Bonus: L2 Regularization (Ridge)
# 
# Here, we run the linear regression pipeline again but with a non-zero lambda value to apply L2 regularization.

# %%
# --- Bonus: L2 Regularization (Ridge) on Linear Model ---
print("\n" + "="*50)
print("BONUS: L2 Regularization (Ridge) on Linear Model")
print("="*50)

# Create a regularized model instance
ridge_model = CustomLinearRegression(learning_rate=0.1, n_iterations=1000, lambda_=1.0)
ridge_pipeline = RegressionPipeline(model=ridge_model, degree=1)

# Use the already processed linear data
ridge_pipeline.train(X_train_lin, y_train_lin)
ridge_pipeline._evaluate(X_test_lin, y_test_lin, "Testing with L2")
print("\nObservation: Compare the MSE and R-squared of this regularized model with the non-regularized one in your report.")

# %% [markdown]
# ### 5.2. Bonus: Experimenting with Learning Rates
# 
# This experiment demonstrates the critical effect of the learning rate on the convergence of the Gradient Descent algorithm.

# %%
# --- Bonus: Experiment with Learning Rates ---
print("\n" + "="*50)
print("BONUS: Experimenting with Learning Rates on Linear Model")
print("="*50)

learning_rates = {
    "Too High": 1.1,
    "Too Low": 0.0001,
    "Just Right": 0.1
}

plt.figure(figsize=(12, 7))
for name, lr in learning_rates.items():
    model_lr = CustomLinearRegression(learning_rate=lr, n_iterations=500)
    # We only need to fit the model to get the cost history
    model_lr.fit(X_train_lin, y_train_lin)
    plt.plot(model_lr.cost_history, label=f"LR = {lr} ({name})")

plt.xlabel("Number of Iterations")
plt.ylabel("Cost (MSE)")
plt.title("Effect of Different Learning Rates on Convergence")
plt.legend()
plt.grid(True)
plt.ylim(0, 5) # Limit y-axis to see the convergence details clearly
plt.show()

print("Observation: Analyze the plot in your report. The high learning rate diverges (cost explodes),")
print("the low rate converges very slowly, and the 'just right' rate converges smoothly.")

