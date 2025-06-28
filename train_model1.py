# train_model.py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  # you can change to other models
from sklearn.metrics import mean_squared_error, r2_score

# Load Data
df = pd.read_csv("data.csv")
print("First 5 rows:")
print(df.head())



# Basic Info
print("\nInfo:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# Drop non-numeric and irrelevant columns
df_numeric = df.drop(columns=["date", "street", "city", "statezip", "country"])

# Compute correlation matrix
correlation_matrix = df_numeric.corr().abs()

# Select upper triangle of correlation matrix
upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

# Identify columns with correlation > 0.9
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]

# Drop highly correlated features
df_reduced = df_numeric.drop(columns=to_drop)

# Define features and target
X = df_reduced.drop(columns=["price"])
y = df_reduced["price"]

print(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Output results
print("Highly Correlated Columns Removed:", to_drop)
print("R² Score:", r2)
print("Root Mean Squared Error (RMSE):", rmse)

# Save model
with open("trained_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save columns
with open("model_columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)
