import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
import joblib
import numpy as np

# Disable interactive mode for matplotlib
plt.ioff()

# Function to compute RSI
def compute_rsi(series, period=14):
    delta = series.diff().dropna()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Load the dataset
data_path = 'D:/MLDP_dataset'
data_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]

if not data_files:
    print("No CSV files found in the specified directory.")
else:
    print(f"Found {len(data_files)} CSV files.")

# Process the first two files
first_one_files = data_files[:1]
print(f"Processing files: {first_one_files}")

# Initialize an empty list to hold the DataFrames
dfs = []

# Read and concatenate the first two files
for file in first_one_files:
    print(f"Reading file: {file}")
    df = pd.read_csv(os.path.join(data_path, file))
    dfs.append(df)

# Concatenate the DataFrames
df = pd.concat(dfs, ignore_index=True)

# Display the first few rows of the combined DataFrame
print("Initial data preview:")
print(df.head())

# Ensure data types are correct
df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
df['High'] = pd.to_numeric(df['High'], errors='coerce')
df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df['Adj Close'] = pd.to_numeric(df['Adj Close'], errors='coerce')
df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

# Handle missing values
df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
print(f"Data after handling missing values:\n{df.info()}")

# Visualize data distributions before cleaning
print("Visualizing data distributions before cleaning...")
df.hist(bins=50, figsize=(20, 15))
plt.suptitle("Before Removing Outliers")
plt.savefig('before_removing_outliers.png')
plt.close()

# Remove extreme outliers
df = df[(df['Open'] > 0) & (df['Open'] < 500)]
df = df[(df['High'] > 0) & (df['High'] < 500)]
df = df[(df['Low'] > 0) & (df['Low'] < 500)]
df = df[(df['Close'] > 0) & (df['Close'] < 500)]
df = df[(df['Adj Close'] > 0) & (df['Adj Close'] < 500)]
df = df[(df['Volume'] > 0) & (df['Volume'] < 1e9)]  # Adjust volume threshold as needed
print(f"Data after removing outliers:\n{df.info()}")

# Visualize data distributions after cleaning
print("Visualizing data distributions after cleaning...")
df.hist(bins=50, figsize=(20, 15))
plt.suptitle("After Removing Outliers")
plt.savefig('after_removing_outliers.png')
plt.close()

# Drop the Date column
if 'Date' in df.columns:
    df.drop('Date', axis=1, inplace=True)
    print("Dropped the Date column.")

# Feature engineering: Adding technical indicators
df['MA50'] = df['Close'].rolling(window=50).mean()
df['MA200'] = df['Close'].rolling(window=200).mean()
df['RSI'] = compute_rsi(df['Close'])
df['Daily_Return'] = df['Close'].pct_change()
df['Volatility'] = df['Close'].rolling(window=21).std()
df = df.dropna()

# Define features and target
features = ['Open', 'MA50', 'MA200', 'RSI', 'Daily_Return', 'Volatility']
X = df[features]
y = df[['High', 'Low', 'Close', 'Adj Close', 'Volume']]
print(f"Features and target defined. X shape: {X.shape}, y shape: {y.shape}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data split into train and test sets. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Features scaled.")

# Save the scaler
scaler_path = 'scaler.joblib'
joblib.dump(scaler, scaler_path)
print(f'Scaler saved to {scaler_path}')

# Define and train the Decision Tree model
model = DecisionTreeRegressor()

# Simplified hyperparameter tuning
param_grid = {
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}
print("Starting hyperparameter tuning...")
grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_
print("Model training completed. Model trained.")

# Get feature importances
feature_importances = best_model.feature_importances_
for feature, importance in zip(features, feature_importances):
    print(f'Feature: {feature}, Importance: {importance}')

# Optional: Plot feature importances
plt.barh(features, feature_importances)
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.savefig('feature_importances.png')
plt.close()

# Evaluate the model
y_pred = best_model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'RMSE: {rmse}')
print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'R-squared: {r2}')

# Save the model
model_path = 'best_model_decision_tree.joblib'
joblib.dump(best_model, model_path)
print(f'Model saved to {model_path}')

# Debugging: Check predictions for a few samples
sample_open_prices = [10, 20, 30, 40, 50]
print("Sample predictions for different Open prices:")
for open_price in sample_open_prices:
    sample_features = np.array([[open_price, 50, 50, 50, 0.01, 0.01]])  # Adjust these values based on your data
    sample_prediction = best_model.predict(scaler.transform(sample_features))
    print(f"Input Open: {open_price} - Predicted: {sample_prediction}")
