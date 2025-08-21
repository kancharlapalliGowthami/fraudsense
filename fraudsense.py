import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report

# --------------------------
# 1. Load Dataset
# --------------------------
data = pd.read_csv("transactions.csv")  # Replace with your dataset

# Example expected columns: ['amount', 'time', 'location', 'isFraud']
print("Dataset Preview:")
print(data.head())

# --------------------------
# 2. Preprocess Data
# --------------------------
X = data.drop(columns=['isFraud'], errors='ignore')
y = data['isFraud'] if 'isFraud' in data.columns else None

# Handle NaN values
X = X.fillna(0)

# --------------------------
# 3. Train Model
# --------------------------
model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
model.fit(X)

# --------------------------
# 4. Predictions
# --------------------------
preds = model.predict(X)
# IsolationForest returns: -1 = anomaly, 1 = normal
preds = np.where(preds == -1, 1, 0)  # Convert to 1=fraud, 0=normal

if y is not None:
    print("\nModel Evaluation:")
    print(classification_report(y, preds))

# --------------------------
# 5. Test Real-time Transaction
# --------------------------
def check_transaction(transaction_features):
    """
    transaction_features: list like [amount, time, location,...]
    """
    df = pd.DataFrame([transaction_features], columns=X.columns)
    result = model.predict(df)[0]
    return "🚨 FRAUD DETECTED!" if result == -1 else "✅ Legit Transaction"

# Example test
sample = X.iloc[0].tolist()
print("\nSample Test Transaction:")
print(check_transaction(sample))
