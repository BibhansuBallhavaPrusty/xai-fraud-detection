
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import os

df = pd.read_csv('data/creditcard.csv')

X = df.drop('Class', axis=1)
y = df['Class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Save model and scaler
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/fraud_xgb_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print(" Model and scaler saved.")