import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf

from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
from joblib import load
from sklearn.base import BaseEstimator, TransformerMixin

# Replacing "$" and ',' with empty space
def clean_currency(val):
    if isinstance(val, str):
        return val.replace("$", "").replace(",", "")
    return val

# Function to compute calculated columns
def add_calculated_columns(df):
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["expires"] = pd.to_datetime(df["expires"], format="%m/%Y", errors="coerce").dt.to_period("M")
    df["acct_open_date"] = pd.to_datetime(df["acct_open_date"], format="%m/%Y", errors="coerce").dt.to_period("M")
    
    df["account_age_months"] = (df["date"].dt.to_period("M") - df["acct_open_date"]).apply(lambda x: x.n)
    df["months_until_expiry"] = (df["expires"] - df["date"].dt.to_period("M")).apply(lambda x: x.n)
    df["years_since_pin_changed"] = df["year_pin_last_changed"] - df["acct_open_date"].dt.year

    return df

# Log1pTransformer
class Log1pTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.log1p(np.clip(X, 0, None))

# CreditScoreLogTransformer
class CreditScoreLogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.max_credit_score = None

    def fit(self, X, y=None):
        self.max_credit_score = X.max().max()
        return self

    def transform(self, X):
        return np.log1p(self.max_credit_score - X)


app = Flask(__name__)


# Load the preprocessing pipeline and model
pipeline = load("preprocessing_pipeline.joblib")

with open("deep_learning_model.pkl", "rb") as f:
    model = pickle.load(f)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if a file was uploaded
        if "file" in request.files:
            file = request.files["file"]
            df = pd.read_csv(file)
        else:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No data received"}), 400
            df = pd.DataFrame([data])

        
        if "id" not in df.columns:
            df["id"] = range(1, len(df) + 1)  # Assign default ID if missing

        
        numeric_cols = ['amount', 'per_capita_income', 'yearly_income', 'total_debt', 'credit_limit']
        for col in numeric_cols:
            df[col] = df[col].apply(clean_currency).astype(float)

    
        df = add_calculated_columns(df)

        df["merchant_state"] = df["merchant_state"].fillna("Unknown")

        
        required_columns = ['amount', 'current_age', 'account_age_months', 'months_until_expiry', 
                            'retirement_age', 'latitude', 'longitude', 'cvv', 'total_debt', 'credit_limit', 
                            'years_since_pin_changed', 'num_credit_cards', 'num_cards_issued', 'yearly_income', 
                            'credit_score', 'use_chip', 'gender', 'card_brand', 'card_type', 'merchant_city', 
                            'merchant_state', 'mcc_name']

        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            return jsonify({"error": f"Missing columns: {missing_cols}"}), 400

        X_input = pipeline.transform(df[required_columns])

        # Predict
        probabilities = model.predict(X_input).flatten() 
        is_fraud = (probabilities > 0.5).astype(int)

        # Convert to JSON response including 'id'
        response = [{"id": int(i), "is_fraud": int(f), "probability": float(p)} 
                    for i, f, p in zip(df["id"], is_fraud, probabilities)]

        print("Response:", response)
        return jsonify(response)

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
