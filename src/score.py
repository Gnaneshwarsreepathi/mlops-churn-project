import json
import joblib
import pandas as pd

model = None

def init():
    global model
    model = joblib.load("churn_model.pkl")

def run(raw_data):
    data = json.loads(raw_data)

    df = pd.DataFrame([data])

    prediction = model.predict(df)
    probability = model.predict_proba(df)

    return {
        "prediction": int(prediction[0]),
        "churn_probability": float(probability[0][1])
    }