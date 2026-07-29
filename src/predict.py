import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/flood_model.pkl")

def predict_flood_risk(input_data):
    prediction = model.predict(input_data)[0]

    risk_labels = {
        0: "Low Risk",
        1: "Medium Risk",
        2: "High Risk"
    }

    return risk_labels[prediction]


if __name__ == "__main__":

    sample = pd.DataFrame([[
        5, 5, 5, 5, 5,
        5, 5, 5, 5, 5,
        5, 5, 5, 5, 5,
        5, 5, 5, 5, 5
    ]], columns=model.feature_names_in_)

    result = predict_flood_risk(sample)

    print(f"Predicted Flood Risk: {result}")