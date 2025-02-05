from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the trained model
model = joblib.load("thorocare_model.pkl")

# Load encoders and scaler
encoders = joblib.load("label_encoders.pkl")
scaler = joblib.load("scaler.pkl")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        
        # Encode categorical variables
        categorical_columns = ["Therapy_Readiness", "Health_Status"]
        for col in categorical_columns:
            df[col] = encoders[col].transform(df[col])
        
        # Scale numerical variables
        numeric_columns = ["Heart_Rate", "Respiration_Rate", "Temperature", "Movement_Activity", "Reaction_Score"]
        df[numeric_columns] = scaler.transform(df[numeric_columns])
        
        # Make predictions
        prediction = model.predict(df)
        
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
