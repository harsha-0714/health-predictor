from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('health_risk_model.pkl')  # Ensure model is in the same directory

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        required_features = ['age', 'sleep_hours', 'exercise_days', 'diet_quality', 'genetic_variant', 'sentiment_score']
        if not all(k in data for k in required_features):
            return jsonify({"error": "Missing features"}), 400
        df_input = pd.DataFrame([data])
        prediction = model.predict(df_input)[0]
        advice = "Low risk: Maintain healthy habits." if prediction < 40 else "Moderate risk: Consider consulting a professional." if prediction < 70 else "High risk: Seek immediate help."
        return jsonify({"risk_score": round(prediction, 2), "advice": advice})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)  # Remove debug for production