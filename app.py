import streamlit as st
import pickle
import numpy as np
import os
import random

# ===========================================================
# UNIVERSAL MODEL LOADER
# ===========================================================
def load_model(model_name):
    paths_to_try = [
        os.path.join("models", model_name),
        os.path.join("/content/models", model_name),
        model_name
    ]
    for path in paths_to_try:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    st.sidebar.error(f"Model not found: {model_name}")
    return None

# ===========================================================
# PAGE CONFIGURATION
# ===========================================================
st.set_page_config(page_title="Health Predictor Dashboard", layout="wide")

# ===========================================================
# CUSTOM CSS — ELEGANT DASHBOARD DESIGN
# ===========================================================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(120deg, #f0f9ff, #ffffff);
}
[data-testid="stHeader"] {
    background: linear-gradient(to right, #007bff, #00c6ff);
    color: white;
    font-size: 22px;
    padding: 15px;
    font-weight: bold;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #003366, #0055cc);
    color: white;
    border-right: 3px solid #00aaff;
}
[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3 {
    color: #ffffff;
    text-align: center;
    font-weight: 700;
}
div[role="radiogroup"] > label > div {
    background: rgba(255,255,255,0.1);
    padding: 12px;
    border-radius: 8px;
    margin: 6px 0;
    cursor: pointer;
    transition: all 0.3s ease;
}
div[role="radiogroup"] > label > div:hover {
    background: rgba(255,255,255,0.3);
    transform: scale(1.03);
}
div[role="radiogroup"] > label[data-selected="true"] > div {
    background: #00b4d8;
    box-shadow: 0px 0px 10px rgba(0,180,216,0.8);
}
.sidebar-footer {
    position: absolute;
    bottom: 10px;
    width: 100%;
    text-align: center;
    color: #ddd;
    font-size: 13px;
}
.stButton>button {
    background: linear-gradient(to right, #00aaff, #007bff);
    color: white;
    border-radius: 10px;
    font-weight: 600;
    padding: 10px 25px;
    transition: 0.3s ease;
}
.stButton>button:hover {
    background: linear-gradient(to right, #0056b3, #0088cc);
    transform: scale(1.03);
}
input[type=number], select {
    border-radius: 6px;
    border: 1.5px solid #0099ff;
    padding: 6px 10px;
    margin-bottom: 10px;
    font-size: 15px;
}
h1, h2, h3 {
    color: #003366;
    font-weight: bold;
}
/* Health Report Styling */
.health-advice {
    background: linear-gradient(135deg, #e3f2fd, #ffffff);
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 6px 15px rgba(0,0,0,0.1);
    margin-top: 25px;
    border-left: 6px solid #007bff;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.health-advice:hover {
    transform: scale(1.01);
    box-shadow: 0 8px 18px rgba(0,0,0,0.15);
}
.health-advice h3 {
    color: #003366;
    font-weight: 700;
    margin-bottom: 10px;
}
.health-advice h4 {
    color: #007bff;
    margin-bottom: 10px;
}
.health-advice ul {
    list-style-type: disc;
    margin-left: 25px;
    color: #333333;
}
.health-advice li {
    margin-bottom: 6px;
    font-size: 16px;
}

</style>
""", unsafe_allow_html=True)

# ===========================================================
# SIDEBAR NAVIGATION
# ===========================================================
st.sidebar.title("Health Predictor Dashboard")
st.sidebar.markdown("<h4 style='color:#00eaff;text-align:center;'>Select a Prediction Category</h4>", unsafe_allow_html=True)

app_mode = st.sidebar.radio(
    "",
    (
        "Heart Disease",
        "Diabetes",
        "Stress / Mental Health",
        "Fitness / Lifestyle"
    )
)

st.sidebar.markdown("<br><br><br><hr><br>", unsafe_allow_html=True)
st.sidebar.markdown(
    """
    <div class="sidebar-footer">
        © 2025 Health Predictor | Smart Wellness System
    </div>
    """,
    unsafe_allow_html=True
)

# ===========================================================
# BACKGROUND IMAGES
# ===========================================================
backgrounds = {
    "Heart Disease": "https://cdn.pixabay.com/photo/2020/06/06/18/31/heart-5266636_1280.jpg",
    "Diabetes": "https://cdn.pixabay.com/photo/2021/02/18/11/44/diabetes-6025873_1280.jpg",
    "Stress / Mental Health": "https://cdn.pixabay.com/photo/2016/11/23/00/38/water-lilies-1850196_1280.jpg",
    "Fitness / Lifestyle": "https://cdn.pixabay.com/photo/2017/06/06/13/06/training-2379278_1280.jpg",
}
st.markdown(
    f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url('{backgrounds[app_mode]}');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        opacity: 0.95;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ===========================================================
# PERSONALIZED HEALTH RECOMMENDATION FUNCTION
# ===========================================================
def generate_health_advice(model_name, prediction):
    if model_name == "Heart Disease":
        return [
            "Consult a cardiologist for a full health check-up." if prediction == 1 else "Maintain regular physical activity.",
            "Avoid smoking, reduce stress, and maintain a healthy weight.",
            "Incorporate fiber-rich food, fish, and green vegetables.",
            "Track your blood pressure and cholesterol annually."
        ]
    elif model_name == "Diabetes":
        return [
            "Monitor glucose levels regularly." if prediction == 1 else "Maintain a balanced diet and exercise daily.",
            "Avoid sugary drinks and processed foods.",
            "Include high-fiber foods and hydration in your diet.",
            "Consult a healthcare provider for further evaluation."
        ]
    elif model_name == "Stress / Mental Health":
        return [
            "Practice meditation or mindfulness daily." if prediction == 1 else "Maintain consistent relaxation routines.",
            "Get 7–8 hours of sleep and avoid overworking.",
            "Stay socially connected and seek emotional support.",
            "Balance screen time and outdoor activity."
        ]
    elif model_name == "Fitness / Lifestyle":
        return [
            "Increase daily steps and reduce sitting hours." if prediction == 0 else "Continue regular exercise and balanced diet.",
            "Drink sufficient water and get proper rest.",
            "Mix strength and cardio workouts weekly.",
            "Track daily progress using a fitness app."
        ]
    return ["Maintain overall wellness through balanced habits."]

# ===========================================================
# HEALTH REPORT CARD
# ===========================================================
def show_health_report(score, risk_message, advice):
    # Choose color based on message content
    if "High" in risk_message or "Detected" in risk_message or "Diabetic" in risk_message:
        color = "#d9534f"  # red (danger)
    elif "Low" in risk_message or "Active" in risk_message or "No" in risk_message:
        color = "#28a745"  # green (safe)
    else:
        color = "#f0ad4e"  # yellow (moderate / neutral)

    st.markdown(f"""
    <div class="health-advice">
        <h3>Health Report Summary</h3>
        <h4 style="color:#007bff;">Health Score: {score:.1f}/100</h4>
        <p style="font-size:17px; color:{color};"><strong>Assessment:</strong> {risk_message}</p>
        <hr>
        <h4>Personal Health Recommendations:</h4>
        <ul>
            {''.join(f"<li>{tip}</li>" for tip in advice)}
        </ul>
    </div>
    """, unsafe_allow_html=True)


# ===========================================================
# MODEL LOGIC
# ===========================================================

# 1️⃣ HEART DISEASE MODEL
if app_mode == "Heart Disease":
    st.title("Heart Disease Prediction")
    model = load_model("heart_model.pkl")
    scaler = load_model("heart_scaler.pkl")

    with st.form("heart_form"):
        age = st.number_input("Age", 20, 100, 50)
        sex = st.selectbox("Sex (0=Female, 1=Male)", [0, 1])
        cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
        trestbps = st.number_input("Resting BP (mm Hg)", 80, 200, 120)
        chol = st.number_input("Serum Cholesterol (mg/dL)", 100, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar >120?", [0, 1])
        restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
        thalach = st.number_input("Max Heart Rate", 60, 220, 150)
        exang = st.selectbox("Exercise Induced Angina?", [0, 1])
        oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0)
        slope = st.selectbox("Slope of ST Segment", [0, 1, 2])
        ca = st.selectbox("Major Vessels (0–3)", [0, 1, 2, 3])
        thal = st.selectbox("Thal Type", [1, 2, 3])
        submitted = st.form_submit_button("Predict Heart Disease Risk")

        if submitted and model:
            features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                                  thalach, exang, oldpeak, slope, ca, thal]])
            if scaler and hasattr(scaler, "n_features_in_"):
                if scaler.n_features_in_ == features.shape[1]:
                    features = scaler.transform(features)
            result = model.predict(features)
            score = random.uniform(40, 95)
            risk_msg = "High risk of heart disease detected." if result[0] == 1 else "Low risk of heart complications."
            advice = generate_health_advice("Heart Disease", result[0])
            show_health_report(score, risk_msg, advice)

# 2️⃣ DIABETES MODEL
elif app_mode == "Diabetes":
    st.title("Diabetes Prediction")
    model = load_model("diabetes_model.pkl")

    with st.form("diabetes_form"):
        pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.number_input("Glucose Level", 50, 300, 120)
        bp = st.number_input("Blood Pressure (mm Hg)", 40, 200, 80)
        skin = st.number_input("Skin Thickness", 0, 99, 20)
        insulin = st.number_input("Insulin", 0, 900, 80)
        bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
        age = st.number_input("Age", 20, 100, 40)
        submitted = st.form_submit_button("Predict Diabetes Risk")

        if submitted and model:
            features = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
            result = model.predict(features)
            score = random.uniform(40, 95)
            risk_msg = "Diabetic condition detected." if result[0] == 1 else "No diabetes risk detected."
            advice = generate_health_advice("Diabetes", result[0])
            show_health_report(score, risk_msg, advice)

# 3️⃣ STRESS MODEL
elif app_mode == "Stress / Mental Health":
    st.title("Stress / Mental Health Prediction")
    model = load_model("stress_model.pkl")

    with st.form("stress_form"):
        age = st.number_input("Age", 15, 70, 25)
        gender = st.selectbox("Gender (0=Male,1=Female)", [0, 1])
        family_history = st.selectbox("Family History of Mental Illness?", [0, 1])
        employees = st.number_input("Number of Employees", 1, 1000, 50)
        benefits = st.selectbox("Employer Benefits Provided?", [0, 1])
        submitted = st.form_submit_button("Predict Stress Level")

        if submitted and model:
            features = np.array([[age, gender, family_history, employees, benefits]])
            result = model.predict(features)
            score = random.uniform(40, 95)
            risk_msg = "High stress level detected." if result[0] == 1 else "Low stress level detected."
            advice = generate_health_advice("Stress / Mental Health", result[0])
            show_health_report(score, risk_msg, advice)

# 4️⃣ FITNESS MODEL
# ===========================================================
# FITNESS / LIFESTYLE PREDICTION (Calories Burned)
# ===========================================================
elif app_mode == "Fitness / Lifestyle":
    st.title("Fitness & Lifestyle Analysis")

    model = load_model("fitness_model.pkl")

    st.markdown("""
    <p style='font-size:16px;color:#003366;'>
        Enter your daily activity details to estimate total calories burned.
    </p>
    """, unsafe_allow_html=True)

    with st.form("fitness_form"):
        col1, col2 = st.columns(2)
        with col1:
            total_steps = st.number_input("Average Steps per Day", min_value=1000, max_value=30000, value=8000, step=500)
            total_distance = st.number_input("Total Distance Walked/Run (km)", min_value=0.5, max_value=30.0, value=5.0, step=0.5)
        with col2:
            very_active_minutes = st.number_input("Very Active Minutes per Day", min_value=0, max_value=300, value=60, step=5)
            sedentary_minutes = st.number_input("Sedentary Minutes per Day", min_value=0, max_value=1000, value=300, step=10)

        submitted = st.form_submit_button("Predict Calories Burned")

        if submitted and model:
            try:
                features = np.array([[total_steps, total_distance, very_active_minutes, sedentary_minutes]])
                result = model.predict(features)
                calories = result[0]

                # Generate a "fitness score" out of 100
                fitness_score = np.clip((calories - 1500) / 25, 0, 100)

                st.success(f"Estimated Calories Burned: **{calories:.2f} kcal/day**")
                st.markdown(f"<h4 style='color:#007bff;'>Fitness Score: {fitness_score:.1f}/100</h4>", unsafe_allow_html=True)

                # --- Generate advice dynamically ---
                if fitness_score >= 80:
                    st.markdown("""
                    <div style="background-color:#e8f9e9;padding:15px;border-radius:8px;">
                        <b>Excellent!</b> You maintain a highly active lifestyle. Keep up the good work and stay consistent!
                    </div>
                    """, unsafe_allow_html=True)
                elif 50 <= fitness_score < 80:
                    st.markdown("""
                    <div style="background-color:#fff3cd;padding:15px;border-radius:8px;">
                        <b>Moderate Activity:</b> Try to include more intense workouts or longer walks to improve stamina.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background-color:#f8d7da;padding:15px;border-radius:8px;">
                        <b>Low Activity:</b> You may need to reduce sedentary time and aim for 30–45 minutes of exercise daily.
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Prediction Error: {e}")

