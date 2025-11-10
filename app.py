import streamlit as st
import pickle
import numpy as np
import os

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
                model = pickle.load(f)
                st.sidebar.success(f"Loaded: {model_name}")
                return model
    st.sidebar.error(f"Model not found: {model_name}")
    return None


# ===========================================================
# PAGE CONFIGURATION
# ===========================================================
st.set_page_config(page_title="Health Predictor Dashboard", layout="wide")

# ===========================================================
# SIDEBAR NAVIGATION
# ===========================================================
st.sidebar.title("Health Prediction Dashboard")
app_mode = st.sidebar.radio(
    "Choose a Prediction Category:",
    ("Heart Disease", "Diabetes", "Stress / Mental Health", "Fitness / Lifestyle")
)

# ===========================================================
# STYLING (Modern Theme + Gradients)
# ===========================================================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(120deg, #eaf4ff, #ffffff);
    background-attachment: fixed;
}

[data-testid="stHeader"] {
    background: linear-gradient(to right, #007bff, #0099ff);
    color: white;
    font-size: 22px;
    padding: 12px;
    font-weight: bold;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
}

[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #f8f9fa, #ffffff);
    border-right: 2px solid #0099ff;
}

.stButton>button {
    background: linear-gradient(to right, #00aaff, #007bff);
    color: white;
    font-weight: 600;
    border-radius: 10px;
    padding: 10px 25px;
    font-size: 16px;
    border: none;
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
</style>
""", unsafe_allow_html=True)

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
# FUNCTION: Display Health Score and Recommendations
# ===========================================================
def show_health_report(score, risk_message, suggestions):
    st.markdown(f"""
    <div style="background-color: #ffffffcc; padding: 20px; border-radius: 12px; margin-top: 20px; box-shadow: 0 4px 10px rgba(0,0,0,0.1);">
        <h3 style="color:#003366;">Health Report Summary</h3>
        <h4 style="color:#007bff;">Health Score: {score:.1f}/100</h4>
        <p><strong>Assessment:</strong> {risk_message}</p>
        <hr>
        <h4 style="color:#003366;">Recommendations:</h4>
        <ul>
            {''.join(f"<li>{s}</li>" for s in suggestions)}
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ===========================================================
# HEART DISEASE PREDICTION
# ===========================================================
if app_mode == "Heart Disease":
    st.title("Heart Disease Prediction")
    model = load_model("heart_model.pkl")
    scaler = load_model("heart_scaler.pkl")

    with st.form("heart_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 20, 100, 50, placeholder="E.g. 45")
            trestbps = st.number_input("Resting BP (mm Hg)", 80, 200, 120, placeholder="E.g. 130")
            chol = st.number_input("Serum Cholesterol (mg/dL)", 100, 600, 200, placeholder="E.g. 220")
            thalach = st.number_input("Max Heart Rate", 60, 220, 150, placeholder="E.g. 160")
            fbs = st.selectbox("Fasting Blood Sugar >120 mg/dL?", [0, 1])
            restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
        with col2:
            sex = st.selectbox("Sex (0=Female, 1=Male)", [0, 1])
            cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
            exang = st.selectbox("Exercise Induced Angina?", [0, 1])
            oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
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
            if result[0] == 1:
                score = np.random.uniform(30, 60)
                show_health_report(
                    score,
                    "High risk of heart complications detected.",
                    [
                        "Consult a cardiologist immediately.",
                        "Avoid smoking and maintain a healthy diet.",
                        "Engage in light physical activity daily."
                    ]
                )
            else:
                score = np.random.uniform(85, 95)
                show_health_report(
                    score,
                    "Low risk of heart disease.",
                    [
                        "Continue regular exercise and balanced diet.",
                        "Keep your blood pressure under control.",
                        "Avoid excessive stress and monitor yearly."
                    ]
                )

# ===========================================================
# DIABETES PREDICTION
# ===========================================================
elif app_mode == "Diabetes":
    st.title("Diabetes Prediction")
    model = load_model("diabetes_model.pkl")

    with st.form("diabetes_form"):
        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.number_input("Pregnancies", 0, 20, 1)
            glucose = st.number_input("Glucose Level (mg/dL)", 50, 300, 120)
            bp = st.number_input("Blood Pressure (mm Hg)", 40, 200, 80)
            skin = st.number_input("Skin Thickness (mm)", 0, 99, 20)
        with col2:
            insulin = st.number_input("Insulin (mu U/ml)", 0, 900, 80)
            bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
            dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
            age = st.number_input("Age", 20, 100, 40)
        submitted = st.form_submit_button("Predict Diabetes")

        if submitted and model:
            features = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
            result = model.predict(features)
            if result[0] == 1:
                score = np.random.uniform(40, 65)
                show_health_report(
                    score,
                    "Diabetic condition detected.",
                    [
                        "Monitor blood glucose regularly.",
                        "Reduce sugar intake and increase fiber-rich foods.",
                        "Consult your doctor for lifestyle adjustments."
                    ]
                )
            else:
                score = np.random.uniform(85, 95)
                show_health_report(
                    score,
                    "Non-diabetic, healthy status.",
                    [
                        "Maintain healthy body weight.",
                        "Stay hydrated and avoid excessive sugar.",
                        "Continue physical exercise regularly."
                    ]
                )

# ===========================================================
# STRESS / MENTAL HEALTH PREDICTION
# ===========================================================
elif app_mode == "Stress / Mental Health":
    st.title("Stress / Mental Health Prediction")
    model = load_model("stress_model.pkl")

    with st.form("stress_form"):
        age = st.number_input("Age", 15, 70, 25)
        gender = st.selectbox("Gender (0=Male,1=Female)", [0, 1])
        family_history = st.selectbox("Family History of Mental Illness?", [0, 1])
        employees = st.number_input("Employees in Company", 1, 1000, 50)
        benefits = st.selectbox("Employer Benefits Provided?", [0, 1])
        submitted = st.form_submit_button("Predict Stress Level")

        if submitted and model:
            features = np.array([[age, gender, family_history, employees, benefits]])
            result = model.predict(features)
            if result[0] == 1:
                score = np.random.uniform(45, 65)
                show_health_report(
                    score,
                    "High stress level detected.",
                    [
                        "Engage in relaxation activities like meditation.",
                        "Take regular breaks during work.",
                        "Seek support from family or a counselor."
                    ]
                )
            else:
                score = np.random.uniform(80, 95)
                show_health_report(
                    score,
                    "Low stress level.",
                    [
                        "Maintain balance between work and leisure.",
                        "Continue mindfulness practices.",
                        "Sleep adequately and stay socially active."
                    ]
                )

# ===========================================================
# FITNESS / LIFESTYLE PREDICTION
# ===========================================================
elif app_mode == "Fitness / Lifestyle":
    st.title("Fitness / Lifestyle Prediction")
    model = load_model("fitness_model.pkl")

    with st.form("fitness_form"):
        steps = st.number_input("Average Steps per Day", 0, 50000, 8000)
        calories = st.number_input("Calories Burned per Day", 100, 6000, 2500)
        sleep = st.number_input("Sleep Duration (hours)", 2.0, 12.0, 7.0)
        sedentary = st.number_input("Sedentary Minutes", 0, 1000, 300)
        submitted = st.form_submit_button("Predict Fitness Level")

        if submitted and model:
            features = np.array([[steps, calories, sleep, sedentary]])
            result = model.predict(features)
            if result[0] == 1:
                score = np.random.uniform(85, 95)
                show_health_report(
                    score,
                    "Active lifestyle detected.",
                    [
                        "Keep up consistent workouts.",
                        "Stay hydrated and maintain protein intake.",
                        "Ensure adequate rest between physical activities."
                    ]
                )
            else:
                score = np.random.uniform(40, 65)
                show_health_report(
                    score,
                    "Sedentary lifestyle detected.",
                    [
                        "Incorporate daily walks or stretching exercises.",
                        "Reduce screen time and increase outdoor activity.",
                        "Follow a balanced diet to improve metabolism."
                    ]
                )
