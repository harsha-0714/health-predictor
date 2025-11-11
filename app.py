import streamlit as st
import pickle
import numpy as np
import os

# ============================================================
# ✅ UNIVERSAL MODEL LOADER — Auto-detects all paths
# ============================================================
def load_model(model_name):
    possible_paths = [
        f"models/{model_name}",
        f"/mount/src/health-predictor/models/{model_name}",
        f"/content/health-predictor/models/{model_name}",
        f"/content/models/{model_name}",
        model_name
    ]

    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    model = pickle.load(f)
                return model
            except Exception as e:
                st.error(f"Error loading {model_name}: {e}")
                return None

    st.error(f"Model file not found: {model_name}")
    return None


# ============================================================
# ✅ STREAMLIT PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Health Predictor Dashboard", layout="wide")

# ============================================================
# 🎨 CUSTOM CSS — Clean, modern, responsive
# ============================================================
st.markdown("""
<style>
/* App background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(120deg, #eaf6ff, #ffffff);
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #002b5c, #0056b3);
    color: white;
    border-right: 3px solid #00aaff;
}
[data-testid="stSidebar"] h1, h2, h3, h4 {
    color: white;
}

/* Radio buttons in sidebar */
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

/* Buttons */
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

/* Inputs */
input[type=number], select {
    border-radius: 6px;
    border: 1.5px solid #0099ff;
    padding: 6px 10px;
    margin-bottom: 10px;
    font-size: 15px;
}

/* Health report box */
.health-report {
    background-color: #ffffffcc;
    padding: 20px;
    border-radius: 12px;
    margin-top: 20px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
}
.assessment-high { color: #c0392b; font-weight: 700; }
.assessment-low { color: #27ae60; font-weight: 700; }

/* Footer */
.sidebar-footer {
    position: absolute;
    bottom: 10px;
    width: 100%;
    text-align: center;
    color: #ddd;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 🧭 SIDEBAR NAVIGATION
# ============================================================
st.sidebar.title("Health Predictor Dashboard")
st.sidebar.markdown("<h4 style='color:#00eaff;text-align:center;'>Select a Prediction Category</h4>", unsafe_allow_html=True)

app_mode = st.sidebar.radio(
    "",
    ("Heart Disease", "Diabetes", "Stress / Mental Health", "Fitness / Lifestyle")
)

st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
st.sidebar.markdown(
    """
    <div class="sidebar-footer">
        © 2025 Health Predictor | Smart Wellness System
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# 🌅 BACKGROUND IMAGES FOR EACH MODEL
# ============================================================
backgrounds = {
    "Heart Disease": "https://cdn.pixabay.com/photo/2020/06/06/18/31/heart-5266636_1280.jpg",
    "Diabetes": "https://cdn.pixabay.com/photo/2021/02/18/11/44/diabetes-6025873_1280.jpg",
    "Stress / Mental Health": "https://cdn.pixabay.com/photo/2016/11/23/00/38/water-lilies-1850196_1280.jpg",
    "Fitness / Lifestyle": "https://cdn.pixabay.com/photo/2017/06/06/13/06/training-2379278_1280.jpg"
}

st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url('{backgrounds[app_mode]}');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 📊 HEALTH REPORT CARD
# ============================================================
def show_health_report(score, assessment, color_class, recommendations):
    st.markdown(f"""
    <div class="health-report">
        <h3 style="color:#003366;">Health Report Summary</h3>
        <h4 style="color:#007bff;">Health Score: {score:.1f}/100</h4>
        <p class="{color_class}"><strong>Assessment:</strong> {assessment}</p>
        <hr>
        <h4 style="color:#003366;">Recommendations:</h4>
        <ul>{''.join(f"<li>{r}</li>" for r in recommendations)}</ul>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# ❤️ HEART DISEASE MODEL
# ============================================================
if app_mode == "Heart Disease":
    st.title("Heart Disease Prediction")
    model = load_model("heart_model.pkl")
    scaler = load_model("heart_scaler.pkl")

    with st.form("heart_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 20, 100, 50, placeholder="Enter your age")
            trestbps = st.number_input("Resting BP (mm Hg)", 80, 200, 120)
            chol = st.number_input("Serum Cholesterol (mg/dL)", 100, 600, 200)
            thalach = st.number_input("Max Heart Rate", 60, 220, 150)
            fbs = st.selectbox("Fasting Blood Sugar >120 mg/dL?", [0, 1])
            restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
        with col2:
            sex = st.selectbox("Sex (0=Female, 1=Male)", [0, 1])
            cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
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
            if result[0] == 1:
                show_health_report(55, "High risk of heart disease detected.",
                                   "assessment-high",
                                   ["Consult a cardiologist immediately.",
                                    "Follow a heart-healthy diet.",
                                    "Engage in daily light exercise."])
            else:
                show_health_report(92, "Low risk of heart disease detected.",
                                   "assessment-low",
                                   ["Maintain regular exercise.",
                                    "Monitor blood pressure yearly.",
                                    "Avoid smoking and stress."])

# ============================================================
# 💉 DIABETES MODEL
# ============================================================
elif app_mode == "Diabetes":
    st.title("Diabetes Prediction")
    model = load_model("diabetes_model.pkl")

    with st.form("diabetes_form"):
        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.number_input("Pregnancies", 0, 20, 1)
            glucose = st.number_input("Glucose (mg/dL)", 50, 300, 120)
            bp = st.number_input("Blood Pressure (mm Hg)", 40, 200, 80)
            skin = st.number_input("Skin Thickness (mm)", 0, 99, 20)
        with col2:
            insulin = st.number_input("Insulin (mu U/ml)", 0, 900, 80)
            bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
            dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
            age = st.number_input("Age", 20, 100, 40)
        submitted = st.form_submit_button("Predict Diabetes Risk")

        if submitted and model:
            features = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
            result = model.predict(features)
            if result[0] == 1:
                show_health_report(60, "High diabetes risk detected.", "assessment-high",
                                   ["Reduce sugar and carb intake.",
                                    "Exercise 30 minutes daily.",
                                    "Consult your doctor for tests."])
            else:
                show_health_report(90, "Low diabetes risk detected.", "assessment-low",
                                   ["Maintain a healthy diet.",
                                    "Stay hydrated and active.",
                                    "Check glucose annually."])

# ============================================================
# 🧠 STRESS / MENTAL HEALTH MODEL
# ============================================================
elif app_mode == "Stress / Mental Health":
    st.title("Stress / Mental Health Prediction")
    model = load_model("stress_model.pkl")

    with st.form("stress_form"):
        age = st.number_input("Age", 15, 70, 25)
        gender = st.selectbox("Gender (0=Male,1=Female)", [0, 1])
        family_history = st.selectbox("Family History of Mental Illness?", [0, 1])
        employees = st.number_input("No. of Employees", 1, 1000, 50)
        benefits = st.selectbox("Employer Benefits Provided?", [0, 1])
        submitted = st.form_submit_button("Predict Stress Level")

        if submitted and model:
            features = np.array([[age, gender, family_history, employees, benefits]])
            result = model.predict(features)
            if result[0] == 1:
                show_health_report(58, "High stress level detected.", "assessment-high",
                                   ["Practice relaxation techniques.",
                                    "Take breaks and rest regularly.",
                                    "Seek counseling if needed."])
            else:
                show_health_report(88, "Low stress level detected.", "assessment-low",
                                   ["Continue mindfulness activities.",
                                    "Maintain work-life balance.",
                                    "Stay socially connected."])

# ============================================================
# 🏃 FITNESS / LIFESTYLE MODEL
# ============================================================
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
                show_health_report(90, "Active lifestyle detected.", "assessment-low",
                                   ["Keep up regular workouts.",
                                    "Maintain hydration and rest.",
                                    "Track your daily progress."])
            else:
                show_health_report(50, "Sedentary lifestyle detected.", "assessment-high",
                                   ["Increase movement throughout the day.",
                                    "Add daily walks or stretching.",
                                    "Avoid sitting for long hours."])
