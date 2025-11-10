import streamlit as st
import pickle
import numpy as np
import os

# ===========================================================
# ✅ UNIVERSAL MODEL LOADER
# ===========================================================
def load_model(model_name):
    path = os.path.join("models", model_name)
    if os.path.exists(path):
        with open(path, "rb") as f:
            model = pickle.load(f)
            st.success(f"✅ Loaded {model_name}")
            return model
    else:
        st.error(f"❌ Model not found: {path}")
        return None


# ===========================================================
# ✅ PAGE CONFIGURATION
# ===========================================================
st.set_page_config(page_title="Health Predictor Dashboard", layout="wide")

# ===========================================================
# ✅ SIDEBAR NAVIGATION
# ===========================================================
st.sidebar.title("🧭 Navigation")
app_mode = st.sidebar.radio(
    "Choose a Health Prediction Model:",
    ("Heart Disease", "Diabetes", "Stress / Mental Health", "Fitness / Lifestyle")
)

# ===========================================================
# ✅ BACKGROUND IMAGES BASED ON MODEL
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
    }}
    [data-testid="stHeader"] {{
        background: rgba(0,0,0,0);
    }}
    [data-testid="stSidebar"] {{
        background: rgba(255,255,255,0.9);
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ===========================================================
# ✅ HEART DISEASE PREDICTION
# ===========================================================
# ===========================================================
# ✅ HEART DISEASE PREDICTION (13 features)
# ===========================================================
if app_mode == "Heart Disease":
    st.title("💖 Heart Disease Prediction")

    model = load_model("heart_model.pkl")
    scaler = load_model("heart_scaler.pkl")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 20, 100, 50)
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
        chol = st.number_input("Serum Cholesterol (mg/dL)", 100, 600, 200)
        thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
        fbs = st.selectbox("Fasting Blood Sugar >120 (1=True,0=False)", [0, 1])
        restecg = st.selectbox("Resting ECG Results (0–2)", [0, 1, 2])

    with col2:
        sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
        cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
        exang = st.selectbox("Exercise Induced Angina (1=True,0=False)", [0, 1])
        oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
        slope = st.selectbox("Slope of ST Segment (0–2)", [0, 1, 2])
        ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
        thal = st.selectbox("Thal (1=Normal, 2=Fixed Defect, 3=Reversible Defect)", [1, 2, 3])

    if st.button("🔍 Predict Heart Disease Risk"):
        if model is not None:
            try:
                # 13 total features (match training)
                features = np.array([[age, sex, cp, trestbps, chol, fbs,
                                      restecg, thalach, exang, oldpeak,
                                      slope, ca, thal]])

                # Safely apply scaler if compatible
                if scaler is not None:
                    try:
                        if scaler.n_features_in_ == features.shape[1]:
                            features = scaler.transform(features)
                        else:
                            st.warning(f"⚠️ Skipping scaling — expected {scaler.n_features_in_} features, got {features.shape[1]}")
                    except Exception as e:
                        st.warning(f"⚠️ Skipping scaling: {e}")

                # Predict
                result = model.predict(features)
                risk = "High Risk" if result[0] == 1 else "Low Risk"
                st.subheader(f"🩺 Prediction: {risk}")

                if result[0] == 1:
                    st.error("⚠️ High Risk: Please consult a cardiologist.")
                else:
                    st.success("✅ Low Risk: Maintain your healthy habits!")
            except Exception as e:
                st.error(f"Prediction Error: {e}")
        else:
            st.error("❌ Model not loaded correctly.")

# ===========================================================
# ✅ DIABETES PREDICTION
# ===========================================================
elif app_mode == "Diabetes":
    st.title("💉 Diabetes Prediction")

    model = load_model("diabetes_model.pkl")

    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.number_input("Glucose Level", 50, 300, 120)
        bp = st.number_input("Blood Pressure", 40, 200, 80)
        skin = st.number_input("Skin Thickness", 0, 99, 20)
    with col2:
        insulin = st.number_input("Insulin", 0, 900, 80)
        bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
        age = st.number_input("Age", 20, 100, 40)

    if st.button("🔍 Predict Diabetes"):
        if model is not None:
            features = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
            result = model.predict(features)
            risk = "Diabetic" if result[0] == 1 else "Non-Diabetic"
            st.subheader(f"🩺 Prediction: {risk}")
            if result[0] == 1:
                st.error("⚠️ Diabetic: Follow medical guidance and diet control.")
            else:
                st.success("✅ Non-Diabetic: Maintain healthy habits.")

# ===========================================================
# ✅ STRESS / MENTAL HEALTH PREDICTION
# ===========================================================
elif app_mode == "Stress / Mental Health":
    st.title("🧠 Stress / Mental Health Prediction")

    model = load_model("stress_model.pkl")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 15, 70, 25)
        gender = st.selectbox("Gender (0=Male,1=Female)", [0, 1])
        family_history = st.selectbox("Family History of Mental Illness (0/1)", [0, 1])
    with col2:
        employees = st.number_input("No. of Employees (approx.)", 1, 1000, 50)
        benefits = st.selectbox("Employer Benefits Provided (0/1)", [0, 1])

    if st.button("🔍 Predict Stress Level"):
        if model is not None:
            features = np.array([[age, gender, family_history, employees, benefits]])
            result = model.predict(features)
            risk = "High Stress Risk" if result[0] == 1 else "Low Stress Risk"
            st.subheader(f"🩺 Prediction: {risk}")
            if result[0] == 1:
                st.error("⚠️ High Stress Risk: Prioritize mental wellness and seek support.")
            else:
                st.success("✅ Low Stress Risk: Keep maintaining emotional balance.")

# ===========================================================
# ✅ FITNESS / LIFESTYLE PREDICTION
# ===========================================================
elif app_mode == "Fitness / Lifestyle":
    st.title("🏃 Fitness / Lifestyle Prediction")

    model = load_model("fitness_model.pkl")

    col1, col2 = st.columns(2)
    with col1:
        steps = st.number_input("Avg. Steps per Day", 0, 50000, 8000)
        calories = st.number_input("Avg. Calories Burned", 100, 6000, 2500)
    with col2:
        sleep = st.number_input("Sleep Duration (hours)", 2.0, 12.0, 7.0)
        sedentary = st.number_input("Sedentary Minutes", 0, 1000, 300)

    if st.button("🔍 Predict Fitness Level"):
        if model is not None:
            features = np.array([[steps, calories, sleep, sedentary]])
            result = model.predict(features)
            fitness = "Active Lifestyle" if result[0] == 1 else "Sedentary Lifestyle"
            st.subheader(f"🩺 Prediction: {fitness}")
            if result[0] == 1:
                st.success("✅ Active Lifestyle: Keep up the great habits!")
            else:
                st.error("⚠️ Sedentary: Increase daily movement and reduce screen time.")
