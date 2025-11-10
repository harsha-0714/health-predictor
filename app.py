import streamlit as st
import pickle
import numpy as np
import os

# =====================================================
# SMART MODEL LOADER – searches multiple paths safely
# =====================================================

def load_model(model_name):
    possible_paths = [
        f"models/{model_name}",
        f"/content/models/{model_name}",
        f"/content/health-predictor/models/{model_name}",
        f"/mount/src/health-predictor/models/{model_name}"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    model = pickle.load(f)
                    st.info(f"✅ Loaded {model_name} from: {path}")
                    return model
            except Exception as e:
                st.error(f"Error reading {model_name} from {path}: {e}")
                return None
    st.error(f"❌ Model not found anywhere: {model_name}")
    return None


# =====================================================
# BACKGROUND SETTER
# =====================================================
def set_background(image_url):
    style = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background: url("{image_url}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    [data-testid="stHeader"], [data-testid="stToolbar"] {{
        background: rgba(0,0,0,0);
    }}
    .main-card {{
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0px 4px 25px rgba(0,0,0,0.2);
        margin-top: 2rem;
    }}
    </style>
    """
    st.markdown(style, unsafe_allow_html=True)


# =====================================================
# SIDEBAR NAVIGATION
# =====================================================
st.sidebar.title("Health Prediction Dashboard")
app_mode = st.sidebar.radio(
    "Select Model",
    ["Heart Disease", "Diabetes", "Stress", "Fitness"]
)

# =====================================================
# HEART DISEASE PREDICTION
# =====================================================
if app_mode == "Heart Disease":
    set_background("https://images.unsplash.com/photo-1588776814546-ec7e5f9a9c75?auto=format&fit=crop&w=1920&q=80")
    st.title("Heart Disease Risk Prediction")
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)

    model = load_model("heart_model.pkl")
    scaler = load_model("heart_scaler.pkl")

    st.subheader("Enter Your Details:")
    age = st.number_input("Age", 20, 100)
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 90, 200)
    chol = st.number_input("Cholesterol (mg/dL)", 100, 600)
    thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220)
    oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 6.0, step=0.1)

    features = np.array([[age, 1, 3, trestbps, chol, 0, 1, thalach, 0, oldpeak, 1, 0, 2]])
    if st.button("Predict Heart Risk"):
        if model and scaler:
            scaled = scaler.transform(features)
            prediction = model.predict(scaled)[0]
            prob = model.predict_proba(scaled)[0][1] * 100
            if prediction == 1:
                st.error(f"High Risk of Heart Disease ({prob:.2f}% probability)")
            else:
                st.success(f"Low Risk of Heart Disease ({prob:.2f}% probability)")

    st.markdown("</div>", unsafe_allow_html=True)


# =====================================================
# DIABETES PREDICTION
# =====================================================
elif app_mode == "Diabetes":
    set_background("https://images.unsplash.com/photo-1582719478250-c89cae4dc85b?auto=format&fit=crop&w=1920&q=80")
    st.title("Diabetes Prediction")
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)

    model = load_model("diabetes_model.pkl")

    st.subheader("Enter Your Details:")
    pregnancies = st.number_input("Pregnancies", 0, 20)
    glucose = st.number_input("Glucose Level", 0, 300)
    blood_pressure = st.number_input("Blood Pressure", 0, 200)
    skin_thickness = st.number_input("Skin Thickness", 0, 100)
    insulin = st.number_input("Insulin Level", 0, 900)
    bmi = st.number_input("BMI", 0.0, 70.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
    age = st.number_input("Age", 10, 100)

    features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    if st.button("Predict Diabetes"):
        if model:
            prediction = model.predict(features)[0]
            if prediction == 1:
                st.error("High Risk of Diabetes detected.")
            else:
                st.success("Low Risk of Diabetes.")

    st.markdown("</div>", unsafe_allow_html=True)


# =====================================================
# STRESS LEVEL PREDICTION
# =====================================================
elif app_mode == "Stress":
    set_background("https://images.unsplash.com/photo-1506126613408-eca07ce68773?auto=format&fit=crop&w=1920&q=80")
    st.title("Stress Level Prediction")
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)

    model = load_model("stress_model.pkl")

    st.subheader("Enter Your Details:")
    age = st.number_input("Age", 15, 80)
    gender = st.selectbox("Gender", ["Male", "Female"])
    family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])
    employees = st.slider("Number of Employees in Company", 1, 500, 50)
    benefits = st.selectbox("Mental Health Benefits at Work?", ["Yes", "No"])

    gender_val = 1 if gender == "Male" else 0
    family_val = 1 if family_history == "Yes" else 0
    benefit_val = 1 if benefits == "Yes" else 0

    features = np.array([[age, gender_val, family_val, employees, benefit_val]])
    if st.button("Predict Stress Level"):
        if model:
            prediction = model.predict(features)[0]
            if prediction == 1:
                st.error("High Stress Level detected. Consider relaxation and self-care.")
            else:
                st.success("Normal Stress Level detected.")

    st.markdown("</div>", unsafe_allow_html=True)


# =====================================================
# FITNESS ACTIVITY PREDICTION
# =====================================================
elif app_mode == "Fitness":
    set_background("https://images.unsplash.com/photo-1594737625785-c0f9f0eeb1e5?auto=format&fit=crop&w=1920&q=80")
    st.title("Fitness Activity Prediction")
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)

    model = load_model("fitness_model.pkl")

    st.subheader("Enter Your Daily Activity Data:")
    steps = st.number_input("Daily Steps", 0, 50000, 5000)
    calories = st.number_input("Calories Burned", 0, 10000, 2000)
    sleep_hours = st.number_input("Sleep Hours", 0.0, 24.0, 7.0)
    heart_rate = st.number_input("Average Heart Rate", 40, 180, 75)

    features = np.array([[steps, calories, sleep_hours, heart_rate]])
    if st.button("Predict Fitness Level"):
        if model:
            prediction = model.predict(features)[0]
            if prediction == 1:
                st.success("Good Fitness Level — keep it up!")
            else:
                st.warning("Low Fitness Level — consider improving daily activity.")

    st.markdown("</div>", unsafe_allow_html=True)
