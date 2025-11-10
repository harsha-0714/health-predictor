import streamlit as st
import pickle
import numpy as np
from PIL import Image
import base64

# --- Utility: Set background image dynamically ---
def set_background(image_path):
    with open(image_path, "rb") as img_file:
        b64_image = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{b64_image}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# --- Load models safely ---
def load_model(path):
    try:
        with open(path, "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error(f"⚠️ Model not found: {path}")
        return None

heart_model = load_model("models/heart_model.pkl")
diabetes_model = load_model("models/diabetes_model.pkl")
stress_model = load_model("models/stress_model.pkl")
fitness_model = load_model("models/fitness_model.pkl")

# --- Sidebar navigation ---
st.sidebar.title("🩺 Health Predictor Dashboard")
page = st.sidebar.radio("Select a Prediction Model:",
                        ("❤️ Heart Disease",
                         "💉 Diabetes Risk",
                         "🧠 Stress & Mental Health",
                         "🏃 Fitness Level Analysis"))

st.sidebar.markdown("---")
st.sidebar.info("Developed by Harsha — Smart Health Prediction Suite")

# --- HEART DISEASE MODEL ---
# --- HEART DISEASE MODEL ---
if page == "❤️ Heart Disease":
    set_background("assets/heart_bg.jpg")
    st.title("❤️ Heart Disease Prediction")

    age = st.slider("Age", 20, 100, 45)
    chol = st.number_input("Cholesterol (mg/dL)", 100, 400, 200)
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    thalach = st.number_input("Maximum Heart Rate Achieved", 60, 210, 150)
    oldpeak = st.number_input("ST Depression Induced by Exercise", 0.0, 6.0, 1.0)

    if st.button("🔍 Predict Heart Condition"):
        features = np.array([[age, chol, trestbps, thalach, oldpeak]])
        try:
            scaler = pickle.load(open("models/heart_scaler.pkl", "rb"))
            features_scaled = scaler.transform(features)
        except:
            features_scaled = features  # fallback if scaler missing

        result = heart_model.predict(features_scaled)[0]

        if result == 1:
            st.error("⚠️ High risk of heart disease detected. Consult a cardiologist soon.")
        else:
            st.success("✅ Low Risk of Heart Disease detected. Keep maintaining your lifestyle!")

# --- DIABETES MODEL ---
elif page == "💉 Diabetes Risk":
    set_background("assets/diabetes_bg.jpg")

    st.title("💉 Diabetes Prediction")
    st.markdown("Fill in the following to predict **diabetes risk**:")

    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose Level", 50, 250, 120)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", 50, 150, 80)
    bmi = st.number_input("Body Mass Index (BMI)", 10.0, 60.0, 25.0)
    age = st.slider("Age", 10, 100, 30)

    if st.button("🔍 Predict Diabetes Risk"):
        features = np.array([[pregnancies, glucose, blood_pressure, bmi, age]])
        result = diabetes_model.predict(features)[0]
        if result == 1:
            st.error("⚠️ High risk of diabetes. Please consult an endocrinologist.")
        else:
            st.success("✅ Low diabetes risk. Continue a balanced lifestyle.")
        st.markdown("---")
        st.info("💡 **Prevention:** Reduce sugar intake, stay active, and maintain a healthy weight.")

# --- STRESS MODEL ---
elif page == "🧠 Stress & Mental Health":
    set_background("assets/stress_bg.jpg")

    st.title("🧠 Stress / Mental Health Prediction")
    st.markdown("Assess your **mental stress level** based on lifestyle factors:")

    age = st.slider("Age", 15, 70, 25)
    work_hours = st.slider("Daily Work Hours", 0, 16, 8)
    sleep_hours = st.slider("Sleep Hours per Day", 0, 12, 7)
    family_support = st.radio("Family Support Available?", ["Yes", "No"])
    exercise = st.radio("Do you exercise regularly?", ["Yes", "No"])

    features = np.array([[1 if family_support == "Yes" else 0,
                          1 if exercise == "Yes" else 0,
                          work_hours,
                          sleep_hours,
                          age]])

    if st.button("🔍 Predict Stress Level"):
        result = stress_model.predict([features[0]])[0]
        if result == 1:
            st.error("⚠️ High stress detected! Take rest and talk to a friend or counselor.")
        else:
            st.success("✅ Stress levels appear healthy. Keep it up!")
        st.markdown("---")
        st.info("💡 **Tip:** Practice meditation, take screen breaks, and avoid burnout.")

# --- FITNESS MODEL ---
elif page == "🏃 Fitness Level Analysis":
    set_background("assets/fitness_bg.jpg")

    st.title("🏃 Fitness & Activity Level")
    st.markdown("Enter your daily activity details:")

    total_steps = st.number_input("Total Steps", 0, 30000, 5000)
    total_distance = st.number_input("Total Distance (km)", 0.0, 20.0, 5.0)
    very_active_minutes = st.number_input("Very Active Minutes", 0, 200, 60)
    sedentary_minutes = st.number_input("Sedentary Minutes", 0, 1000, 300)

    if st.button("🔍 Analyze Fitness"):
        features = np.array([[total_steps, total_distance, very_active_minutes, sedentary_minutes]])
        prediction = fitness_model.predict(features)[0]
        st.success(f"🏅 Estimated Fitness Score: **{round(prediction, 2)}**")

        st.markdown("---")
        if prediction < 1200:
            st.error("⚠️ Low activity detected. Increase your daily movement and hydration.")
        elif prediction < 2000:
            st.warning("⚠️ Moderate activity level. Add more consistent workouts.")
        else:
            st.success("✅ Great activity level! Keep your body moving and stay hydrated.")
        st.markdown("---")
        st.info("💡 **Suggestion:** Aim for 8,000+ steps/day and limit sedentary hours.")

# --- FOOTER ---
st.markdown("---")
st.caption("© 2025 Smart Health Predictor | Designed by Harsha 💻 | Powered by Streamlit")
