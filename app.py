import streamlit as st
import pickle
import numpy as np
from fpdf import FPDF
import datetime

# ----------------------------
# 🧩 LOAD MODELS
# ----------------------------
with open("models/heart_model.pkl", "rb") as f:
    heart_model = pickle.load(f)
with open("models/diabetes_model.pkl", "rb") as f:
    diabetes_model = pickle.load(f)
with open("models/stress_model.pkl", "rb") as f:
    stress_model = pickle.load(f)
with open("models/fitness_model.pkl", "rb") as f:
    fitness_model = pickle.load(f)

# ----------------------------
# 🌐 APP CONFIG
# ----------------------------
st.set_page_config(page_title="AI Health Insight Dashboard", page_icon="💊", layout="wide")
st.title("💊 AI Health Insight Dashboard")
st.write("An intelligent multi-model system that predicts **Heart Disease**, **Diabetes**, **Stress**, and **Fitness Levels** — and generates a personalized **Health Report**.")

st.sidebar.title("🔍 Choose a Module")
page = st.sidebar.radio("Select one:", ["Heart Disease", "Diabetes", "Stress Level", "Fitness Level", "Generate Health Report"])

# Store results for final report
if "results" not in st.session_state:
    st.session_state["results"] = {}

# ----------------------------
# ❤️ HEART DISEASE MODULE
# ----------------------------
if page == "Heart Disease":
    st.header("❤️ Heart Disease Prediction")
    age = st.number_input("Age", 10, 100)
    sex = st.selectbox("Gender", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type (0-3)", [0,1,2,3])
    trestbps = st.number_input("Resting Blood Pressure", 80, 200)
    chol = st.number_input("Cholesterol (mg/dl)", 100, 600)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0,1])
    restecg = st.selectbox("Resting ECG Results (0-2)", [0,1,2])
    thalach = st.number_input("Max Heart Rate Achieved", 60, 250)
    exang = st.selectbox("Exercise Induced Angina (1=Yes, 0=No)", [0,1])

    if st.button("Predict Heart Risk"):
        gender = 1 if sex == "Male" else 0
        features = np.array([[age, gender, cp, trestbps, chol, fbs, restecg, thalach, exang]])
        result = heart_model.predict(features)[0]
        if result == 1:
            st.session_state["results"]["Heart Disease"] = "⚠️ High Risk"
            st.error("⚠️ High Risk of Heart Disease detected.")
        else:
            st.session_state["results"]["Heart Disease"] = "✅ Low Risk"
            st.success("✅ Low Risk of Heart Disease detected.")

# ----------------------------
# 💉 DIABETES MODULE
# ----------------------------
elif page == "Diabetes":
    st.header("💉 Diabetes Prediction")
    pregnancies = st.number_input("Pregnancies", 0, 20)
    glucose = st.number_input("Glucose Level", 0, 300)
    blood_pressure = st.number_input("Blood Pressure", 0, 200)
    skin_thickness = st.number_input("Skin Thickness", 0, 99)
    insulin = st.number_input("Insulin Level", 0, 900)
    bmi = st.number_input("BMI", 0.0, 70.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5)
    age = st.number_input("Age", 10, 100)

    if st.button("Predict Diabetes Risk"):
        features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        result = diabetes_model.predict(features)[0]
        if result == 1:
            st.session_state["results"]["Diabetes"] = "⚠️ High Risk"
            st.error("⚠️ High Risk of Diabetes detected.")
        else:
            st.session_state["results"]["Diabetes"] = "✅ Low Risk"
            st.success("✅ Low Risk of Diabetes detected.")

# ----------------------------
# 🧠 STRESS MODULE
# ----------------------------
elif page == "Stress Level":
    st.header("🧠 Stress Level Prediction")
    sleep_hours = st.slider("Average Sleep Hours per Night", 0, 12, 6)
    work_hours = st.slider("Work or Study Hours per Day", 0, 16, 8)
    social = st.slider("Social Activity Hours per Day", 0, 8, 2)
    exercise = st.slider("Physical Exercise Hours per Day", 0, 6, 1)
    workload = st.selectbox("Current Workload Level", ["Low", "Medium", "High"])

    workload_val = {"Low": 0, "Medium": 1, "High": 2}[workload]
    features = np.array([[sleep_hours, work_hours, social, exercise, workload_val]])

    if st.button("Predict Stress Level"):
        result = stress_model.predict(features)[0]
        if result == 2:
            st.session_state["results"]["Stress Level"] = "⚠️ High"
            st.error("⚠️ High Stress Level detected.")
        elif result == 1:
            st.session_state["results"]["Stress Level"] = "⚠️ Moderate"
            st.warning("⚠️ Moderate Stress detected.")
        else:
            st.session_state["results"]["Stress Level"] = "✅ Low"
            st.success("✅ Low Stress Level detected.")

# ----------------------------
# 🏃‍♂️ FITNESS MODULE
# ----------------------------
elif page == "Fitness Level":
    st.header("🏃‍♂️ Fitness Level Prediction")
    steps = st.number_input("Total Steps per Day", 0, 50000)
    distance = st.number_input("Total Distance (km)", 0.0, 50.0)
    active_minutes = st.number_input("Active Minutes per Day", 0, 300)
    calories = st.number_input("Calories Burned per Day", 0, 8000)

    if st.button("Analyze Fitness Level"):
        features = np.array([[steps, distance, active_minutes, calories]])
        prediction = fitness_model.predict(features)[0]
        st.session_state["results"]["Fitness"] = f"{prediction:.2f} Calories Burned"
        st.info(f"Predicted Calories Burned: **{prediction:.2f}**")

        if prediction < 2000:
            st.warning("⚠️ Low activity detected. Increase movement.")
        elif 2000 <= prediction < 3000:
            st.success("✅ Moderate fitness activity level.")
        else:
            st.success("🏅 Excellent activity and calorie output!")

# ----------------------------
# 🩺 GENERAL HEALTH REPORT (PDF)
# ----------------------------
elif page == "Generate Health Report":
    st.header("🩺 Generate Personalized Health Report")

    if st.session_state["results"]:
        st.write("### Summary of Predictions:")
        for key, val in st.session_state["results"].items():
            st.write(f"- **{key}:** {val}")

        if st.button("📄 Download Health Report as PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 18)
            pdf.cell(200, 10, txt="AI Health Insight Report", ln=True, align="C")

            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt=f"Date: {datetime.date.today()}", ln=True)
            pdf.cell(200, 10, txt="----------------------------------------", ln=True)

            pdf.set_font("Arial", "B", 14)
            pdf.cell(200, 10, txt="Prediction Summary:", ln=True)
            pdf.set_font("Arial", size=12)
            for key, val in st.session_state["results"].items():
                pdf.cell(200, 8, txt=f"{key}: {val}", ln=True)

            pdf.ln(8)
            pdf.set_font("Arial", "B", 14)
            pdf.cell(200, 10, txt="Preventive Recommendations:", ln=True)
            pdf.set_font("Arial", size=12)
            recommendations = [
                "❤️ Exercise 30–45 mins/day.",
                "🥗 Eat a balanced, low-sugar diet.",
                "😴 Sleep 7–8 hours per night.",
                "💧 Drink 2–3L water daily.",
                "🧘 Practice stress management.",
                "📅 Schedule regular checkups."
            ]
            for r in recommendations:
                pdf.cell(200, 8, txt=f"- {r}", ln=True)

            pdf.ln(10)
            pdf.set_font("Arial", "B", 14)
            pdf.cell(200, 10, txt="Future Possibilities:", ln=True)
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 8, txt=(
                "Future integration with IoT wearables can enable real-time monitoring "
                "of heart rate, blood glucose, and stress levels. Predictive analytics "
                "using AI could provide early warnings for potential health conditions."
            ))

            pdf.output("Health_Report.pdf")
            with open("Health_Report.pdf", "rb") as file:
                st.download_button("⬇️ Download Report", file, file_name="Health_Report.pdf", mime="application/pdf")

    else:
        st.warning("⚠️ Please run predictions in other modules first.")
