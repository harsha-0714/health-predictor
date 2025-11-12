import streamlit as st
import pickle
import numpy as np
import os
import pandas as pd
from fpdf import FPDF

# ============================================================
#  UNIVERSAL MODEL LOADER
# ============================================================
def load_model(model_name):
    paths = [
        f"models/{model_name}",
        f"/content/models/{model_name}",
        f"/content/health-predictor/models/{model_name}",
        f"/mount/src/health-predictor/models/{model_name}",
        model_name
    ]
    for path in paths:
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    model = pickle.load(f)
                return model
            except Exception:
                return None
    return None


# ============================================================
#  FALLBACK RULE-BASED LOGIC FOR ALL MODELS
# ============================================================
def fallback_predict(model_name, features):
    try:
        if model_name == "heart_model.pkl":
            age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal = features[0]
            risk, score, recs = 0, 95, []
            if age > 55 or trestbps > 140 or chol > 240 or thalach < 120:
                risk, score = 1, 60
                recs = [
                    "Consult a cardiologist for regular heart checkups.",
                    "Reduce sodium and processed foods in diet.",
                    "Engage in light cardio exercise like brisk walking.",
                ]
            else:
                recs = ["Maintain a healthy lifestyle and balanced diet."]
            assessment = "High risk of heart disease" if risk else "Low risk of heart disease"
            return risk, assessment, score, recs

        elif model_name == "diabetes_model.pkl":
            pregnancies, glucose, bp, skin, insulin, bmi, dpf, age = features[0]
            risk, score, recs = 0, 95, []
            if glucose > 140 or bmi > 30 or insulin > 200 or bp > 130:
                risk, score = 1, 60
                recs = [
                    "Monitor glucose levels regularly.",
                    "Adopt a low-carb diet.",
                    "Exercise daily to regulate blood sugar."
                ]
            else:
                recs = ["Maintain your fitness and healthy eating habits."]
            assessment = "High diabetes risk" if risk else "Low diabetes risk"
            return risk, assessment, score, recs

        elif model_name == "stress_model.pkl":
            age, gender, family_history, employees, benefits = features[0]
            risk, score, recs = 0, 90, []
            if family_history == 1 or benefits == 0 or employees > 200:
                risk, score = 1, 65
                recs = [
                    "Engage in mindfulness or yoga.",
                    "Take regular breaks during work.",
                    "Stay socially connected and active."
                ]
            else:
                recs = ["Maintain regular rest and stress-free work habits."]
            assessment = "High stress level" if risk else "Low stress level"
            return risk, assessment, score, recs

        elif model_name == "fitness_model.pkl":
            steps, calories, sleep, sedentary = features[0]
            risk, score, recs = 0, 90, []
            if steps < 5000 or sleep < 6 or sedentary > 600 or calories < 1500:
                risk, score = 1, 65
                recs = [
                    "Increase daily steps or workouts.",
                    "Sleep at least 7–8 hours regularly.",
                    "Reduce screen time and move every hour."
                ]
            else:
                recs = ["Excellent fitness level — keep it up!"]
            assessment = "Low fitness level" if risk else "Excellent fitness routine"
            return risk, assessment, score, recs

        return 0, "No data found", 70, ["Please check your input values."]
    except Exception:
        return 0, "Prediction error", 70, ["Please check your input values."]


# ============================================================
#  DISPLAY HEALTH REPORT (Enhanced Dark Font Styling)
# ============================================================
def show_health_report(category, score, assessment, recs, color_class):
    st.markdown(f"""
    <div style='background-color:#f9f9f9; padding:25px; border-radius:12px;
                margin-top:25px; box-shadow:0px 4px 10px rgba(0,0,0,0.25);
                border-left:6px solid {color_class}; font-family:"Segoe UI", sans-serif;'>
        <h3 style='color:#1a1a1a; font-weight:700;'>{category} Health Report</h3>
        <p style='color:#1a1a1a; font-size:16px;'>
            <strong>Health Score:</strong>
            <span style='color:{color_class}; font-weight:800;'>{score:.1f}/100</span>
        </p>
        <p style='color:#1a1a1a; font-size:16px;'>
            <strong>Assessment:</strong>
            <span style='color:{color_class}; font-weight:600;'>{assessment}</span>
        </p>
        <hr style='border-top: 1px solid #ccc;'>
        <h4 style='color:#000; font-weight:700;'>Recommended Steps:</h4>
        <ul style='color:#1c1c1c; font-size:15px; line-height:1.6;'>
            {''.join(f"<li>{r}</li>" for r in recs)}
        </ul>
    </div>
    """, unsafe_allow_html=True)
    return score


# ============================================================
#  COMBINED SCORE
# ============================================================
def show_combined_score(scores):
    avg_score = np.mean(scores)
    color = "#27ae60" if avg_score >= 75 else "#e67e22" if avg_score >= 50 else "#c0392b"
    status = (
        "Excellent Health – Keep up the great work!" if avg_score >= 75 else
        "Moderate Health – Some improvements needed." if avg_score >= 50 else
        "Health Risk – Consult a doctor and adopt healthy changes."
    )
    st.markdown(f"""
    <div style='background-color:#eaf6ff; padding:25px; border-radius:10px;
                margin-top:40px; box-shadow:0 4px 10px rgba(0,0,0,0.15);'>
        <h2 style='color:#003366;'>Overall Health Summary</h2>
        <h3 style='color:{color};'>Combined Health Score: {avg_score:.1f}/100</h3>
        <p style='color:#1a1a1a;'><strong>Status:</strong> {status}</p>
    </div>
    """, unsafe_allow_html=True)
    return avg_score


# ============================================================
# PDF GENERATION
# ============================================================
def generate_pdf(user_data, combined_score):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Health Predictor Report", ln=True, align="C")
    pdf.set_font("Arial", size=12)
    for key, val in user_data.items():
        pdf.cell(200, 8, f"{key}: {val}", ln=True)
    pdf.cell(200, 10, f"Combined Health Score: {combined_score:.1f}/100", ln=True)
    pdf.output("health_report.pdf")
    with open("health_report.pdf", "rb") as f:
        st.download_button("Download Health Report (PDF)", f, "health_report.pdf")


# ============================================================
# MAIN APP CONFIG
# ============================================================
st.set_page_config(page_title="AI-Driven Health Predictor", layout="wide")
st.sidebar.title("Smart Health Dashboard")

app_mode = st.sidebar.radio(
    "Choose a Prediction Module:",
    ("Heart Disease", "Diabetes", "Stress / Mental Health", "Fitness / Lifestyle")
)

st.title("Intelligent Hybrid Health Predictor Dashboard")

scores, user_details = [], {}

# ============================================================
# HEART MODULE
# ============================================================
if app_mode == "Heart Disease":
    model = load_model("heart_model.pkl")
    scaler = load_model("heart_scaler.pkl")
    st.subheader("Heart Health Analysis")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 20, 100, 45)
        trestbps = st.number_input("Resting BP", 80, 200, 120)
        chol = st.number_input("Cholesterol", 100, 600, 200)
        thalach = st.number_input("Max Heart Rate", 60, 220, 150)
        fbs = st.selectbox("Fasting Blood Sugar >120?", [0, 1])
    with col2:
        sex = st.selectbox("Sex (0=Female,1=Male)", [0, 1])
        cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
        exang = st.selectbox("Exercise Induced Angina?", [0, 1])
        oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0)
        slope = st.selectbox("ST Slope (0–2)", [0, 1, 2])
        ca = st.selectbox("Major Vessels (0–3)", [0, 1, 2, 3])
        thal = st.selectbox("Thal (1–3)", [1, 2, 3])

    if st.button("Generate Heart Report"):
        features = np.array([[age, sex, cp, trestbps, chol, fbs, 0, thalach, exang, oldpeak, slope, ca, thal]])
        if model:
            try:
                if scaler:
                    features = scaler.transform(features)
                result = model.predict(features)
                risk = int(result[0])
                assessment = "High risk of heart disease" if risk else "Low risk of heart disease"
                score = np.random.uniform(55, 65) if risk else np.random.uniform(85, 95)
                recs = ["Consult a doctor immediately." if risk else "Maintain regular exercise."]
            except Exception:
                risk, assessment, score, recs = fallback_predict("heart_model.pkl", features)
        else:
            risk, assessment, score, recs = fallback_predict("heart_model.pkl", features)
        color = "#c0392b" if risk else "#27ae60"
        score = show_health_report("Heart", score, assessment, recs, color)
        scores.append(score)
        user_details["Heart"] = assessment

# ============================================================
# DIABETES MODULE
# ============================================================
elif app_mode == "Diabetes":
    model = load_model("diabetes_model.pkl")
    st.subheader("Diabetes Risk Evaluation")
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose", 50, 300, 120)
    bp = st.number_input("Blood Pressure", 40, 200, 80)
    skin = st.number_input("Skin Thickness", 0, 99, 20)
    insulin = st.number_input("Insulin", 0, 900, 80)
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.number_input("Age", 20, 100, 40)

    if st.button("Generate Diabetes Report"):
        features = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        if model:
            try:
                result = model.predict(features)
                risk = int(result[0])
                assessment = "High diabetes risk" if risk else "Low diabetes risk"
                score = np.random.uniform(55, 65) if risk else np.random.uniform(85, 95)
                recs = ["Monitor glucose levels." if risk else "Maintain your fitness."]
            except Exception:
                risk, assessment, score, recs = fallback_predict("diabetes_model.pkl", features)
        else:
            risk, assessment, score, recs = fallback_predict("diabetes_model.pkl", features)
        color = "#c0392b" if risk else "#27ae60"
        score = show_health_report("Diabetes", score, assessment, recs, color)
        scores.append(score)
        user_details["Diabetes"] = assessment

# ============================================================
# STRESS MODULE
# ============================================================
elif app_mode == "Stress / Mental Health":
    model = load_model("stress_model.pkl")
    st.subheader("Mental Health Assessment")
    age = st.number_input("Age", 15, 70, 25)
    gender = st.selectbox("Gender", [0, 1])
    family_history = st.selectbox("Family History", [0, 1])
    employees = st.number_input("Number of Employees", 1, 1000, 50)
    benefits = st.selectbox("Employer Benefits", [0, 1])

    if st.button("Generate Stress Report"):
        features = np.array([[age, gender, family_history, employees, benefits]])
        risk, assessment, score, recs = fallback_predict("stress_model.pkl", features)
        color = "#c0392b" if risk else "#27ae60"
        score = show_health_report("Stress", score, assessment, recs, color)
        scores.append(score)
        user_details["Stress"] = assessment

# ============================================================
# FITNESS MODULE
# ============================================================
elif app_mode == "Fitness / Lifestyle":
    model = load_model("fitness_model.pkl")
    st.subheader("Fitness Evaluation")
    steps = st.number_input("Steps per Day", 0, 50000, 8000)
    calories = st.number_input("Calories Burned", 100, 6000, 2500)
    sleep = st.number_input("Sleep Duration (hours)", 2.0, 12.0, 7.0)
    sedentary = st.number_input("Sedentary Minutes", 0, 1000, 300)

    if st.button("Generate Fitness Report"):
        features = np.array([[steps, calories, sleep, sedentary]])
        risk, assessment, score, recs = fallback_predict("fitness_model.pkl", features)
        color = "#c0392b" if risk else "#27ae60"
        score = show_health_report("Fitness", score, assessment, recs, color)
        scores.append(score)
        user_details["Fitness"] = assessment

# ============================================================
# FINAL COMBINED REPORT
# ============================================================
if scores:
    combined_score = show_combined_score(scores)
    user_details["Combined Score"] = combined_score
    generate_pdf(user_details, combined_score)
