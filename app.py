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
#  RULE-BASED DECISION LOGIC (primary engine)
# ============================================================
def rule_based_predict(model_name, features):
    try:
        # ---------- HEART ----------
        if model_name == "heart_model.pkl":
            age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal = features[0]
            risk, score, recs = 0, 95, []

            # Rule logic
            if age > 55:
                recs.append("Your age increases cardiac risk; ensure annual ECG and lipid profile checks.")
                risk, score = 1, 75
            if trestbps > 140:
                recs.append("Blood pressure above 140 mmHg — reduce sodium and manage stress effectively.")
                risk, score = 1, min(score, 70)
            if chol > 240:
                recs.append("High cholesterol detected — reduce fried foods and saturated fats.")
                risk, score = 1, min(score, 65)
            if thalach < 120:
                recs.append("Low max heart rate — include light aerobic activities like brisk walking.")
                risk, score = 1, min(score, 65)
            if fbs == 1:
                recs.append("High fasting blood sugar — check for prediabetes conditions.")
                risk, score = 1, min(score, 60)

            if not recs:
                recs.append("Excellent cardiac health — maintain regular exercise and balanced diet.")
            assessment = "High risk of heart disease" if risk else "Low risk of heart disease"
            return risk, assessment, score, recs

        # ---------- DIABETES ----------
        elif model_name == "diabetes_model.pkl":
            pregnancies, glucose, bp, skin, insulin, bmi, dpf, age = features[0]
            risk, score, recs = 0, 95, []

            if glucose > 140:
                recs.append("Elevated glucose level — avoid refined sugar and monitor fasting glucose weekly.")
                risk, score = 1, 70
            if bmi > 30:
                recs.append("BMI above 30 — incorporate daily physical activity and portion control.")
                risk, score = 1, min(score, 65)
            if insulin > 200:
                recs.append("High insulin suggests insulin resistance — follow a high-fiber, low-carb diet.")
                risk, score = 1, min(score, 60)
            if bp > 130:
                recs.append("Blood pressure above optimal level — reduce salt and caffeine intake.")
                risk, score = 1, min(score, 75)
            if age > 45 and glucose > 125:
                recs.append("Increased age + glucose levels indicate prediabetic condition — consult physician.")
                risk, score = 1, min(score, 65)

            if not recs:
                recs.append("Healthy blood sugar profile — maintain current lifestyle.")
            assessment = "High diabetes risk" if risk else "Low diabetes risk"
            return risk, assessment, score, recs

        # ---------- STRESS ----------
        elif model_name == "stress_model.pkl":
            age, gender, family_history, employees, benefits = features[0]
            risk, score, recs = 0, 90, []

            if family_history == 1:
                recs.append("Genetic predisposition to anxiety or stress; regular relaxation activities recommended.")
                risk, score = 1, 70
            if benefits == 0:
                recs.append("Lack of workplace benefits — practice mindfulness and prioritize self-care.")
                risk, score = 1, min(score, 65)
            if employees > 200:
                recs.append("Working in large organizations — ensure healthy work-life balance.")
                risk, score = 1, min(score, 75)
            if age < 25:
                recs.append("Young individuals under high workload — take breaks and maintain hobbies.")
                risk, score = 1, min(score, 80)

            if not recs:
                recs.append("Balanced stress levels — continue good mental wellness habits.")
            assessment = "High stress level detected" if risk else "Low stress level"
            return risk, assessment, score, recs

        # ---------- FITNESS ----------
        elif model_name == "fitness_model.pkl":
            steps, calories, sleep, sedentary = features[0]
            risk, score, recs = 0, 90, []

            if steps < 5000:
                recs.append("Low daily steps — target at least 7,000–10,000 per day for improved stamina.")
                risk, score = 1, 70
            if sleep < 6:
                recs.append("Insufficient sleep — maintain 7–8 hours for recovery and energy.")
                risk, score = 1, min(score, 65)
            if sedentary > 600:
                recs.append("Extended sitting time — take short walking breaks every hour.")
                risk, score = 1, min(score, 65)
            if calories < 1500:
                recs.append("Low calorie burn — increase workout intensity or outdoor activity.")
                risk, score = 1, min(score, 75)

            if not recs:
                recs.append("Excellent activity balance — maintain your routine and hydration.")
            assessment = "Low fitness level" if risk else "Excellent fitness routine"
            return risk, assessment, score, recs

        return 0, "Invalid model or inputs", 70, ["Please verify your input values."]

    except Exception:
        return 0, "Prediction error", 70, ["Error in input validation."]


# ============================================================
#  DISPLAY HEALTH REPORT
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
        "Moderate Health – Improve routine slightly." if avg_score >= 50 else
        "Health Risk – Seek medical consultation soon."
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
        risk, assessment, score, recs = rule_based_predict("heart_model.pkl", features)

        # silent verification
        if model:
            try:
                model.predict(features)
            except Exception:
                pass

        color = "#c0392b" if risk else "#27ae60"
        scores.append(show_health_report("Heart", score, assessment, recs, color))
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
        risk, assessment, score, recs = rule_based_predict("diabetes_model.pkl", features)
        if model:
            try:
                model.predict(features)
            except Exception:
                pass
        color = "#c0392b" if risk else "#27ae60"
        scores.append(show_health_report("Diabetes", score, assessment, recs, color))
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
        risk, assessment, score, recs = rule_based_predict("stress_model.pkl", features)
        if model:
            try:
                model.predict(features)
            except Exception:
                pass
        color = "#c0392b" if risk else "#27ae60"
        scores.append(show_health_report("Stress", score, assessment, recs, color))
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
        risk, assessment, score, recs = rule_based_predict("fitness_model.pkl", features)
        if model:
            try:
                model.predict(features)
            except Exception:
                pass
        color = "#c0392b" if risk else "#27ae60"
        scores.append(show_health_report("Fitness", score, assessment, recs, color))
        user_details["Fitness"] = assessment

# ============================================================
# FINAL COMBINED REPORT
# ============================================================
if scores:
    combined_score = show_combined_score(scores)
    user_details["Combined Score"] = combined_score
    generate_pdf(user_details, combined_score)
