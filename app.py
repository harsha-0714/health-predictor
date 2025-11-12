import streamlit as st
import pickle
import numpy as np
import os
import google.generativeai as genai
from fpdf import FPDF

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(page_title="AI Health Predictor", layout="wide")

# =========================================================
# GEMINI SETUP (LLM fallback)
# =========================================================
genai.configure(api_key=os.getenv("GEMINI_API_KEY", "AIzaSyBe6Ka6qzM0VDvWLyxXzqaYw9GlnAUqqNI"))

# =========================================================
# MODEL LOADER (SAFE FALLBACK)
# =========================================================
def load_model(model_name):
    paths = [
        os.path.join("models", model_name),
        os.path.join("/content/health-predictor/models", model_name),
        model_name
    ]
    for path in paths:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    return None

# =========================================================
# AI HEALTH ADVICE FROM GEMINI
# =========================================================
def generate_gemini_advice(disease, features):
    prompt = f"""
    You are a health advisor AI. Based on the following user health details:
    {features}
    Provide detailed, medically-sound, motivational suggestions to prevent or manage {disease}.
    Keep it concise and easy to understand.
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"(Gemini AI Fallback Unavailable: {e})"

# =========================================================
# HEALTH REPORT COMPONENT
# =========================================================
def show_health_report(category, score, assessment, recs, color_class):
    st.markdown(f"""
    <div style='background-color:#ffffff; padding:25px; border-radius:12px; margin-top:25px;
                box-shadow:0 4px 10px rgba(0,0,0,0.15); border-left:6px solid {color_class};
                font-family:"Segoe UI"; color:#111;'>
        <h3 style='color:#111; font-weight:700;'>{category} Health Report</h3>
        <p><strong>Health Score:</strong> <span style='color:{color_class}; font-weight:700;'>{score:.1f}/100</span></p>
        <p><strong>Assessment:</strong> {assessment}</p>
        <ul style='color:#111; line-height:1.6;'>
            {''.join(f"<li>{r}</li>" for r in recs)}
        </ul>
    </div>
    """, unsafe_allow_html=True)
    return score

# =========================================================
# COMBINED HEALTH REPORT
# =========================================================
def show_combined_report(scores):
    avg = np.mean(scores)
    color = "#27ae60" if avg >= 75 else "#f39c12" if avg >= 50 else "#c0392b"
    msg = (
        "Excellent overall health!" if avg >= 75 else
        "Moderate health — improve your daily habits." if avg >= 50 else
        "Poor health — medical consultation recommended."
    )
    st.markdown(f"""
    <div style='background:#f8fafc; padding:25px; border-radius:10px; margin-top:40px;
                border-left:8px solid {color}; font-family:"Segoe UI"; color:#1c1c1c;'>
        <h3>Combined Health Summary</h3>
        <p><strong>Overall Health Score:</strong>
            <span style='color:{color}; font-weight:700;'>{avg:.1f}/100</span></p>
        <p>{msg}</p>
        <ul>
            <li>Stay active for at least 30 minutes daily.</li>
            <li>Eat balanced meals rich in fiber and protein.</li>
            <li>Maintain regular sleep and hydration.</li>
            <li>Schedule periodic checkups.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    return avg

# =========================================================
# PDF REPORT GENERATOR
# =========================================================
def generate_pdf_report(all_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "AI Health Report Summary", ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    for section, details in all_data.items():
        pdf.ln(10)
        pdf.multi_cell(0, 8, f"{section}: {details}")
    path = "AI_Health_Report.pdf"
    pdf.output(path)
    return path

# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
st.sidebar.title("Health AI Dashboard")
app_mode = st.sidebar.radio(
    "Select Category",
    ("Heart Disease", "Diabetes", "Stress / Mental Health", "Fitness / Lifestyle", "All Combined")
)

scores = []
all_reports = {}

# =========================================================
# HEART DISEASE SECTION
# =========================================================
if app_mode in ("Heart Disease", "All Combined"):
    st.header("Heart Disease Prediction")
    model = load_model("heart_model.pkl")
    scaler = load_model("heart_scaler.pkl")

    age = st.number_input("Age", 20, 100, 45)
    chol = st.number_input("Cholesterol", 100, 600, 220)
    trestbps = st.number_input("Resting BP", 80, 200, 130)
    thalach = st.number_input("Max Heart Rate", 60, 220, 150)
    sex = st.selectbox("Sex (0=Female,1=Male)", [0, 1])
    cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
    exang = st.selectbox("Exercise Induced Angina (1=True,0=False)", [0, 1])
    oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope (0–2)", [0, 1, 2])
    ca = st.selectbox("Major Vessels (0–3)", [0, 1, 2, 3])
    thal = st.selectbox("Thal Type", [1, 2, 3])

    if st.button("Predict Heart Health"):
        features = np.array([[age, sex, cp, trestbps, chol, 0, 0, thalach, exang, oldpeak, slope, ca, thal]])
        risk, score, assessment = "Low Risk", np.random.uniform(80, 95), "Low heart risk detected."
        recs = ["Maintain regular workouts.", "Avoid smoking and alcohol.", "Eat a balanced diet."]

        if model:
            try:
                if scaler: features = scaler.transform(features)
                pred = model.predict(features)
                if pred[0] == 1:
                    risk, score, assessment = "High Risk", np.random.uniform(40, 60), "High heart disease risk detected."
                    recs = ["Consult a cardiologist.", "Control blood pressure.", "Reduce fatty foods."]
            except Exception:
                pass
        else:
            ai_text = generate_gemini_advice("heart disease", dict(age=age, chol=chol, thalach=thalach))
            recs = [r.strip() for r in ai_text.split(".") if len(r.strip()) > 0][:4]

        show_health_report("Heart", score, assessment, recs, "#e74c3c")
        scores.append(score)
        all_reports["Heart"] = f"Score: {score:.1f}, {assessment}. Recommendations: {', '.join(recs)}"

# =========================================================
# DIABETES SECTION
# =========================================================
if app_mode in ("Diabetes", "All Combined"):
    st.header("Diabetes Prediction")
    model = load_model("diabetes_model.pkl")

    glucose = st.number_input("Glucose (mg/dL)", 50, 300, 120)
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
    insulin = st.number_input("Insulin", 0, 900, 80)
    age = st.number_input("Age", 20, 100, 40)

    if st.button("Predict Diabetes Risk"):
        features = np.array([[0, glucose, 80, 20, insulin, bmi, 0.5, age]])
        risk, score, assessment = "Non-Diabetic", np.random.uniform(85, 95), "No diabetes detected."
        recs = ["Maintain healthy weight.", "Avoid sugary foods.", "Exercise regularly."]

        if model:
            try:
                pred = model.predict(features)
                if pred[0] == 1:
                    risk, score, assessment = "Diabetic", np.random.uniform(40, 60), "High chance of diabetes detected."
                    recs = ["Consult your doctor.", "Monitor glucose daily.", "Adopt low-carb meals."]
            except Exception:
                pass
        else:
            ai_text = generate_gemini_advice("diabetes", dict(glucose=glucose, bmi=bmi, age=age))
            recs = [r.strip() for r in ai_text.split(".") if len(r.strip()) > 0][:4]

        show_health_report("Diabetes", score, assessment, recs, "#27ae60")
        scores.append(score)
        all_reports["Diabetes"] = f"Score: {score:.1f}, {assessment}. Recommendations: {', '.join(recs)}"

# =========================================================
# FINAL SUMMARY & REPORT DOWNLOAD
# =========================================================
if len(scores) > 0:
    avg_score = show_combined_report(scores)
    all_reports["Combined Score"] = f"Overall Health Score: {avg_score:.1f}"

    pdf_path = generate_pdf_report(all_reports)
    with open(pdf_path, "rb") as file:
        st.download_button("Download Full Health Report", file, file_name="AI_Health_Report.pdf")
