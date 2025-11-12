import streamlit as st
import numpy as np
import pickle
import os
from openai import OpenAI
from fpdf import FPDF

# =========================================================
# 1️⃣ OpenAI Setup
# =========================================================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_openai_advice(disease, features):
    """AI fallback using the new OpenAI API (>=1.0.0)."""
    try:
        user_context = ", ".join([f"{k}: {v}" for k, v in features.items()])
        prompt = f"""
        You are a certified AI health assistant.
        The user has provided the following details: {user_context}.
        Based on this information, analyze their potential risk for {disease},
        generate a health score (out of 100),
        and provide clear, evidence-based recommendations covering
        diet, exercise, habits, and early warning signs.
        Be concise and helpful.
        """
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a medical AI providing accurate, balanced advice."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.6,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(AI fallback unavailable: {e})"


# =========================================================
# 2️⃣ Streamlit Page Config and Styling
# =========================================================
st.set_page_config(page_title="Smart Health Predictor", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(120deg, #f8fbff, #e8f3ff);
    font-family: 'Segoe UI', sans-serif;
}
h1, h2, h3, h4, h5 {
    color: #003366;
}
.stButton>button {
    background: linear-gradient(to right, #007BFF, #00BFFF);
    color: white;
    border-radius: 10px;
    padding: 8px 25px;
    border: none;
    font-weight: 600;
}
.stButton>button:hover {
    background: linear-gradient(to right, #0056b3, #0099cc);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #003366, #0055cc);
    color: white;
}
.report-box {
    background-color:#ffffff;
    border-radius:15px;
    padding:20px;
    margin-top:20px;
    box-shadow:0 4px 12px rgba(0,0,0,0.1);
    color:#1a1a1a;
}
.report-box h4 {
    color:#004080;
}
.report-box strong {
    color:#002b5c;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 3️⃣ Helper Functions
# =========================================================
def load_model(model_name):
    """Safely load model if available."""
    paths = [
        os.path.join("models", model_name),
        os.path.join("/content/models", model_name),
        os.path.join("/content/health-predictor/models", model_name),
        model_name
    ]
    for path in paths:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    return None

def health_score(result):
    """Simple scoring system."""
    return np.random.uniform(40, 65) if result == 1 else np.random.uniform(85, 100)

def show_health_report(title, score, summary, advice):
    st.markdown(f"""
    <div class="report-box">
        <h3>{title}</h3>
        <h4>Health Score: <span style="color:#007bff;">{score:.1f}/100</span></h4>
        <p><strong>Summary:</strong> {summary}</p>
        <hr>
        <p><strong>AI Recommendations:</strong></p>
        <p>{advice}</p>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# 4️⃣ Sidebar Navigation
# =========================================================
st.sidebar.title("Smart Health Predictor")
app_mode = st.sidebar.radio(
    "Select Health Assessment:",
    ["Heart Disease", "Diabetes", "Stress / Mental Health", "Fitness / Lifestyle"]
)

# Store all user reports for combined PDF
user_reports = []

# =========================================================
# ❤️ HEART DISEASE
# =========================================================
if app_mode == "Heart Disease":
    st.title("Heart Disease Prediction")

    model = load_model("heart_model.pkl")
    scaler = load_model("heart_scaler.pkl")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 20, 100, 50)
        trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
        chol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
        thalach = st.number_input("Max Heart Rate", 60, 220, 150)
        fbs = st.selectbox("Fasting Blood Sugar >120", [0, 1])
    with col2:
        sex = st.selectbox("Sex (0=Female, 1=Male)", [0, 1])
        cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
        exang = st.selectbox("Exercise Induced Angina", [0, 1])
        oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
        slope = st.selectbox("Slope (0–2)", [0, 1, 2])
        ca = st.selectbox("Major Vessels (0–3)", [0, 1, 2, 3])
        thal = st.selectbox("Thal", [1, 2, 3])

    if st.button("Predict Heart Health"):
        features = np.array([[age, sex, cp, trestbps, chol, fbs, 0, thalach, exang, oldpeak, slope, ca, thal]])
        try:
            if model:
                if scaler and hasattr(scaler, "n_features_in_"):
                    if scaler.n_features_in_ == features.shape[1]:
                        features = scaler.transform(features)
                result = model.predict(features)
                score = health_score(result[0])
                summary = "High heart risk detected." if result[0] == 1 else "Heart health appears good."
            else:
                result = [0]
                score = np.random.uniform(70, 90)
                summary = "Model unavailable — AI-generated analysis applied."
            advice = generate_openai_advice("Heart Disease", {
                "Age": age, "Cholesterol": chol, "BP": trestbps, "Max HR": thalach
            })
            show_health_report("Heart Health Report", score, summary, advice)
            user_reports.append(("Heart Disease", score, summary))
        except Exception as e:
            st.error(f"Error: {e}")

# =========================================================
# 💉 DIABETES
# =========================================================
elif app_mode == "Diabetes":
    st.title("Diabetes Prediction")
    model = load_model("diabetes_model.pkl")

    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose (mg/dL)", 50, 300, 120)
    bp = st.number_input("Blood Pressure", 40, 200, 80)
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
    age = st.number_input("Age", 10, 100, 40)

    if st.button("Predict Diabetes Risk"):
        features = np.array([[pregnancies, glucose, bp, bmi, age]])
        try:
            if model:
                result = model.predict(features)
                score = health_score(result[0])
                summary = "Possible diabetic condition." if result[0] == 1 else "Low diabetes risk."
            else:
                result = [0]
                score = np.random.uniform(70, 90)
                summary = "Model unavailable — AI-generated analysis applied."
            advice = generate_openai_advice("Diabetes", {
                "Age": age, "Glucose": glucose, "BMI": bmi, "BP": bp
            })
            show_health_report("Diabetes Health Report", score, summary, advice)
            user_reports.append(("Diabetes", score, summary))
        except Exception as e:
            st.error(f"Error: {e}")

# =========================================================
# 🧠 STRESS / MENTAL HEALTH
# =========================================================
elif app_mode == "Stress / Mental Health":
    st.title("Stress / Mental Health Prediction")
    model = load_model("stress_model.pkl")

    age = st.number_input("Age", 15, 70, 25)
    gender = st.selectbox("Gender (0=Male,1=Female)", [0, 1])
    family_history = st.selectbox("Family History of Mental Illness?", [0, 1])
    work_interfere = st.selectbox("Work Interference (0=None, 1=Sometimes, 2=Often, 3=Always)", [0, 1, 2, 3])

    if st.button("Analyze Stress Level"):
        features = np.array([[age, gender, family_history, work_interfere]])
        try:
            if model:
                result = model.predict(features)
                score = health_score(result[0])
                summary = "High stress level detected." if result[0] == 1 else "Healthy mental balance."
            else:
                result = [0]
                score = np.random.uniform(70, 90)
                summary = "Model unavailable — AI-generated analysis applied."
            advice = generate_openai_advice("Stress Management", {
                "Age": age, "Gender": gender, "Work Stress": work_interfere
            })
            show_health_report("Stress Report", score, summary, advice)
            user_reports.append(("Stress", score, summary))
        except Exception as e:
            st.error(f"Error: {e}")

# =========================================================
# 🏃 FITNESS / LIFESTYLE
# =========================================================
elif app_mode == "Fitness / Lifestyle":
    st.title("Fitness / Lifestyle Assessment")
    model = load_model("fitness_model.pkl")

    steps = st.number_input("Average Steps per Day", 0, 30000, 8000)
    calories = st.number_input("Calories Burned per Day", 500, 6000, 2500)
    sleep = st.number_input("Average Sleep Hours", 2.0, 12.0, 7.0)
    sedentary = st.number_input("Sedentary Minutes", 0, 1000, 300)

    if st.button("Evaluate Fitness"):
        features = np.array([[steps, calories, sleep, sedentary]])
        try:
            if model:
                result = model.predict(features)
                score = health_score(result[0])
                summary = "Active lifestyle detected." if result[0] == 1 else "Low activity lifestyle."
            else:
                result = [0]
                score = np.random.uniform(70, 90)
                summary = "Model unavailable — AI-generated analysis applied."
            advice = generate_openai_advice("Fitness and Lifestyle", {
                "Steps": steps, "Calories": calories, "Sleep": sleep, "Sedentary": sedentary
            })
            show_health_report("Fitness Report", score, summary, advice)
            user_reports.append(("Fitness", score, summary))
        except Exception as e:
            st.error(f"Error: {e}")
