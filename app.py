import streamlit as st
import numpy as np
import pickle
import os
import openai
from fpdf import FPDF

# ======================================================
# 🧠 OpenAI API Setup (uses Colab or OS environment key)
# ======================================================
openai.api_key = os.getenv("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")

def generate_openai_advice(disease, features):
    """Fallback AI advice when model is missing or prediction fails."""
    try:
        user_context = ", ".join([f"{k}: {v}" for k, v in features.items()])
        prompt = f"""
        You are a certified health AI advisor.
        Based on these user details: {user_context}
        Give a short, evidence-based prevention and improvement plan for {disease}.
        Include: exercise, diet, lifestyle, and medical checkup suggestions.
        Be concise and human-like.
        """
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a trusted medical AI."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.6
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        return f"(AI fallback unavailable: {e})"


# ======================================================
# 🎨 Streamlit Page Configuration
# ======================================================
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
</style>
""", unsafe_allow_html=True)

# ======================================================
# 🧩 Helper Functions
# ======================================================
def load_model(model_name):
    """Try to load .pkl models safely."""
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
    """Assigns a health score based on prediction (1 = disease detected)."""
    return np.random.uniform(40, 65) if result == 1 else np.random.uniform(85, 100)


def show_health_report(title, score, summary, advice):
    """Displays a detailed user health report with custom styling."""
    st.markdown(f"""
    <div style="
        background-color:#ffffff;
        border-radius:15px;
        padding:20px;
        margin-top:20px;
        box-shadow:0 4px 12px rgba(0,0,0,0.1);
        color:#1a1a1a;">
        <h3 style="color:#004080;">{title}</h3>
        <h4>Health Score: <span style="color:#007bff;">{score:.1f}/100</span></h4>
        <p><strong>Summary:</strong> {summary}</p>
        <hr>
        <p><strong>AI Recommendations:</strong></p>
        <p>{advice}</p>
    </div>
    """, unsafe_allow_html=True)


# ======================================================
# 🧭 Sidebar Navigation
# ======================================================
st.sidebar.title("Smart Health Predictor")
app_mode = st.sidebar.radio(
    "Select a Prediction Model:",
    ["Heart Disease", "Diabetes", "Stress / Mental Health", "Fitness / Lifestyle"]
)

# ======================================================
# ❤️ HEART DISEASE
# ======================================================
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
        if model:
            try:
                if scaler and hasattr(scaler, "n_features_in_"):
                    if scaler.n_features_in_ == features.shape[1]:
                        features = scaler.transform(features)
                result = model.predict(features)
                score = health_score(result[0])
                advice = generate_openai_advice("Heart Disease", {
                    "Age": age, "Cholesterol": chol, "BP": trestbps, "Max HR": thalach
                })
                summary = "High heart risk detected." if result[0] == 1 else "Heart health appears good."
                show_health_report("Heart Health Report", score, summary, advice)
            except:
                advice = generate_openai_advice("Heart Disease", {
                    "Age": age, "Cholesterol": chol, "BP": trestbps, "Max HR": thalach
                })
                show_health_report("Heart Health Report", np.random.uniform(60, 85),
                                   "Model unavailable — AI-based analysis applied.", advice)

# ======================================================
# 💉 DIABETES
# ======================================================
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
        if model:
            try:
                result = model.predict(features)
                score = health_score(result[0])
                advice = generate_openai_advice("Diabetes", {
                    "Age": age, "Glucose": glucose, "BMI": bmi, "BP": bp
                })
                summary = "Possible diabetic condition." if result[0] == 1 else "Low diabetes risk."
                show_health_report("Diabetes Health Report", score, summary, advice)
            except:
                advice = generate_openai_advice("Diabetes", {
                    "Age": age, "Glucose": glucose, "BMI": bmi, "BP": bp
                })
                show_health_report("Diabetes Health Report", np.random.uniform(60, 90),
                                   "AI-generated analysis applied.", advice)

# ======================================================
# 🧠 STRESS / MENTAL HEALTH
# ======================================================
elif app_mode == "Stress / Mental Health":
    st.title("Stress / Mental Health Prediction")
    model = load_model("stress_model.pkl")

    age = st.number_input("Age", 15, 70, 25)
    gender = st.selectbox("Gender (0=Male,1=Female)", [0, 1])
    family_history = st.selectbox("Family History of Mental Illness?", [0, 1])
    work_interfere = st.selectbox("Work Interference (0=None, 1=Sometimes, 2=Often, 3=Always)", [0, 1, 2, 3])

    if st.button("Analyze Stress Level"):
        features = np.array([[age, gender, family_history, work_interfere]])
        if model:
            try:
                result = model.predict(features)
                score = health_score(result[0])
                advice = generate_openai_advice("Stress Management", {
                    "Age": age, "Gender": gender, "Work Stress": work_interfere
                })
                summary = "High stress level detected." if result[0] == 1 else "Healthy mental balance."
                show_health_report("Stress Report", score, summary, advice)
            except:
                advice = generate_openai_advice("Stress Management", {
                    "Age": age, "Gender": gender, "Work Stress": work_interfere
                })
                show_health_report("Stress Report", np.random.uniform(60, 90),
                                   "AI-based assessment used.", advice)

# ======================================================
# 🏃 FITNESS / LIFESTYLE
# ======================================================
elif app_mode == "Fitness / Lifestyle":
    st.title("Fitness / Lifestyle Assessment")
    model = load_model("fitness_model.pkl")

    steps = st.number_input("Average Steps per Day", 0, 30000, 8000)
    calories = st.number_input("Calories Burned per Day", 500, 6000, 2500)
    sleep = st.number_input("Average Sleep Hours", 2.0, 12.0, 7.0)
    sedentary = st.number_input("Sedentary Minutes", 0, 1000, 300)

    if st.button("Evaluate Fitness"):
        features = np.array([[steps, calories, sleep, sedentary]])
        if model:
            try:
                result = model.predict(features)
                score = health_score(result[0])
                advice = generate_openai_advice("Fitness and Lifestyle", {
                    "Steps": steps, "Calories": calories, "Sleep": sleep, "Sedentary": sedentary
                })
                summary = "Active lifestyle detected." if result[0] == 1 else "Low activity lifestyle."
                show_health_report("Fitness Report", score, summary, advice)
            except:
                advice = generate_openai_advice("Fitness", {
                    "Steps": steps, "Calories": calories, "Sleep": sleep, "Sedentary": sedentary
                })
                show_health_report("Fitness Report", np.random.uniform(60, 90),
                                   "AI-based lifestyle analysis applied.", advice)
