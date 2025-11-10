# Refactored Health Predictor app with modern UI, Lottie, gauges, and robust sidebar fallback
import streamlit as st
try:
    from streamlit_option_menu import option_menu
except Exception:
    # Fallback: provide a minimal option_menu replacement using sidebar radio
    def option_menu(menu_title=None, options=None, icons=None, menu_icon=None, default_index=0, orientation="vertical"):
        if menu_title:
            st.sidebar.header(menu_title)
        labels = []
        if icons and len(icons) >= len(options):
            for i, opt in enumerate(options):
                try:
                    labels.append(f"{icons[i]}  {opt}")
                except Exception:
                    labels.append(opt)
        else:
            labels = options
        sel = st.sidebar.radio("", labels, index=default_index)
        for i, lab in enumerate(labels):
            if lab == sel:
                return options[i]
        return options[default_index]

from streamlit_lottie import st_lottie
import plotly.graph_objects as go
import numpy as np
import pickle
import os
import requests
from typing import Optional

# ---------------------------
# Page config & cache
# ---------------------------
st.set_page_config(page_title="Health Predictor",
                   page_icon="❤️",
                   layout="wide",
                   initial_sidebar_state="expanded")

# ---------------------------
# Utilities
# ---------------------------
@st.cache_resource
def load_pickle_resource(model_name: str):
    possible_paths = [
        os.path.join("models", model_name),
        os.path.join("assets", "models", model_name),
        model_name,
        os.path.join("/content/models", model_name),
    ]
    for p in possible_paths:
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                st.sidebar.error(f"Failed to load {model_name}: {e}")
                return None
    return None


def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None


def start_card():
    st.markdown('<div class="card">', unsafe_allow_html=True)


def end_card():
    st.markdown('</div>', unsafe_allow_html=True)


def show_gauge(prob: float, accent: str, title: str = "Risk Probability"):
    pct = float(np.clip(prob, 0.0, 1.0)) * 100
    color = accent
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={'suffix': '%'},
        title={'text': title},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': '#e6fffa'},
                {'range': [50, 100], 'color': '#fff7ed'}
            ],
            'threshold': {
                'line': {'color': '#ff6b6b', 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=230)
    st.plotly_chart(fig, use_container_width=True)


def predict_with_model(model, features: np.ndarray, scaler=None) -> (float, int):
    if scaler is not None:
        try:
            if hasattr(scaler, "transform"):
                features = scaler.transform(features)
        except Exception:
            pass

    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(features)[0][1]
            pred = int(proba >= 0.5)
            return float(proba), pred
        except Exception:
            pass

    pred = int(model.predict(features)[0])
    proba = float(pred)
    return proba, pred

# ---------------------------
# Styling (glass / modern)
# ---------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #f6fbff 0%, #ffffff 100%);
        color: #0f172a;
    }
    .card {
        background: rgba(255,255,255,0.86);
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 6px 20px rgba(15,23,42,0.06);
        margin-bottom: 16px;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .card:hover { transform: translateY(-6px); box-shadow: 0 14px 40px rgba(15,23,42,0.10); }
    .model-title { font-weight: 600; font-size: 20px; margin-bottom: 6px; color: #042a55; }
    .muted { color: #6b7280; font-size: 13px; margin-bottom: 8px; }
    .accent-pill { display:inline-block; padding:6px 10px; border-radius:999px; color:white; font-weight:600; }
    .stButton>button {
        background: linear-gradient(90deg,#4f46e5,#06b6d4);
        color: white;
        border: none;
        padding: 8px 12px;
        border-radius: 8px;
    }
    [data-testid="stSidebar"] { background: rgba(255,255,255,0.98); }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Sidebar menu
# ---------------------------
with st.sidebar:
    st.image("assets/logo.png" if os.path.exists("assets/logo.png") else "https://static.thenounproject.com/png/3407196-200.png",
             width=72)
    selection = option_menu(
        menu_title="Models",
        options=["Home", "Heart Disease", "Diabetes", "Stress / Mental Health", "Fitness / Lifestyle"],
        icons=["house", "heart", "droplet", "emoji-smile", "activity"],
        menu_icon="activity",
        default_index=1
    )
    st.write("---")
    st.header("Profile (optional)")
    if "profile" not in st.session_state:
        st.session_state.profile = {}
    p_age = st.number_input("Age", min_value=10, max_value=120, value=30, key="sidebar_age")
    p_gender = st.selectbox("Gender", ["Prefer not to say", "Male", "Female"], key="sidebar_gender")
    if st.button("Save profile"):
        st.session_state.profile = {"age": p_age, "gender": p_gender}
        st.success("Profile saved")

# ---------------------------
# Shared assets (Lotties & accents)
# ---------------------------
LOTTIES = {
    "Heart Disease": "https://assets6.lottiefiles.com/packages/lf20_x62chJ.json",
    "Diabetes": "https://assets6.lottiefiles.com/packages/lf20_0yfsb3a1.json",
    "Stress / Mental Health": "https://assets6.lottiefiles.com/packages/lf20_jtbfg2nb.json",
    "Fitness / Lifestyle": "https://assets6.lottiefiles.com/packages/lf20_vfmmn1gs.json",
}
ACCENTS = {
    "Heart Disease": "#ef476f",
    "Diabetes": "#06b6d4",
    "Stress / Mental Health": "#6366f1",
    "Fitness / Lifestyle": "#f97316",
}

# ---------------------------
# Home / landing
# ---------------------------
if selection == "Home":
    start_card()
    st.markdown('<div class="model-title">Welcome to Health Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Choose a model from the left to get started. Save a profile to pre-fill some inputs.</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([2,1])
    with col1:
        st.write("This demo provides quick, friendly health risk predictions using your inputs. The UI uses cards, Lottie animations, and interactive gauges for nicer feedback.")
        st.write("Tips:")
        st.markdown("- Use the Profile area to save age/gender and speed up data entry.")
        st.markdown("- Models are loaded from the models/ directory (heart_model.pkl, diabetes_model.pkl, stress_model.pkl, fitness_model.pkl).")
    with col2:
        l = load_lottie_url("https://assets6.lottiefiles.com/packages/lf20_hdy0htc1.json")
        if l:
            st_lottie(l, height=220)
    end_card()
    st.stop()

# ---------------------------
# Helper for model sections
# ---------------------------
def model_section_header(name: str):
    start_card()
    st.markdown(f'<div style="display:flex;align-items:center;justify-content:space-between;">
                f'<div><div class="model-title">{name} Predictor</div>'
                f'<div class="muted">Enter your details and click Predict</div></div>'
                f'<div><span class="accent-pill" style="background:{ACCENTS.get(name, "#10b981")};">'
                f'{name.split()[0]}</span></div></div>',
                unsafe_allow_html=True)

# ---------------------------
# Model: Heart Disease
# ---------------------------
if selection == "Heart Disease":
    model_section_header("Heart Disease")
    lottie = load_lottie_url(LOTTIES["Heart Disease"])
    if lottie:
        st_lottie(lottie, height=140)
    model = load_pickle_resource("heart_model.pkl")
    scaler = load_pickle_resource("heart_scaler.pkl")

    with st.form("heart_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 20, 100, value=int(st.session_state.profile.get("age", 50)))
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
            chol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
            thalach = st.number_input("Max Heart Rate", 60, 220, 150)
            fbs = st.selectbox("Fasting Blood Sugar >120", [0, 1])
        with col2:
            sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
            exang = st.selectbox("Exercise Induced Angina", [0, 1])
            oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, step=0.1)
            slope = st.selectbox("Slope (0–2)", [0, 1, 2])
        submitted = st.form_submit_button("🔍 Predict Heart Risk")

    if submitted:
        if model is None:
            st.error("Model not found. Make sure models/heart_model.pkl exists.")
        else:
            try:
                features = np.array([[age, sex, cp, trestbps, chol, fbs, exang, thalach, oldpeak, slope]])
                prob, pred = predict_with_model(model, features, scaler)
                risk_text = "High Risk" if pred == 1 else "Low Risk"
                start_card()
                st.metric(label="Risk Level", value=risk_text, delta=f"{int(prob*100)}%")
                show_gauge(prob, ACCENTS["Heart Disease"], title="Heart Disease Risk")
                if pred == 1:
                    st.error("⚠️ High Risk: Please consult a cardiologist.")
                else:
                    st.success("✅ Low Risk: Keep up a healthy lifestyle.")
                end_card()
            except Exception as e:
                st.error(f"Prediction failed: {e}")
    end_card()

# ---------------------------
# Model: Diabetes
# ---------------------------
elif selection == "Diabetes":
    model_section_header("Diabetes")
    lottie = load_lottie_url(LOTTIES["Diabetes"])
    if lottie:
        st_lottie(lottie, height=140)
    model = load_pickle_resource("diabetes_model.pkl")

    with st.form("diabetes_form"):
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
            age = st.number_input("Age", 20, 100, int(st.session_state.profile.get("age", 40)))
        submitted = st.form_submit_button("🔍 Predict Diabetes")

    if submitted:
        if model is None:
            st.error("Model not found. Make sure models/diabetes_model.pkl exists.")
        else:
            try:
                features = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
                prob, pred = predict_with_model(model, features, scaler=None)
                risk_text = "Diabetic" if pred == 1 else "Non-Diabetic"
                start_card()
                st.metric(label="Diabetes Status", value=risk_text, delta=f"{int(prob*100)}%")
                show_gauge(prob, ACCENTS["Diabetes"], title="Diabetes Probability")
                if pred == 1:
                    st.error("⚠️ Diabetic: Follow medical guidance and diet control.")
                else:
                    st.success("✅ Non-Diabetic: Maintain healthy habits.")
                end_card()
            except Exception as e:
                st.error(f"Prediction failed: {e}")
    end_card()

# ---------------------------
# Model: Stress / Mental Health
# ---------------------------
elif selection == "Stress / Mental Health":
    model_section_header("Stress / Mental Health")
    lottie = load_lottie_url(LOTTIES["Stress / Mental Health"])
    if lottie:
        st_lottie(lottie, height=140)
    model = load_pickle_resource("stress_model.pkl")

    with st.form("stress_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 15, 70, int(st.session_state.profile.get("age", 25)))
            gender = st.selectbox("Gender (0=Male,1=Female)", [0, 1])
            family_history = st.selectbox("Family History of Mental Illness (0/1)", [0, 1])
        with col2:
            employees = st.number_input("No. of Employees (approx.)", 1, 1000, 50)
            benefits = st.selectbox("Employer Benefits Provided (0/1)", [0, 1])
        submitted = st.form_submit_button("🔍 Predict Stress Level")

    if submitted:
        if model is None:
            st.error("Model not found. Make sure models/stress_model.pkl exists.")
        else:
            try:
                features = np.array([[age, gender, family_history, employees, benefits]])
                prob, pred = predict_with_model(model, features, scaler=None)
                risk_text = "High Stress Risk" if pred == 1 else "Low Stress Risk"
                start_card()
                st.metric(label="Stress Level", value=risk_text, delta=f"{int(prob*100)}%")
                show_gauge(prob, ACCENTS["Stress / Mental Health"], title="Stress Risk")
                if pred == 1:
                    st.error("⚠️ High Stress Risk: Prioritize mental wellness and seek support.")
                else:
                    st.success("✅ Low Stress Risk: Keep maintaining emotional balance.")
                end_card()
            except Exception as e:
                st.error(f"Prediction failed: {e}")
    end_card()

# ---------------------------
# Model: Fitness / Lifestyle
# ---------------------------
elif selection == "Fitness / Lifestyle":
    model_section_header("Fitness / Lifestyle")
    lottie = load_lottie_url(LOTTIES["Fitness / Lifestyle"])
    if lottie:
        st_lottie(lottie, height=140)
    model = load_pickle_resource("fitness_model.pkl")

    with st.form("fitness_form"):
        col1, col2 = st.columns(2)
        with col1:
            steps = st.number_input("Avg. Steps per Day", 0, 50000, 8000)
            calories = st.number_input("Avg. Calories Burned", 100, 6000, 2500)
        with col2:
            sleep = st.number_input("Sleep Duration (hours)", 2.0, 12.0, 7.0)
            sedentary = st.number_input("Sedentary Minutes", 0, 1000, 300)
        submitted = st.form_submit_button("🔍 Predict Fitness Level")

    if submitted:
        if model is None:
            st.error("Model not found. Make sure models/fitness_model.pkl exists.")
        else:
            try:
                features = np.array([[steps, calories, sleep, sedentary]])
                prob, pred = predict_with_model(model, features, scaler=None)
                fitness_text = "Active Lifestyle" if pred == 1 else "Sedentary Lifestyle"
                start_card()
                st.metric(label="Fitness Assessment", value=fitness_text, delta=f"{int(prob*100)}%")
                show_gauge(prob, ACCENTS["Fitness / Lifestyle"], title="Fitness Score")
                if pred == 1:
                    st.success("✅ Active Lifestyle: Keep up the great habits!")
                else:
                    st.error("⚠️ Sedentary: Increase daily movement and reduce screen time.")
                end_card()
            except Exception as e:
                st.error(f"Prediction failed: {e}")
    end_card()