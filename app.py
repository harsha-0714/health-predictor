# app.py
import streamlit as st
import pickle, os
import numpy as np
import datetime
from fpdf import FPDF

st.set_page_config(page_title="AI Health Insight Dashboard", layout="wide", page_icon="💊")

# ---------- Helper: load model safely ----------
def load_model(path):
    try:
        with open(path, "rb") as f:
            m = pickle.load(f)
        return m
    except Exception as e:
        st.sidebar.error(f"Model load error: {os.path.basename(path)} — {e}")
        return None

# ---------- Backgrounds (replace URLs with your own or local static files) ----------
bg_images = {
    "Heart Disease": "https://images.unsplash.com/photo-1550831107-1553da8c8464?auto=format&fit=crop&w=1600&q=60",
    "Diabetes": "https://images.unsplash.com/photo-1582719478250-c3e9e2d4a5af?auto=format&fit=crop&w=1600&q=60",
    "Stress Level": "https://images.unsplash.com/photo-1524504388940-b1c1722653e1?auto=format&fit=crop&w=1600&q=60",
    "Fitness Level": "https://images.unsplash.com/photo-1526403224749-0b6a5aa1d1d2?auto=format&fit=crop&w=1600&q=60",
    "Generate Health Report": "https://images.unsplash.com/photo-1505755662778-989d0524087e?auto=format&fit=crop&w=1600&q=60"
}

# ---------- CSS helper to set moving background per page ----------
def set_background_for(page_name):
    img = bg_images.get(page_name, "")
    # Animated overlay + subtle parallax-like zoom
    css = f"""
    <style>
    .stApp {{
        background: linear-gradient(180deg, rgba(0,0,0,0.25), rgba(0,0,0,0.25)), url("{img}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        animation: zoom 20s infinite alternate;
    }}
    @keyframes zoom {{
        0% {{ transform: scale(1); filter: brightness(0.9) contrast(1); }}
        100% {{ transform: scale(1.03); filter: brightness(1) contrast(1.02); }}
    }}
    /* Card style */
    .card {{
      background: rgba(255,255,255,0.85);
      border-radius: 14px;
      padding: 18px;
      box-shadow: 0 6px 18px rgba(0,0,0,0.25);
      backdrop-filter: blur(4px);
      color: #0b1320;
    }}
    /* Accent buttons */
    div.stButton > button:first-child {{
      background-image: linear-gradient(90deg,#5561ff,#9b6bff);
      color: white;
      border: none;
      height: 42px;
      width: 100%;
      border-radius: 10px;
    }}
    /* Small text muted */
    .muted {{ color: rgba(11,19,32,0.6); font-size:0.9rem; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ---------- Load models ----------
heart_model = load_model("/content/heart_model.pkl")
diabetes_model = load_model("models/diabetes_model.pkl")
stress_model = load_model("models/stress_model.pkl")
fitness_model = load_model("models/fitness_model.pkl")

# ---------- Sidebar ----------
st.sidebar.markdown("<div class='card'><h2>AI Health Insight</h2><p class='muted'>Select a module</p></div>", unsafe_allow_html=True)
page = st.sidebar.radio("", ["Heart Disease", "Diabetes", "Stress Level", "Fitness Level", "Generate Health Report"])

# provide a quick status in sidebar for loaded models
with st.sidebar.expander("Model status"):
    def model_status(m):
        if m is None: return "❌ not loaded"
        return f"✅ loaded (expects {getattr(m,'n_features_in_', '?')} features)"
    st.write("Heart:", model_status(heart_model))
    st.write("Diabetes:", model_status(diabetes_model))
    st.write("Stress:", model_status(stress_model))
    st.write("Fitness:", model_status(fitness_model))

set_background_for(page)

# ---------- Session state for results ----------
if "results" not in st.session_state:
    st.session_state["results"] = {}

# ---------- Utility: save PDF ----------
def generate_pdf(results, overall_score, recommendations):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, "AI Health Insight Report", ln=True, align="C")
    pdf.ln(6)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, f"Date: {datetime.date.today()}", ln=True)
    pdf.ln(6)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, "Prediction Summary:", ln=True)
    pdf.set_font("Arial", size=12)
    for k, v in results.items():
        pdf.cell(0, 8, f"- {k}: {v}", ln=True)
    pdf.ln(6)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, f"Overall Health Score: {overall_score}/100", ln=True)
    pdf.ln(6)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, "Recommendations:", ln=True)
    pdf.set_font("Arial", size=12)
    for r in recommendations:
        pdf.multi_cell(0, 7, f"- {r}")
    out_path = "Health_Report.pdf"
    pdf.output(out_path)
    return out_path

# ---------- UX header ----------
st.markdown("<div class='card'><h1 style='margin:0'>💊 AI Health Insight Dashboard</h1><p class='muted' style='margin-top:6px'>Multi-model health assistant — predictions, combined score & report</p></div>", unsafe_allow_html=True)
st.write("")

# ---------- Pages ----------
# 1) HEART (13 features expected for standard UCI model)
if page == "Heart Disease":
    st.markdown("<div class='card'><h3>❤️ Heart Disease Predictor</h3><p class='muted'>Enter the values below — model expects 13 features (UCI heart dataset).</p></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", 10, 120, 45)
        sex = st.selectbox("Gender", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type (0–3)", [0,1,2,3])
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    with col2:
        chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0,1])
        restecg = st.selectbox("Resting ECG (0–2)", [0,1,2])
        thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
    with col3:
        exang = st.selectbox("Exercise Induced Angina (0/1)", [0,1])
        oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0, step=0.1)
        slope = st.selectbox("Slope (0–2)", [0,1,2])
        ca = st.number_input("Number of major vessels (0–3)", 0, 3, 0)
        thal = st.selectbox("Thal (0=normal,1=fixed,2=reversible)", [0,1,2])

    if st.button("Predict Heart Risk"):
        if heart_model is None:
            st.error("Heart model not loaded.")
        else:
            gender_val = 1 if sex=="Male" else 0
            features = np.array([[age,gender_val,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
            expected = getattr(heart_model, "n_features_in_", None)
            if expected and expected != features.shape[1]:
                st.error(f"Feature mismatch: model expects {expected} but you're passing {features.shape[1]}. Update app or model.")
            else:
                pred = heart_model.predict(features)[0]
                label = "⚠️ High Risk" if int(pred)==1 else "✅ Low Risk"
                st.session_state["results"]["Heart Disease"] = label
                if pred==1:
                    st.error("⚠️ High Risk of Heart Disease detected.")
                else:
                    st.success("✅ Low Risk of Heart Disease detected.")

# 2) DIABETES (PIMA: 8 features)
elif page == "Diabetes":
    st.markdown("<div class='card'><h3>💉 Diabetes Predictor</h3><p class='muted'>PIMA-style inputs.</p></div>", unsafe_allow_html=True)
    pregnancies = st.number_input("Pregnancies", 0, 20, 0)
    glucose = st.number_input("Glucose", 0, 300, 100)
    bp = st.number_input("Blood Pressure", 0, 200, 70)
    skin = st.number_input("Skin Thickness", 0, 100, 20)
    insulin = st.number_input("Insulin", 0, 900, 80)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 5.0, 0.5)
    age_d = st.number_input("Age", 10, 120, 35)

    if st.button("Predict Diabetes Risk"):
        if diabetes_model is None:
            st.error("Diabetes model not loaded.")
        else:
            features = np.array([[pregnancies,glucose,bp,skin,insulin,bmi,dpf,age_d]])
            expected = getattr(diabetes_model, "n_features_in_", None)
            if expected and expected != features.shape[1]:
                st.error(f"Feature mismatch: model expects {expected} but you're passing {features.shape[1]}.")
            else:
                pred = diabetes_model.predict(features)[0]
                label = "⚠️ High Risk" if int(pred)==1 else "✅ Low Risk"
                st.session_state["results"]["Diabetes"] = label
                if pred==1: st.error("⚠️ High Risk of Diabetes detected.")
                else: st.success("✅ Low Risk of Diabetes detected.")

# 3) STRESS (simplified 5-feature model we retrained)
elif page == "Stress Level":
    st.markdown("<div class='card'><h3>🧠 Stress Level Predictor</h3><p class='muted'>Simplified inputs (fast & user-friendly).</p></div>", unsafe_allow_html=True)
    age_s = st.number_input("Age", 15, 100, 28)
    gender_s = st.selectbox("Gender", ["Male","Female","Other"])
    family_history = st.selectbox("Family History of Mental Illness?", ["No","Yes"])
    company_size = st.selectbox("Company Size", ["1-5","6-25","26-100","100-500","500-1000","More than 1000"])
    benefits = st.selectbox("Mental Health Benefits Provided?", ["No","Yes"])

    if st.button("Predict Stress Level"):
        if stress_model is None:
            st.error("Stress model not loaded.")
        else:
            gender_map = {"Male":0,"Female":1,"Other":2}
            fam_map = {"No":0,"Yes":1}
            benefits_map = {"No":0,"Yes":1}
            size_map = {"1-5":0,"6-25":1,"26-100":2,"100-500":3,"500-1000":4,"More than 1000":5}
            features = np.array([[age_s, gender_map[gender_s], fam_map[family_history], size_map[company_size], benefits_map[benefits]]])
            expected = getattr(stress_model, "n_features_in_", None)
            if expected and expected != features.shape[1]:
                st.error(f"Feature mismatch: model expects {expected} but you're passing {features.shape[1]}.")
            else:
                pred = stress_model.predict(features)[0]
                if pred >= 2:
                    label = "⚠️ High"
                    st.error("⚠️ High Stress Level detected.")
                elif pred == 1:
                    label = "⚠️ Moderate"
                    st.warning("⚠️ Moderate Stress detected.")
                else:
                    label = "✅ Low"
                    st.success("✅ Low Stress Level detected.")
                st.session_state["results"]["Stress Level"] = label

# 4) FITNESS (simple 4-feature regression to predict calories or fitness)
elif page == "Fitness Level":
    st.markdown("<div class='card'><h3>🏃‍♂️ Fitness Predictor</h3><p class='muted'>Enter daily activity values.</p></div>", unsafe_allow_html=True)
    steps = st.number_input("Total Steps per Day", 0, 100000, 8000)
    distance = st.number_input("Total Distance (km)", 0.0, 100.0, 6.0)
    active_minutes = st.number_input("Active Minutes per Day", 0, 1440, 60)
    sedentary = st.number_input("Sedentary Minutes per Day", 0, 1440, 600)

    if st.button("Analyze Fitness Level"):
        if fitness_model is None:
            st.error("Fitness model not loaded.")
        else:
            features = np.array([[steps, distance, active_minutes, sedentary]])
            expected = getattr(fitness_model, "n_features_in_", None)
            if expected and expected != features.shape[1]:
                st.error(f"Feature mismatch: model expects {expected} but you're passing {features.shape[1]}.")
            else:
                try:
                    pred = fitness_model.predict(features)[0]
                    st.session_state["results"]["Fitness"] = f"{pred:.1f} (predicted calories)"
                    st.success(f"Predicted calories: {pred:.1f}")
                except Exception as e:
                    st.error(f"Prediction error: {e}")

# 5) GENERATE REPORT
elif page == "Generate Health Report":
    st.markdown("<div class='card'><h3>🩺 Generate Health Report</h3><p class='muted'>Combines model outputs into a single health score and a downloadable PDF.</p></div>", unsafe_allow_html=True)
    if st.session_state["results"]:
        st.write("### Summary of latest predictions")
        for k,v in st.session_state["results"].items():
            st.write(f"- **{k}**: {v}")

        # Compute a simple overall health score:
        # Heart: Low=30 pts, High=5 pts
        # Diabetes: Low=25, High=5
        # Stress: Low=20, Moderate=10, High=0
        # Fitness: normalized 0-25 based on predicted calories (better calories -> more points)
        def score_from_results(res):
            score = 0
            # heart
            heart = res.get("Heart Disease", None)
            if heart == "✅ Low Risk": score += 30
            elif heart == "⚠️ High Risk": score += 5
            # diabetes
            diab = res.get("Diabetes", None)
            if diab == "✅ Low Risk": score += 25
            elif diab == "⚠️ High Risk": score += 5
            # stress
            stv = res.get("Stress Level", None)
            if stv == "✅ Low": score += 20
            elif stv == "⚠️ Moderate": score += 10
            elif stv == "⚠️ High": score += 0
            # fitness
            fit = res.get("Fitness", None)
            if fit:
                try:
                    cal = float(str(fit).split()[0])
                    # assume 2000–3000 is good range
                    if cal < 1500:
                        score += 5
                    elif cal < 2000:
                        score += 10
                    elif cal < 3000:
                        score += 20
                    else:
                        score += 25
                except:
                    score += 10
            return min(100, int(score))

        overall = score_from_results(st.session_state["results"])

        st.metric("Overall Health Score", f"{overall}/100")

        # recommendations
        recommendations = [
            "Exercise 30–45 mins daily (mix cardio + strength).",
            "Balanced diet: reduce processed sugar, prefer whole foods.",
            "Aim for 7–8 hours of sleep nightly.",
            "Practice stress-reduction: breathing, short breaks, or meditation.",
            "Regular medical checkups (BP, cholesterol, glucose) as advised."
        ]

        if st.button("📄 Download PDF Report"):
            path = generate_pdf(st.session_state["results"], overall, recommendations)
            with open(path, "rb") as f:
                st.download_button("⬇️ Download Health Report", f, file_name="Health_Report.pdf", mime="application/pdf")

    else:
        st.warning("No predictions found yet — run the modules first (Heart / Diabetes / Stress / Fitness).")
