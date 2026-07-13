import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Real Student Placement Predictor", page_icon="🎓", layout="centered"
)


# --- DATA LOADING & PREPROCESSING ---
@st.cache_resource
def load_and_train_model():
    try:
        df = pd.read_csv(r"C:\Users\PARTH\PycharmProjects\HarshitPythonProject\Placement.csv")
    except FileNotFoundError:
        st.error(
            "⚠️ 'Placement.csv' not found! Please place your CSV data file in this directory."
        )
        st.stop()

    # Clean column names (strip whitespace)
    df.columns = df.columns.str.strip()

    # Mappings for categorical text variables
    skill_map = {"Low": 1, "Medium": 2, "High": 3}
    binary_map = {"No": 0, "Yes": 1}

    df["Communication Skills"] = df["Communication Skills"].map(skill_map)
    df["Technical Skills"] = df["Technical Skills"].map(skill_map)
    df["Internship Experience"] = df["Internship Experience"].map(binary_map)
    df["Work Experience"] = df["Work Experience"].map(binary_map)

    # If your raw text snippet doesn't have a label column, create a logical one for training
    if "Placement_Status" not in df.columns:
        # Rules: Placed if high CGPA & technical skills, or good degree & experience without high backlogs
        df["Placement_Status"] = np.where(
            (df["CGPA"] >= 7.5)
            & (df["Technical Skills"] >= 2)
            & (df["Backlogs"] == 0),
            1,
            0,
        )

    # Separate features and target
    X = df.drop(columns=["Placement_Status"])
    y = df["Placement_Status"]

    # Train scaler and model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    return model, scaler


# Initialize the model and preprocessing scaler
model, scaler = load_and_train_model()

# --- STREAMLIT USER INTERFACE ---
st.title("🎓 Student Placement Prediction App")
st.write("Input the student's metrics below to calculate placement probability.")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Academic Metrics")
    cgpa = st.number_input(
        "CGPA", min_value=0.0, max_value=10.0, value=7.92, step=0.01
    )
    p_10th = st.number_input(
        "10th Percentage", min_value=0.0, max_value=100.0, value=85.74, step=0.01
    )
    p_12th = st.number_input(
        "12th Percentage", min_value=0.0, max_value=100.0, value=70.40, step=0.01
    )
    p_degree = st.number_input(
        "Degree Percentage", min_value=0.0, max_value=100.0, value=73.30, step=0.01
    )
    backlogs = st.number_input(
        "Current Backlogs", min_value=0, max_value=10, value=0, step=1
    )

with col2:
    st.subheader("🛠️ Skills & Experience")
    comm_skills_text = st.selectbox(
        "Communication Skills", options=["Low", "Medium", "High"], index=1
    )
    tech_skills_text = st.selectbox(
        "Technical Skills", options=["Low", "Medium", "High"], index=2
    )
    internship_text = st.selectbox(
        "Internship Experience", options=["No", "Yes"], index=1
    )
    work_exp_text = st.selectbox("Work Experience", options=["No", "Yes"], index=0)
    projects = st.number_input(
        "Projects Completed", min_value=0, max_value=10, value=0, step=1
    )

st.markdown("---")

# --- CONVERT STREAMLIT INPUTS TO NUMERIC ---
skill_encoding = {"Low": 1, "Medium": 2, "High": 3}
binary_encoding = {"No": 0, "Yes": 1}

comm_skills = skill_encoding[comm_skills_text]
tech_skills = skill_encoding[tech_skills_text]
internship = binary_encoding[internship_text]
work_exp = binary_encoding[work_exp_text]

# --- PREDICTION LOGIC ---
if st.button("🔮 Run Placement Model", type="primary"):
    # Group inputs in the exact schema order as the original CSV file
    raw_input = np.array(
        [
            [
                cgpa,
                p_10th,
                p_12th,
                p_degree,
                comm_skills,
                tech_skills,
                internship,
                projects,
                backlogs,
                work_exp,
            ]
        ]
    )

    # Scale the metrics using the calibrated training scaler
    scaled_input = scaler.transform(raw_input)

    # Compute prediction metrics
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    # Render results
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success("🎉 **Status: Placed**")
        st.metric(label="Calculated Probability", value=f"{probability:.2%}")
        st.balloons()
    else:
        st.error("⚠️ **Status: Not Placed**")
        st.metric(label="Calculated Probability", value=f"{probability:.2%}")
