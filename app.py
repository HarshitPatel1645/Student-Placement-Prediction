import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Student Placement Prediction",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Placement Prediction System")
st.markdown(
    "Predict whether a student will be placed and estimate the expected salary."
)

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("studentdata.csv")

# -----------------------------
# Encode Target
# -----------------------------
le = LabelEncoder()
df["Placement_Status"] = le.fit_transform(df["Placement_Status"])

features = [
    "CGPA",
    "Internships",
    "Projects",
    "Aptitude_Test_Score",
    "Communication_Score",
    "Academic_Backlogs"
]

# -----------------------------
# Logistic Regression Model
# -----------------------------
X = df[features]
y = df["Placement_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, log_model.predict(X_test))

# -----------------------------
# Linear Regression Model
# -----------------------------
salary_df = df[df["Salary_LPA"] > 0]

X_salary = salary_df[features]
y_salary = salary_df["Salary_LPA"]

Xs_train, Xs_test, ys_train, ys_test = train_test_split(
    X_salary,
    y_salary,
    test_size=0.20,
    random_state=42
)

linear_model = LinearRegression()
linear_model.fit(Xs_train, ys_train)

salary_prediction = linear_model.predict(Xs_test)

mae = mean_absolute_error(ys_test, salary_prediction)
r2 = r2_score(ys_test, salary_prediction)

# -----------------------------
# Sidebar Input
# -----------------------------
st.sidebar.header("Enter Student Details")

cgpa = st.sidebar.slider("CGPA", 0.0, 10.0, 8.0)

internships = st.sidebar.number_input(
    "Internships",
    min_value=0,
    max_value=10,
    value=1
)

projects = st.sidebar.number_input(
    "Projects",
    min_value=0,
    max_value=20,
    value=2
)

aptitude = st.sidebar.slider(
    "Aptitude Test Score",
    0,
    100,
    75
)

communication = st.sidebar.slider(
    "Communication Score",
    0,
    100,
    80
)

backlogs = st.sidebar.number_input(
    "Academic Backlogs",
    min_value=0,
    max_value=10,
    value=0
)

# -----------------------------
# Prediction
# -----------------------------
if st.sidebar.button("Predict Placement"):

    sample = pd.DataFrame({
        "CGPA": [cgpa],
        "Internships": [internships],
        "Projects": [projects],
        "Aptitude_Test_Score": [aptitude],
        "Communication_Score": [communication],
        "Academic_Backlogs": [backlogs]
    })

    placement = log_model.predict(sample)
    probability = log_model.predict_proba(sample)

    salary = linear_model.predict(sample)

    status = le.inverse_transform(placement)[0]

    placed_probability = probability[0][1] * 100
    not_placed_probability = probability[0][0] * 100

    st.divider()
    st.subheader("Prediction Result")

    if status == "Placed":
        st.success("🎉 Congratulations! The student is likely to be placed.")
    else:
        st.error("❌ The student has a lower chance of getting placed.")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Placement Status",
            value=status
        )

    with col2:
        st.metric(
            label="Expected Salary (LPA)",
            value=f"{salary[0]:.2f}"
        )

    st.subheader("Placement Probability")

    st.progress(int(placed_probability))

    col3, col4 = st.columns(2)

    with col3:
        st.metric(
            "Placed Probability",
            f"{placed_probability:.2f}%"
        )

    with col4:
        st.metric(
            "Not Placed Probability",
            f"{not_placed_probability:.2f}%"
        )

    if placed_probability >= 80:
        st.success("🌟 Excellent placement chances!")
    elif placed_probability >= 60:
        st.info("👍 Good placement chances. Keep improving your skills.")
    elif placed_probability >= 40:
        st.warning("⚠️ Average placement chances. More preparation is recommended.")
    else:
        st.error("❌ Low placement chances. Focus on improving academics and skills.")

# -----------------------------
# Model Performance
# -----------------------------
st.divider()

st.subheader("Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Classification Accuracy", f"{accuracy*100:.2f}%")

with col2:
    st.metric("Salary MAE", f"{mae:.2f}")

with col3:
    st.metric("Salary R² Score", f"{r2:.2f}")

st.divider()
st.caption("🎓 Student Placement Prediction System")
