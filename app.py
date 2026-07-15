import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
# ------------------------------------------------
# Page Configuration
# ------------------------------------------------
st.set_page_config(
    page_title="Student Placement Prediction",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Placement Prediction System")
st.markdown("Predict whether a student will be placed and estimate the expected salary.")

# ------------------------------------------------
# Load Dataset
# ------------------------------------------------
df = pd.read_csv("student_placement__dataset.csv")

# ------------------------------------------------
# Encode Target
# ------------------------------------------------
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



# ------------------------------------------------
# Prepare Data
# ------------------------------------------------

X = df[features]

y = df["Placement_Status"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ------------------------------------------------
# Multiple ML Models
# ------------------------------------------------

models = {

    "Logistic Regression":LogisticRegression(max_iter=1000),

    "Random Forest":
        RandomForestClassifier(n_estimators=100,random_state=42),

    "Decision Tree":DecisionTreeClassifier(random_state=42),

    "SVM":SVC(probability=True),

    "KNN":KNeighborsClassifier(n_neighbors=5)
}

model_accuracy = {}

trained_models = {}


# Train Models

for name, model in models.items():

    model.fit(X_train,y_train)

    prediction = model.predict(X_test)

    acc = accuracy_score(y_test,prediction)

    model_accuracy[name] = acc

    trained_models[name] = model

# Best Model

best_model_name = max(model_accuracy,key=model_accuracy.get)

best_model = trained_models[best_model_name]

# Best Model Selection

best_model_name = max(model_accuracy,key=model_accuracy.get)

# ------------------------------------------------
# Linear Regression
# ------------------------------------------------
salary_df = df[df["Salary_LPA"] > 0]

X_salary = salary_df[features]
y_salary = salary_df["Salary_LPA"]

Xs_train, Xs_test, ys_train, ys_test = train_test_split(
    X_salary, y_salary, test_size=0.20, random_state=42
)

linear_model = LinearRegression()
linear_model.fit(Xs_train, ys_train)

salary_prediction = linear_model.predict(Xs_test)

mae = mean_absolute_error(ys_test, salary_prediction)
r2 = r2_score(ys_test, salary_prediction)

# ------------------------------------------------
# Student Input Form (Main Page)
# ------------------------------------------------

st.divider()

st.header("🧑‍🎓 Enter Student Details")

col1, col2, col3 = st.columns(3)


with col1:

    cgpa = st.text_input("CGPA", "8.0")

    internships = st.text_input("Internships", '1')


with col2:

    projects = st.text_input("Projects", '2')

    aptitude = st.text_input("Aptitude Test Score", '75')


with col3:

    communication = st.text_input("Communication Score", '80')

    backlogs = st.text_input("Academic Backlogs", '0')


# Convert Text Input into Numbers

cgpa = float(cgpa)

internships = int(internships)

projects = int(projects)

aptitude = int(aptitude)

communication = int(communication)

backlogs = int(backlogs)

predict_button = st.button("🚀 Predict Placement")

# Convert Text to Number

cgpa = float(cgpa)

internships = int(internships)

projects = int(projects)

aptitude = int(aptitude)

communication = int(communication)

backlogs = int(backlogs)

# ------------------------------------------------
# Prediction
# ------------------------------------------------
if predict_button:

    sample = pd.DataFrame({
        "CGPA":[cgpa],
        "Internships":[internships],
        "Projects":[projects],
        "Aptitude_Test_Score":[aptitude],
        "Communication_Score":[communication],
        "Academic_Backlogs":[backlogs]
    })

    placement = best_model.predict(sample)

    probability = best_model.predict_proba(sample)

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

    with col1: st.metric(label="Placement Status",value=status)

    with col2:
        st.metric(
            label="Expected Salary (LPA)",
            value=f"{salary[0]:.2f}")

    st.subheader("Placement Probability")

    st.progress(int(placed_probability))

    col3, col4 = st.columns(2)

    with col3:
        st.metric(
            "Placed Probability",f"{placed_probability:.2f}%")

    with col4:
        st.metric(
            "Not Placed Probability",f"{not_placed_probability:.2f}%")

    if placed_probability >= 80:
        st.success("🟢 Excellent placement chances!")

    elif placed_probability >= 60:
        st.info("🟡 Good placement chances. Keep improving your skills.")

    elif placed_probability >= 40:
        st.warning("🟠 Average placement chances. More preparation is recommended.")

    else:
        st.error("🔴 Low placement chances. Focus on improving CGPA, aptitude, communication, and practical experience.")

# --------------------------------------------
# End of Application
# --------------------------------------------
st.divider()

st.caption("© Student Placement Prediction System")
