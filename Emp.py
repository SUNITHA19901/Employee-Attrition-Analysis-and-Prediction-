import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Employee Attrition Analysis", layout="wide")

# Load cleaned employee data
@st.cache_data
def load_data():
    return pd.read_csv("D:/Employee/env/Scripts/cleaned_employee_data.csv")

df = load_data()

df.rename(columns={
    'EmployeeNumber': 'employee_no',
    'PerformanceRating': 'performance_score_percent',
    'JobSatisfaction': 'job_satisfaction'
}, inplace=True)

# Cache model training
@st.cache_resource
def train_model(df):
    df_model = df.copy()
    
    # Separate target column before encoding
    target_col = 'Attrition'
    if target_col not in df_model.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")
    
    y = df_model[target_col].apply(lambda x: 1 if str(x).lower() in ['yes', '1', 'true'] else 0)
    X = df_model.drop(columns=[target_col])

    # One-hot encode features only
    X_encoded = pd.get_dummies(X, drop_first=True)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_encoded, y)

    return model, X_encoded.columns.tolist(), target_col

# Example usage (make sure your df has 'Attrition' column)
model, feature_cols, target_col = train_model(df)

# Prepare prediction DataFrame
df_pred = pd.get_dummies(df.copy(), drop_first=True)
for col in feature_cols:
    if col not in df_pred.columns:
        df_pred[col] = 0
X_pred = df_pred[feature_cols]

# Predict attrition probabilities
attrition_probs = model.predict_proba(X_pred)[:, 1]
df['attrition_risk'] = attrition_probs

# Sidebar navigation
st.sidebar.title("Employee Attrition Analysis")
page = st.sidebar.radio("Go to", ["Home", "Predict Employee Attrition"])

# -------------------------
# Page 1: Home Dashboard
if page == "Home":
    st.markdown("## :bar_chart: Employee Insights Dashboard")

    # Create three columns
    col1, col2 = st.columns(2)

    # 🔴 High-Risk Employees
    with col1:
        st.markdown("### 🔴 High-Risk Employees")
        if 'attrition_risk' in df.columns:
            high_risk = df[df['attrition_risk'] > 0.5]
            option = st.selectbox("Show", options=["All", "Top 10"], key="high_risk")
            if option == "Top 10":
                high_risk = high_risk.sort_values(by='attrition_risk', ascending=False).head(10)
            display_cols_risk = ['employee_no', 'attrition_risk', 'performance_score_percent', 'job_satisfaction']
            existing_cols_risk = [col for col in display_cols_risk if col in high_risk.columns]
            st.dataframe(high_risk[existing_cols_risk], use_container_width=True)
        else:
            st.warning("Column 'attrition_risk' not found.")

    
    # 😊 High Job Satisfaction
    with col2:
        st.markdown("### 😊 High Job Satisfaction")
        if 'job_satisfaction' in df.columns:
            high_satis = df[df['job_satisfaction'] >= 4]
            option = st.selectbox("Show", options=["All", "Top 10"], key="high_satis")
            if option == "Top 10":
                high_satis = high_satis.sort_values(by='job_satisfaction', ascending=False).head(10)
            display_cols_satis = ['employee_no', 'job_satisfaction', 'performance_score_percent', 'attrition_risk']
            existing_cols_satis = [col for col in display_cols_satis if col in high_satis.columns]
            st.dataframe(high_satis[existing_cols_satis], use_container_width=True)
        else:
            st.warning("Column 'job_satisfaction' not found.")

           

# -------------------------
# Page 2: Prediction Page
# -------------------------
elif page == "Predict Employee Attrition":
    st.markdown("## 🤖 Predict Employee Attrition")

    with st.form("attrition_form"):
        st.markdown("### Enter Employee Details")

        age = st.slider("Age", 18, 60, 30)
        stock_option = st.selectbox("Stock Option Level", [0, 1, 2, 3])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
        income = st.slider("Monthly Income", 1000, 20000, 5000, step=500)
        edu_level = st.selectbox("Education Level", [1, 2, 3, 4, 5])
        job_involvement = st.selectbox("Job Involvement", [1, 2, 3, 4])
        perf_rating = st.selectbox("Performance Rating", [1, 2, 3, 4])
        years_at_company = st.slider("Years at Company", 0, 40, 5)
        satisfaction = st.slider("Overall Satisfaction", 0.0, 1.0, 0.5)

        submit_btn = st.form_submit_button("🔍 Predict Attrition")

        if submit_btn:
            # Manual one-hot encoding for categorical fields
            input_data = pd.DataFrame([{
                'Age': age,
                'StockOptionLevel': stock_option,
                'JobSatisfaction': job_satisfaction,
                'MonthlyIncome': income,
                'Education': edu_level,
                'JobInvolvement': job_involvement,
                'PerformanceRating': perf_rating,
                'YearsAtCompany': years_at_company,
                'OverallSatisfaction': satisfaction,
                'MaritalStatus_Married': 1 if marital_status == 'Married' else 0,
                'MaritalStatus_Single': 1 if marital_status == 'Single' else 0,
                # If 'Divorced' is the dropped category, no need to add a column
            }])

            for col in feature_cols:
                if col not in input_data.columns:
                    input_data[col] = 0

            input_data = input_data[feature_cols]

            prediction = model.predict(input_data)[0]
            proba = model.predict_proba(input_data)[0][1]

            if prediction == 1:
                st.error(f"⚠️ The employee is likely to leave. Risk Score: {proba:.2f}")
            else:
                st.success(f"✅ The employee is likely to stay. Risk Score: {proba:.2f}")
