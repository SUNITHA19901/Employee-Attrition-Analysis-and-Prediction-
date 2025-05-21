🚀 Employee Attrition Analysis and Prediction

This project aims to analyze patterns and predict employee attrition using machine learning techniques and provide actionable insights through an interactive Streamlit dashboard. By leveraging HR analytics, we empower organizations to better understand why employees leave and take proactive steps to improve retention.

Key Features
 1. Data Analysis Dashboard- High-Risk Employees
 2. Based on attrition prediction probability.
 3. Highly Satisfied Employees
 4. Sorted by job satisfaction levels.
 5. Real-time filtering (Top 10 or All entries).
     
 Attrition Prediction Form
 
 Custom form input to simulate employee details.
 Predicts whether the employee is at risk of attrition.
 Displays a risk score with an intuitive success/warning message.
 
Technologies Used

Frontend/UI: Streamlit
Machine Learning: Random Forest Classifier (Scikit-learn)
Data Processing: Pandas, NumPy
Visualization: Streamlit Tables and Widgets
Model Caching: Streamlit @st.cache_data and @st.cache_resource

Dataset Used
 
The dataset used for this project is a cleaned HR dataset containing
Demographics: Age, Gender, Marital Status
Job-related: Job Satisfaction, Performance Rating, Job Involvement
Company-related: Years at Company, Stock Options
Monthly Income- Target: Attrition (Yes/No)

 Machine Learning Approach
 
 Model: Random Forest Classifier
 Target: Attrition (binary classification)
 Encoding: One-hot and manual encoding
 Evaluation Metric: Predicted probability of attrition

Conclusion

This project provides a practical and intelligent solution for HR teams to reduce attrition risk by offering
insights and predictions based on real employee data. It combines data analysis with machine learning to
support informed decision-making.



