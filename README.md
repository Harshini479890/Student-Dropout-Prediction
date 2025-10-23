# Student-Dropout-Prediction

Student Dropout Prediction – DropGuard
Overview
DropGuard is an interactive web dashboard built with Streamlit for predicting student dropouts using machine learning. The system is designed for teachers and mentors to proactively identify students at risk, so they can intervene with timely support and counseling.
Features a secure teacher login/signup system, risk-level analytics, and actionable mentoring suggestions.

Features
Teacher/mentor signup and login: Each teacher creates an account to access predictions securely.

CSV Upload & Prediction: Upload student data, and get ML-powered predictions on dropout risk.

Risk Analytics Dashboard:

Color-coded risk levels (High, Moderate, Low)

Mentor/counseling suggestions per student

Quick metrics (risk summary by count)

Downloadable results table

Modern Streamlit interface: Responsive and easy to use with tabs and search/filter functionality.

How to Run This Project

1. Clone the Repository
   
git clone https://github.com/Harshini479890/student-dropout-prediction.git
cd student-dropout-prediction

3. Create a Virtual Environment
   
python -m venv venv

Activate the environment:

Windows:

.\venv\Scripts\activate

Mac/Linux:

source venv/bin/activate

4. Install Required Packages

pip install -r requirements.txt
(If requirements.txt is not present, install manually: pip install streamlit pandas scikit-learn)

4. Run the Streamlit App

streamlit run app.py

6. Using the App
Open the web page shown in the terminal (usually http://localhost:8501).

Sign up to create a teacher/mentor account, then log in.

Upload a CSV file with student features (example columns: "Name", "Marks", "Attendance", "Fees", etc.)

View the predictions, risk levels, and download reports.

Project Explanation

This project aims to help schools and colleges detect students at risk of dropout by leveraging machine learning and user-friendly data dashboards:

Prediction Model: Built using pre-trained ML models (Random Forest, etc.), with scikit-learn.

Data Privacy: Each teacher sees predictions only after authenticating.

Risk Levels: Dropout predictions are categorized as Low/Moderate/High risk according to key columns like marks and attendance.
