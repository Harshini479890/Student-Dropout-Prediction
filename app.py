import streamlit as st
import pandas as pd
import pickle
import os

# ---------- ACCOUNT MANAGEMENT FUNCTIONS ----------
ACCOUNTS_FILE = "teacher_accounts.csv"

def load_accounts():
    if not os.path.exists(ACCOUNTS_FILE):
        df = pd.DataFrame(columns=["username", "password"])
        df.to_csv(ACCOUNTS_FILE, index=False)
    return pd.read_csv(ACCOUNTS_FILE)

def save_account(username, password):
    df = load_accounts()
    if username in df['username'].values:
        return False  # Username taken
    df = pd.concat([df, pd.DataFrame([{"username": username, "password": password}])], ignore_index=True)
    df.to_csv(ACCOUNTS_FILE, index=False)
    return True

def authenticate(username, password):
    df = load_accounts()
    user_row = df[(df['username'] == username) & (df['password'] == password)]
    return not user_row.empty

# ---------- AUTH UI (SIGNUP/LOGIN) ----------
if "logged_in" not in st.session_state:
    st.session_state['logged_in'] = False
    st.session_state['username'] = ""

auth_mode = st.sidebar.selectbox("Authentication", ["Log in", "Sign up"])

if not st.session_state['logged_in']:
    st.title("Teacher Portal")
    if auth_mode == "Sign up":
        st.subheader("Create a teacher account")
        new_user = st.text_input("Choose username")
        new_pass = st.text_input("Choose password", type="password")
        if st.button("Sign up"):
            if not new_user or not new_pass:
                st.warning("Username and password required")
            elif save_account(new_user, new_pass):
                st.success("Account created successfully! You can now log in.")
            else:
                st.error("Username already taken.")
    else:
        st.subheader("Log in")
        user = st.text_input("Username")
        pw = st.text_input("Password", type="password")
        if st.button("Log in"):
            if authenticate(user, pw):
                st.session_state['logged_in'] = True
                st.session_state['username'] = user
                st.success(f"Welcome, {user}!")
            else:
                st.error("Invalid username or password")
    st.stop()

# ----------- MAIN APP -----------
st.markdown(f"<p style='font-size:28px;color:#4B8BBE;font-weight:bold;'>Dashboard: {st.session_state['username']}</p>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📊 Predict Dropouts", "📄 Dashboard"])

# Load model/scaler once logged in
model = pickle.load(open('rf_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

with tab1:
    uploaded_file = st.file_uploader("Upload student data CSV", type="csv")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:", data.head())

        # Encode categorical columns
        for col in data.select_dtypes(include='object').columns:
            data[col] = data[col].astype('category').cat.codes

        # Drop the target/output/dropout column if present
        if 'Dropped_out' in data.columns:
            data_nopred = data.drop('Dropped_out', axis=1)
        else:
            data_nopred = data

        X = scaler.transform(data_nopred)
        predictions = model.predict(X)

        data['Predicted_Dropout'] = predictions

        def risk_label(row):
            if row['Predicted_Dropout'] == 1:
                if 'Marks' in row and row['Marks'] < 50:
                    return "High Risk"
                elif 'Attendance' in row and row['Attendance'] < 60:
                    return "High Risk"
                else:
                    return "Moderate Risk"
            else:
                return "Low Risk"
        data['Risk_Level'] = data.apply(risk_label, axis=1)
        suggestions = {
            'High Risk': "Provide urgent counseling, check for financial/academic support, monitor attendance.",
            'Moderate Risk': "Schedule additional mentoring, offer tutoring, engage parents.",
            'Low Risk': "Monitor progress, send congratulations, maintain engagement."
        }
        data['Mentor_Suggestion'] = data['Risk_Level'].map(suggestions)

        st.write("Full Dashboard with Risks and Mentor Actions:")
        st.dataframe(data)

        # Save processed data to session for dashboard tab
        st.session_state['last_data'] = data

with tab2:
    if 'last_data' in st.session_state:
        data = st.session_state['last_data']
        # Metrics summary
        col1, col2, col3 = st.columns(3)
        col1.metric("High Risk Students", len(data[data['Risk_Level']=='High Risk']))
        col2.metric("Moderate Risk", len(data[data['Risk_Level']=='Moderate Risk']))
        col3.metric("Low Risk", len(data[data['Risk_Level']=='Low Risk']))

        st.markdown("### 🚩 **High Risk Students (Need Immediate Attention):**")
        high_risk = data[data['Risk_Level'] == 'High Risk']
        st.dataframe(high_risk)
        # Download option
        st.download_button(
            label="Download Full Table (CSV)",
            data=data.to_csv(index=False).encode('utf-8'),
            file_name='dropguard_predictions.csv',
            mime='text/csv'
        )
    else:
        st.warning("Please use the Predict Dropouts tab to upload and process data first.")

